"""Portable, finite commands. Nothing installs or activates a scheduler."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import timedelta
import fcntl
import json
import logging
import time
from pathlib import Path

from .common import hour, iso, utc, year_before, write_json
from .store import Store


@contextmanager
def lock(path, wait_seconds=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "An update/backfill is already running for this database; retry after it yields"
                    ) from None
                time.sleep(1)
        yield


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=Path("var/stormsense/state.sqlite"))
    p.add_argument("--model-root", type=Path, default=Path("downloads/models"))
    p.add_argument("--device", default=None)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("bootstrap")
    sub.add_parser("discover")
    cmd = sub.add_parser(
        "update", help="Update live estimates and reconcile missed archive hours"
    )
    cmd.add_argument("--workers", type=int, default=4)
    sub.add_parser("verify")
    for name in ("discover-history", "backfill", "coverage"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--start", default=None)
        cmd.add_argument("--end", default=None)
        if name == "backfill":
            cmd.add_argument("--workers", type=int, default=4)
            cmd.add_argument("--limit", type=int)
            cmd.add_argument("--storms", nargs="+")
            cmd.add_argument("--retry-gaps", action="store_true")
            cmd.add_argument("--skip-discovery", action="store_true")
    cmd = sub.add_parser("evaluate")
    cmd.add_argument("--start")
    cmd.add_argument("--end")
    cmd.add_argument(
        "--output", type=Path, default=Path("var/stormsense/evaluation.json")
    )
    cmd = sub.add_parser(
        "imagery",
        help="Archive easy-to-obtain hourly GeoColor WebP crops; no raw-data fallback",
    )
    cmd.add_argument("--start")
    cmd.add_argument("--end")
    cmd.add_argument("--recent-hours", type=int)
    cmd.add_argument("--workers", type=int, default=4)
    cmd.add_argument("--limit", type=int)
    cmd.add_argument("--storms", nargs="+")
    cmd.add_argument("--active-only", action="store_true")
    cmd.add_argument("--retry-gaps", action="store_true")
    cmd = sub.add_parser("export")
    cmd.add_argument("--output", type=Path, default=Path("var/stormsense/export"))
    cmd.add_argument("--start")
    cmd.add_argument("--end")
    cmd = sub.add_parser("publish")
    cmd.add_argument("--output", type=Path, default=Path("var/stormsense/export"))
    cmd.add_argument(
        "--stage-only",
        action="store_true",
        help="Upload and verify immutable assets without advancing latest.json",
    )
    cmd = sub.add_parser("retain")
    cmd.add_argument("--apply", action="store_true")
    cmd.add_argument(
        "--remote",
        action="store_true",
        help="Include reference-aware remote R2 retention",
    )
    cmd.add_argument("--output", type=Path, default=Path("var/stormsense/export"))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    from .models import Models, bootstrap
    from .pipeline import discover, discover_history, update, backfill
    from .export import export_release, publish, coverage, prune_local, prune_remote
    from .evaluate import evaluate

    if args.command == "bootstrap":
        bootstrap(args.model_root)
        return
    store = Store(args.db)
    try:
        start = (
            utc(getattr(args, "start", None))
            if getattr(args, "start", None)
            else year_before()
        )
        end = utc(getattr(args, "end", None))
        result = None
        if args.command == "discover":
            result = discover(store)
        elif args.command == "discover-history":
            discover_history(store, start, end)
        elif args.command == "update":
            if not 1 <= args.workers <= 32:
                raise ValueError("workers must be between 1 and 32")
            request = Path(str(args.db) + ".update-requested")
            request.touch()
            acquired = False
            try:
                with lock(str(args.db) + ".lock", wait_seconds=180):
                    # Our own priority marker must not stop our catch-up job.
                    request.unlink(missing_ok=True)
                    acquired = True
                    result = update(
                        store,
                        Models(args.model_root, args.device),
                        workers=args.workers,
                        model_root=args.model_root,
                    )
            finally:
                if not acquired:
                    request.unlink(missing_ok=True)
        elif args.command == "backfill":
            if not 1 <= args.workers <= 32:
                raise ValueError("workers must be between 1 and 32")
            with lock(str(args.db) + ".lock"):
                if not args.skip_discovery:
                    discover_history(store, start - timedelta(hours=12), end)
                result = backfill(
                    store,
                    Models(args.model_root, args.device),
                    start - timedelta(hours=12),
                    end,
                    args.workers,
                    args.limit,
                    args.retry_gaps,
                    args.storms,
                    args.model_root,
                )
        elif args.command == "coverage":
            result = coverage(store, start, end)
            write_json(args.db.parent / "coverage.json", result)
        elif args.command == "imagery":
            from .geocolor import backfill_images

            if not 1 <= args.workers <= 8 or (
                args.limit is not None and args.limit < 1
            ):
                raise ValueError(
                    "imagery workers must be 1–8 and limit must be positive"
                )
            if args.recent_hours is not None:
                if args.recent_hours < 1:
                    raise ValueError("recent-hours must be positive")
                start = hour(end) - timedelta(hours=args.recent_hours)
            with lock(str(args.db) + ".lock"):
                result = backfill_images(
                    store,
                    start,
                    end,
                    args.workers,
                    args.limit,
                    args.retry_gaps,
                    args.storms,
                    args.active_only,
                )
        elif args.command == "evaluate":
            result = evaluate(store, args.model_root, start, end)
            write_json(args.output, result)
        elif args.command == "verify":
            from .verify import verify

            result = verify(store, args.model_root, args.device)
            write_json(args.db.parent / "verification.json", result)
        elif args.command == "export":
            with lock(str(args.output) + ".publish.lock"):
                catalog = export_release(store, args.output, start, end)
            result = {
                "release": catalog["release"],
                "storms": len(catalog["storms"]),
                "coverage": catalog["coverage"],
            }
        elif args.command == "publish":
            with lock(str(args.output) + ".publish.lock"):
                publish(args.output, advance_pointer=not args.stage_only)
        elif args.command == "retain":
            cutoff = hour(year_before()) - timedelta(hours=12)
            result = {"cutoff": iso(cutoff), "apply": args.apply}
            from .geocolor import retain_assets

            with lock(str(args.db) + ".lock"):
                if args.apply:
                    result["database"] = store.retain(cutoff)
                result["obsolete_imagery"] = retain_assets(store, apply=args.apply)
            with lock(str(args.output) + ".publish.lock"):
                result["obsolete_files"] = prune_local(args.output, apply=args.apply)
                if args.remote:
                    result["obsolete_remote_files"] = prune_remote(apply=args.apply)
        if result is not None:
            print(json.dumps(result, indent=2, allow_nan=False))
    finally:
        store.close()


if __name__ == "__main__":
    main()
