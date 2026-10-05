"""Versioned, resumable GOES crops at exact ATCF best-track fixes.

Run with python -m geo2wf.historical.dataset --help. No operational state writes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import gzip
import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path
import random
import re
import sqlite3
import time

import numpy as np
from aiohttp import ClientError

from .acquisition import acquire_rows
from geo2wf.operational import satellite, tracks
from geo2wf.operational.common import fetch, file_hash, iso, utc, KNOT
from geo2wf.operational.evaluate import training_storms
from geo2wf.operational.models import MANIFEST, resolve_file

VERSION = "historical-storms-v1"
START = "2020-01-01T00:00:00Z"
END = "2025-10-01T00:00:00Z"
HOLDOUT_START = "2025-10-02T00:00:00Z"
HOLDOUT_END = "2026-10-03T00:00:00Z"
TARGETS = ("eye", "rmw", "r34", "r50", "r64")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def parse_labels(body, sid):
    """Keep quadrant zeros; all-zero radii above threshold are ambiguous/missing.

    Below-threshold radii are physically absent and explicitly marked derived zero.
    Missing/negative quadrants invalidate an equivalent-area radius, never impute it.
    """
    raw = gzip.decompress(body) if body.startswith(b"\x1f\x8b") else body
    fixes = tracks.parse_atcf(raw, sid)
    rows = defaultdict(list)
    for line in raw.decode().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 11 and f[4] == "BEST":
            rows[f[2]].append(f)
    for fix in fixes:
        values = {key: None for key in TARGETS}
        origins = {key: "missing" for key in TARGETS}
        stamp = utc(fix["time"]).strftime("%Y%m%d%H")
        for f in rows[stamp]:
            if len(f) > 19:
                try:
                    rmw = float(f[19])
                    if math.isfinite(rmw) and 0 < rmw < 999:
                        values["rmw"], origins["rmw"] = rmw * 1.852, "reported"
                except ValueError:
                    pass
            if len(f) < 17 or f[11] not in {"34", "50", "64"}:
                continue
            try:
                q = [float(v) for v in f[13:17]]
            except ValueError:
                continue
            if f[12] == "AAA":
                q = [q[0]] * 4
            elif f[12] not in {"NEQ", "SEQ", "SWQ", "NWQ"}:
                continue
            if all(math.isfinite(v) and 0 <= v < 999 for v in q) and any(
                v > 0 for v in q
            ):
                key = "r" + f[11]
                values[key] = math.sqrt(sum(v * v for v in q) / 4) * 1.852
                origins[key] = "reported"
        for threshold in (34, 50, 64):
            key = f"r{threshold}"
            if fix["wind_ms"] is not None and fix["wind_ms"] < threshold * KNOT:
                values[key], origins[key] = 0.0, "below_intensity_threshold"
        fix["labels"] = {"vmax": fix["wind_ms"], **values}
        fix["label_valid"] = {k: v is not None for k, v in fix["labels"].items()}
        fix["label_origin"] = origins
    return fixes


def split_storms(storm_ids, excluded=()):
    groups = defaultdict(list)
    for sid in sorted(set(storm_ids) - set(excluded)):
        groups[(sid[-4:], sid[:2])].append(sid)
    result = {}
    for group, ids in sorted(groups.items()):
        rng = random.Random("42:" + ":".join(group))
        rng.shuffle(ids)
        nval = max(1, round(len(ids) * 0.15)) if len(ids) > 1 else 0
        result.update(
            {sid: ("val" if i < nval else "train") for i, sid in enumerate(ids)}
        )
    return result


def research_exclusions():
    paths = sorted(
        set(Path("data").glob("*/val/manifest.csv"))
        | set(Path("data").glob("*/test/manifest.csv"))
    )
    if not paths:
        raise RuntimeError(
            "No research validation/test manifests found; cannot establish exclusions"
        )
    ids, sources = set(), []
    for path in paths:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "storm_id" not in rows[0]:
            raise ValueError(f"Missing research storm identities: {path}")
        for row in rows:
            sid = row["storm_id"].upper()
            if not tracks.STORM_ID.fullmatch(sid):
                # Other basins are outside this dataset, but unknown IDs are unsafe.
                if not re.fullmatch(r"[A-Z]{2}\d{6}", sid):
                    raise ValueError(f"Unresolved research storm identity: {sid}")
            ids.add(sid)
        sources.append({"path": str(path), "sha256": file_hash(path)})
    return ids, sources


def discover(root, state, model_root):
    manifest = root / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    with sqlite3.connect(f"file:{Path(state).resolve()}?mode=ro", uri=True) as db:
        storms = [json.loads(r[0]) for r in db.execute("SELECT body FROM storms")]
    holdout = {
        s["id"]
        for s in storms
        if any(HOLDOUT_START <= f["time"] < HOLDOUT_END for f in s.get("track", []))
    }
    if not holdout:
        raise ValueError("Website holdout is empty")
    research, sources = research_exclusions()
    pretrained = training_storms(model_root)
    if not pretrained or any(
        not re.fullmatch(r"[A-Z]{2}\d{6}", sid) for sid in pretrained
    ):
        raise ValueError(
            "Unresolved pretraining storm identities; cannot label unseen storms"
        )
    stats = resolve_file(MANIFEST["models"]["nowcast"]["stats"], model_root)
    if file_hash(stats) != MANIFEST["models"]["nowcast"]["stats_sha256"]:
        raise ValueError("Normalization checksum mismatch")
    (root / "stats.json").write_bytes(stats.read_bytes())
    fixes, snapshots, failures = [], [], []
    for year in range(2020, 2027):
        try:
            urls = tracks.track_urls(year)
        except Exception as exc:
            failures.append({"year": year, "error": str(exc)})
            continue
        for sid, url in urls:
            if year == 2026 and sid not in holdout:
                continue
            path = root / "sources" / f"{sid}.dat"
            try:
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    body = fetch(url)
                    temporary = path.with_suffix(".tmp")
                    temporary.write_bytes(body)
                    temporary.replace(path)
                body = path.read_bytes()
                parsed = parse_labels(body, sid)
                # Also exclude archive storms crossing the holdout even if website missed them.
                if any(HOLDOUT_START <= f["time"] < HOLDOUT_END for f in parsed):
                    holdout.add(sid)
                snapshots.append(
                    {
                        "storm_id": sid,
                        "url": url,
                        "path": str(path.relative_to(root)),
                        "sha256": file_hash(path),
                        "retrieved_at": iso(
                            datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                        ),
                    }
                )
                by_time = {f["time"]: f for f in parsed}
                for fix in parsed:
                    historical = START <= fix["time"] < END
                    evaluation = (
                        sid in holdout and HOLDOUT_START <= fix["time"] < HOLDOUT_END
                    )
                    if not (historical or evaluation) or fix["wind_ms"] is None:
                        continue
                    previous = by_time.get(iso(utc(fix["time"]) - timedelta(hours=24)))
                    ri = None
                    if previous and previous["wind_ms"] is not None:
                        window = [
                            f
                            for f in parsed
                            if previous["time"] <= f["time"] <= fix["time"]
                        ]
                        if all(
                            (utc(b["time"]) - utc(a["time"])).total_seconds() <= 21600
                            for a, b in zip(window, window[1:])
                        ):
                            ri = (
                                fix["wind_ms"] - previous["wind_ms"] >= 30 * KNOT - 1e-6
                            )
                    fixes.append(
                        {
                            **fix,
                            "storm_id": sid,
                            "basin": sid[:2],
                            "source_url": url,
                            "track_sha256": file_hash(path),
                            "is_ri": ri,
                            "sample_id": sid
                            + "_"
                            + utc(fix["time"]).strftime("%Y%m%dT%H%M%SZ"),
                        }
                    )
            except Exception as exc:
                failures.append({"storm_id": sid, "url": url, "error": str(exc)})
    missing_holdout = holdout - {s["storm_id"] for s in snapshots}
    failures.extend(
        {"storm_id": sid, "error": "Holdout source missing"}
        for sid in sorted(missing_holdout)
    )
    write_json(root / "discovery.json", {"failures": failures, "sources": snapshots})
    if failures:
        raise RuntimeError(
            f"Discovery incomplete: {len(failures)} failures; rerun to resume"
        )
    assignments = split_storms({f["storm_id"] for f in fixes}, holdout | research)
    samples, exclusions = [], Counter()
    for fix in fixes:
        sid = fix["storm_id"]
        if sid in holdout:
            if not HOLDOUT_START <= fix["time"] < HOLDOUT_END:
                exclusions["holdout_storm_earlier_fix"] += 1
                continue
            split = "test"
        elif sid in research:
            exclusions["research_validation_test"] += 1
            continue
        else:
            split = assignments[sid]
        samples.append(
            {
                **fix,
                "split": split,
                "pretraining_membership": (
                    "seen"
                    if sid in pretrained
                    else ("unseen" if tracks.STORM_ID.fullmatch(sid) else "unresolved")
                ),
            }
        )
    result = {
        "version": VERSION,
        "created_at": iso(),
        "start": START,
        "end_exclusive": END,
        "holdout_start": HOLDOUT_START,
        "holdout_end_exclusive": HOLDOUT_END,
        "seed": 42,
        "holdout_storms": sorted(holdout),
        "research_excluded_storms": sorted(research),
        "research_manifests": sources,
        "excluded_fix_counts": dict(exclusions),
        "model": MANIFEST,
        "stats_sha256": file_hash(root / "stats.json"),
        "sources": snapshots,
        "samples": sorted(samples, key=lambda r: (r["time"], r["storm_id"])),
    }
    write_json(manifest, result)
    return result


def database(root):
    db = sqlite3.connect(root / "collection.sqlite")
    db.execute(
        "CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY, status TEXT, body TEXT)"
    )
    return db


def collect_one(root, row):
    started = time.monotonic()
    try:
        array, valid, source = satellite.acquire(row["time"], row)
        path = root / "crops" / row["storm_id"] / (row["sample_id"] + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, array=array.astype(np.float32), valid=valid.astype(bool)
            )
        temporary.replace(path)
        return "ready", {
            "path": str(path.relative_to(root)),
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
            "source": source,
            "seconds": time.monotonic() - started,
        }
    except satellite.DataGap as exc:
        return "gap", {"reason": exc.reason, "detail": str(exc), "retryable": True}
    except (OSError, TimeoutError, ClientError) as exc:
        return "failed", {
            "reason": type(exc).__name__,
            "detail": str(exc),
            "retryable": True,
        }


def collect(
    root, manifest, split="historical", limit=None, workers=4, retry=False, pilot=False
):
    with database(root) as db:
        existing = {
            r[0]: r[1] for r in db.execute("SELECT id,status FROM observations")
        }
        rows = [
            r
            for r in manifest["samples"]
            if (r["split"] == "test") == (split == "test")
            and (
                r["sample_id"] not in existing
                or (retry and existing[r["sample_id"]] != "ready")
            )
        ]
        if pilot:
            groups = defaultdict(list)
            for r in rows:
                sat = satellite.satellite_order(r["time"], r["lat"], r["lon"])[0]
                groups[(r["time"][:4], r["basin"], sat)].append(r)
            rows = [v[len(v) // 2] for _, v in sorted(groups.items())]
        if limit is not None:
            rows = rows[:limit]
        completed = 0

        def heartbeat(active):
            write_json(
                root / "collection-progress.json",
                {
                    "updated_at": iso(),
                    "split": split,
                    "completed_this_run": completed,
                    "total_this_run": len(rows),
                    "active": active,
                    "task_timeout_seconds": 600,
                },
            )

        for row, (status, body) in acquire_rows(
            root, rows, workers=workers, heartbeat=heartbeat
        ):
            body.update({"attempted_at": iso(), "version": VERSION})
            with db:
                db.execute(
                    "INSERT OR REPLACE INTO observations VALUES (?,?,?)",
                    (row["sample_id"], status, json.dumps(body, allow_nan=False)),
                )
            completed += 1
            print(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "status": status,
                        "progress": completed,
                        "total": len(rows),
                    }
                ),
                flush=True,
            )
        heartbeat([])
    return coverage(root, manifest)


def coverage(root, manifest):
    with database(root) as db:
        observations = {
            r[0]: (r[1], json.loads(r[2]))
            for r in db.execute("SELECT * FROM observations")
        }
    counts, groups, labels = Counter(), defaultdict(Counter), Counter()
    total_bytes, seconds, errors = 0, 0.0, Counter()
    for row in manifest["samples"]:
        status, body = observations.get(row["sample_id"], ("pending", {}))
        key = row["split"]
        counts[key + "/" + status] += 1
        groups[f"{key}/{row['time'][:4]}/{row['basin']}"][status] += 1
        if status == "ready":
            total_bytes += body["bytes"]
            seconds += body["seconds"]
            labels.update(f"{key}/{k}" for k, v in row["label_valid"].items() if v)
        elif status != "pending":
            errors[body.get("reason", status)] += 1
    report = {
        "version": VERSION,
        "manifest_sha256": file_hash(root / "manifest.json"),
        "counts": dict(counts),
        "groups": dict(groups),
        "valid_labels": dict(labels),
        "bytes": total_bytes,
        "summed_acquisition_seconds": seconds,
        "gap_reasons": dict(errors),
        "storms": {
            s: len({r["storm_id"] for r in manifest["samples"] if r["split"] == s})
            for s in ("train", "val", "test")
        },
        "holdout_pretraining_audit": {
            membership: {
                "storm_ids": sorted(
                    {
                        r["storm_id"]
                        for r in manifest["samples"]
                        if r["split"] == "test"
                        and r.get("pretraining_membership") == membership
                    }
                ),
                "observations": sum(
                    r["split"] == "test"
                    and r.get("pretraining_membership") == membership
                    for r in manifest["samples"]
                ),
            }
            for membership in ("seen", "unseen", "unresolved")
        },
    }
    write_json(root / "coverage.json", report)
    return report


def verify(root, manifest, split="historical", allow_pending=False):
    errors, checked = [], 0
    with database(root) as db:
        observations = {
            r[0]: (r[1], json.loads(r[2]))
            for r in db.execute("SELECT * FROM observations")
        }
    if file_hash(root / "stats.json") != manifest["stats_sha256"]:
        errors.append("stats checksum")
    for source in manifest["sources"]:
        if file_hash(root / source["path"]) != source["sha256"]:
            errors.append("source checksum: " + source["storm_id"])
    for row in manifest["samples"]:
        if (row["split"] == "test") != (split == "test"):
            continue
        status, body = observations.get(row["sample_id"], ("pending", {}))
        if status == "pending":
            if not allow_pending:
                errors.append("pending: " + row["sample_id"])
            continue
        if status == "failed":
            errors.append("failed: " + row["sample_id"] + ": " + body.get("detail", ""))
            continue
        if status != "ready":
            continue
        try:
            path = root / body["path"]
            assert file_hash(path) == body["sha256"], "checksum"
            with np.load(path, allow_pickle=False) as data:
                a, v = data["array"], data["valid"]
                assert a.shape == (10, 256, 256) and a.dtype == np.float32, "array"
                assert v.shape == (256, 256) and v.dtype == bool, "mask"
                assert np.isfinite(a[:, v]).all(), "finite valid pixels"
                assert (
                    v[32:224, 32:224].mean() >= 0.9 and v[126:130, 126:130].all()
                ), "coverage"
            age = (utc(row["time"]) - utc(body["source"]["end"])).total_seconds()
            assert 0 <= age <= 1800, "scan timing"
            if row["split"] == "train":
                assert (
                    row["storm_id"]
                    not in manifest["holdout_storms"]
                    + manifest["research_excluded_storms"]
                ), "leakage"
            checked += 1
        except (AssertionError, OSError, ValueError) as exc:
            errors.append(row["sample_id"] + ": " + str(exc))
    report = {
        "checked": checked,
        "errors": errors,
        "manifest_sha256": file_hash(root / "manifest.json"),
        "split": split,
        "allow_pending": allow_pending,
    }
    write_json(root / f"verification-{split}.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["discover", "collect", "coverage", "verify"])
    p.add_argument("--root", type=Path, default=Path("data/historical_storms_v1"))
    p.add_argument("--state", default="var/stormsense/state.sqlite")
    p.add_argument("--model-root", default="downloads/models")
    p.add_argument("--split", choices=["historical", "test"], default="historical")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--retry-gaps", action="store_true")
    p.add_argument(
        "--allow-pending",
        action="store_true",
        help="Verify pilot crops without declaring the dataset complete",
    )
    args = p.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        p.error("workers and limit must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    writing = args.command in {"discover", "collect"}
    with (args.root / ".lock").open("w") if writing else nullcontext() as lock:
        if writing:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = (
            discover(args.root, args.state, args.model_root)
            if args.command == "discover"
            else json.loads((args.root / "manifest.json").read_text())
        )
        if args.command == "collect":
            result = collect(
                args.root,
                manifest,
                args.split,
                args.limit,
                args.workers,
                args.retry_gaps,
                args.pilot,
            )
        elif args.command == "verify":
            result = verify(args.root, manifest, args.split, args.allow_pending)
        else:
            result = coverage(args.root, manifest)
        print(json.dumps(result, indent=2))
        if args.command == "verify" and result["errors"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
