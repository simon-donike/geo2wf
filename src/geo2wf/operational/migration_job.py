"""Durable, quality-gated archive rebuild and optional production cutover.

The pilot must already have passed. --deploy explicitly enables deployment,
publisher reconfiguration, and monitoring; without it the job stops at validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from .common import iso, utc, write_json
from .models import ROOT, version
from .cli import lock

REMOTE = "r2:tcd/explorer/stormsense"


def require_quality(path, phase):
    report = json.loads(path.read_text())
    if not (
        report.get("passed") is True
        and report.get("phase") == phase
        and report.get("candidate_version") == version("nowcast")
    ):
        raise RuntimeError(f"{phase} quality gate failed: {report.get('failures', [])}")
    return report


def fingerprint():
    paths = list((ROOT / "src/geo2wf/operational").glob("*.py"))
    paths += list((ROOT / "src/geo2wf/operational").glob("models*.json"))
    paths += [p for p in (ROOT / "apps/stormsense/src").rglob("*") if p.is_file()]
    paths += [
        ROOT / "apps/stormsense/runner/cycle.sh",
        ROOT / "apps/stormsense/wrangler.jsonc",
    ]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_images(before_root, before_catalog, after_root, after_catalog):
    """Preserve old in-window frames and their bytes; new frames are allowed."""
    after_storms = {s["id"]: s for s in after_catalog["storms"]}
    checked = 0
    for summary in before_catalog["storms"]:
        before = json.loads((before_root / summary["series"]).read_text())
        frames = [
            f
            for f in before.get("imagery", [])
            if f.get("status") == "ready"
            and after_catalog["window"]["start"]
            <= f["time"]
            <= after_catalog["window"]["end"]
        ]
        if not frames:
            continue
        after = json.loads(
            (after_root / after_storms[summary["id"]]["series"]).read_text()
        )
        by_time = {f["time"]: f for f in after.get("imagery", [])}
        for frame in frames:
            if by_time.get(frame["time"]) != frame:
                raise RuntimeError(
                    "Existing image metadata changed: "
                    + summary["id"]
                    + " "
                    + frame["time"]
                )
            keys = [frame["metadata"]] if frame.get("metadata") else []
            for part in frame.get("parts", []):
                keys.extend(
                    part[k]
                    for k in ("image", "sidecar", "preview", "preview_sidecar")
                    if part.get(k)
                )
            for key in keys:
                if (before_root / key).read_bytes() != (after_root / key).read_bytes():
                    raise RuntimeError("Existing image bytes changed: " + key)
            checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--quality-policy", choices=["strict", "wind-priority"], default="strict")
    parser.add_argument(
        "--node-bin", type=Path, help="Directory containing Node >=22 for deployment"
    )
    args = parser.parse_args()
    folder = ROOT / "var/stormsense-migration"
    db = folder / "state.sqlite"
    source = ROOT / "var/stormsense/state.sqlite"
    before_root = ROOT / "var/stormsense/export"
    output = folder / "release"
    python = str(ROOT / ".venv/bin/python")
    manifest = ROOT / "src/geo2wf/operational/models-finetuned.json"
    env = dict(
        os.environ,
        STORMSENSE_MODEL_MANIFEST=str(manifest),
        MPLCONFIGDIR=str(folder / "matplotlib"),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    if args.node_bin:
        env["PATH"] = str(args.node_bin.resolve()) + os.pathsep + env["PATH"]
    base = [
        python,
        "-m",
        "geo2wf.operational.cli",
        "--db",
        str(db),
        "--model-manifest",
        str(manifest),
        "--device",
        "cuda",
    ]
    paused = False
    pointer_may_have_changed = False
    dropin = (
        Path.home() / ".config/systemd/user/stormsense-local.service.d/finetuned.conf"
    )
    old_dropin = dropin.read_text() if dropin.exists() else None
    original_fingerprint = fingerprint()

    def status(stage, **extra):
        write_json(
            folder / "job-status.json",
            {
                "stage": stage,
                "pid": os.getpid(),
                "updated_at": iso(),
                "model_version": version("nowcast"),
                **extra,
            },
        )
        print(stage, extra, flush=True)

    def run(command, cwd=ROOT, extra_env=None):
        subprocess.run(command, cwd=cwd, env={**env, **(extra_env or {})}, check=True)

    def rebuild(command, *extra):
        run([python, "-m", "geo2wf.operational.rebuild", command, *extra,
             "--quality-policy", args.quality_policy])

    def backfill():
        migration = json.loads((folder / "migration.json").read_text())
        command = [
            *base,
            "backfill",
            "--skip-discovery",
            "--workers",
            str(args.workers),
            "--start",
            migration["window"]["start"],
            "--end",
            migration["window"]["end"],
            "--storms",
            *migration["eligible_slots"],
        ]
        run(command)
        run([*command, "--retry-gaps"])

    with lock(str(folder / "migration-job.lock")):
        try:
            require_quality(folder / "pilot-comparison.json", "pilot")
            status("rebuilding_archive")
            backfill()
            rebuild("compare", "--full")
            require_quality(folder / "full-comparison.json", "full")
            if fingerprint() != original_fingerprint:
                raise RuntimeError(
                    "Source changed during rebuild; production remains unchanged"
                )
            status("archive_validated")
            if not args.deploy:
                return
            status("deploying_compatible_frontend")
            run(["npm", "run", "deploy:check"], ROOT / "apps/stormsense")
            run(["npm", "run", "deploy"], ROOT / "apps/stormsense")
            status("reconciling_production")
            run(["systemctl", "--user", "stop", "stormsense-local.timer"])
            paused = True
            run(["systemctl", "--user", "stop", "stormsense-local.service"])
            run(
                [
                    "rclone",
                    "copyto",
                    REMOTE + "/latest.json",
                    str(folder / "rollback-latest.json"),
                ]
            )
            with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src:
                with sqlite3.connect(folder / "before-cutover.sqlite") as dst:
                    src.backup(dst)
            write_json(
                folder / "rollback-config.json",
                {"dropin": str(dropin), "content": old_dropin},
            )
            before_pointer = json.loads((before_root / "latest.json").read_text())
            before_catalog = json.loads(
                (before_root / before_pointer["manifest"]).read_text()
            )
            rebuild("reconcile")
            run([*base, "update", "--workers", str(args.workers), "--imagery-workers", "8"])
            backfill()
            rebuild("compare", "--full")
            require_quality(folder / "full-comparison.json", "full")
            run([*base, "evaluate", "--output", str(folder / "evaluation.json")])
            run([*base, "export", "--output", str(output)])
            pointer = json.loads((output / "latest.json").read_text())
            catalog = json.loads((output / pointer["manifest"]).read_text())
            images = verify_images(before_root, before_catalog, output, catalog)
            write_json(
                folder / "image-verification.json",
                {"unchanged_frames": images, "release": pointer["version"]},
            )
            # Both the original migration baseline and the freshest rollback
            # release protect their assets from subsequent reference-aware GC.
            initial = json.loads((folder / "migration.json").read_text())[
                "baseline_pointer"
            ]
            for pin_pointer in (initial, before_pointer):
                path = before_root / pin_pointer["manifest"].replace(
                    "catalog.json", "pin.json"
                )
                write_json(
                    path, {"reason": "finetuning rollback and image preservation"}
                )
                run(
                    [
                        "rclone",
                        "copyto",
                        str(path),
                        REMOTE + "/" + str(path.relative_to(before_root)),
                    ]
                )
            run([*base, "publish", "--output", str(output), "--stage-only"])
            status("switching_release")
            pointer_may_have_changed = True
            run([*base, "publish", "--output", str(output)])
            run(
                [
                    "npm",
                    "exec",
                    "--",
                    "playwright",
                    "test",
                    "tests/two-hour-predictions.spec.ts",
                ],
                ROOT / "apps/stormsense",
                {"STORMSENSE_BASE_URL": "https://stormsense.hyperalislabs.com"},
            )
            dropin.parent.mkdir(parents=True, exist_ok=True)
            dropin.write_text(
                f"[Service]\nEnvironment=STORMSENSE_DB={db}\nEnvironment=STORMSENSE_EXPORT={output}\nEnvironment=STORMSENSE_MODEL_MANIFEST={manifest}\n"
            )
            run(["systemctl", "--user", "daemon-reload"])
            cutover = iso()
            run(["systemctl", "--user", "start", "stormsense-local.timer"])
            paused = False
            status("monitoring", release=pointer["version"], cutover=cutover)
            cycles = set()
            deadline = time.monotonic() + 7 * 3600
            while len(cycles) < 2:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Two successful even-hour cycles were not observed within seven hours"
                    )
                time.sleep(30)
                with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                    row = conn.execute(
                        "SELECT body FROM status WHERE key='update'"
                    ).fetchone()
                update = json.loads(row[0]) if row else {}
                stamp = update.get("requested_at", "")
                if stamp <= cutover or stamp in cycles or utc(stamp).hour % 2:
                    continue
                result = subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        "stormsense-local.service",
                        "-p",
                        "Result",
                        "--value",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if result.stdout.strip() != "success":
                    continue
                remote_pointer = json.loads(
                    subprocess.check_output(["rclone", "cat", REMOTE + "/latest.json"])
                )
                remote_catalog = json.loads(
                    subprocess.check_output(
                        ["rclone", "cat", REMOTE + "/" + remote_pointer["manifest"]]
                    )
                )
                if remote_catalog["models"]["nowcast"]["version"] != version("nowcast"):
                    continue
                if (
                    remote_catalog.get("source_status", {})
                    .get("update", {})
                    .get("requested_at")
                    != stamp
                ):
                    continue
                cycles.add(stamp)
                status(
                    "monitoring",
                    cycles=sorted(cycles),
                    release=remote_pointer["version"],
                )
            status("complete", cycles=sorted(cycles))
        except BaseException as error:
            if paused or pointer_may_have_changed:
                run(["systemctl", "--user", "stop", "stormsense-local.timer"])
                run(["systemctl", "--user", "stop", "stormsense-local.service"])
                if pointer_may_have_changed:
                    run(
                        [
                            "rclone",
                            "copyto",
                            str(folder / "rollback-latest.json"),
                            REMOTE + "/latest.json",
                        ]
                    )
                if old_dropin is None:
                    dropin.unlink(missing_ok=True)
                else:
                    dropin.write_text(old_dropin)
                run(["systemctl", "--user", "daemon-reload"])
                run(["systemctl", "--user", "start", "stormsense-local.timer"])
            status("stopped", error=str(error), rolled_back=pointer_may_have_changed)
            raise


if __name__ == "__main__":
    main()
