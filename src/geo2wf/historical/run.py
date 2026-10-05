"""Collect and verify historical observations; training is always started manually.

Run from the repository root:
    python -m geo2wf.historical.run --workers 8

This resumes an interrupted collection, retries gaps once, verifies the dataset,
then exits. If a collector is already running, it waits for its lock first.
It never starts training, evaluation, or GPU work.

After reviewing coverage and verification, manually run either experiment:
    python -m geo2wf.historical.training train --mode heads --output logs/historical/heads
    python -m geo2wf.historical.training train --mode encoder --output logs/historical/encoder
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from .dataset import write_json
from geo2wf.operational.common import iso


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/historical_storms_v1"))
    p.add_argument("--output", type=Path, default=Path("logs/historical"))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def status(stage, **kw):
            write_json(
                args.output / "status.json",
                {
                    "stage": stage,
                    "pid": os.getpid(),
                    "updated_at": iso(),
                    "automatic_training": False,
                    **kw,
                },
            )
            print(stage, flush=True)

        def execute(command, *extra):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "geo2wf.historical.dataset",
                    command,
                    "--root",
                    str(args.root),
                    *extra,
                ],
                check=True,
            )

        try:
            status("waiting_for_active_collection")
            with (args.root / ".lock").open("a") as collection_lock:
                fcntl.flock(collection_lock, fcntl.LOCK_EX)
            # Existing trained cohorts must not change when this runner is rerun.
            if not (args.output / "heads/config.json").exists():
                status("collecting_historical")
                execute("collect", "--workers", str(args.workers))
                status("retrying_historical_gaps")
                execute("collect", "--workers", str(args.workers), "--retry-gaps")
            status("verifying_historical")
            execute("verify")
            status("dataset_ready")
        except BaseException as exc:
            status("stopped", error=str(exc))
            raise


if __name__ == "__main__":
    main()
