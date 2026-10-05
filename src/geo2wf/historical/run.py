"""Resume collection, verify the frozen cohort, then run both experiments.

The runner waits for an idle GPU and never stops other GPU workloads.

From the repository root, with the operational dependencies installed:
    python -m geo2wf.historical.dataset discover
    python -m geo2wf.historical.dataset collect --pilot --workers 3
    python -m geo2wf.historical.dataset verify --allow-pending
    python -m geo2wf.historical.run

Observe progress using dataset coverage and logs/historical/status.json.
Rerun the same command to resume acquisition or a saved last.ckpt. Historical
ready samples are frozen once training begins. Website-period evaluation runs
only after both fits complete. Reconstructed fields and forecasts are unvalidated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .dataset import write_json
from geo2wf.operational.common import iso


def free_gpu():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        index, memory, utilization = [int(x.strip()) for x in line.split(",")]
        if memory < 1000 and utilization < 10:
            return str(index)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/historical_storms_v1"))
    p.add_argument("--output", type=Path, default=Path("logs/historical"))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = (args.output / ".runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(stage, **kw):
        write_json(
            args.output / "status.json",
            {"stage": stage, "pid": os.getpid(), "updated_at": iso(), **kw},
        )
        print(stage, flush=True)

    def execute(module, arguments, env=None):
        subprocess.run([sys.executable, "-m", module, *arguments], check=True, env=env)

    base = ["--root", str(args.root)]
    data = "geo2wf.historical.dataset"
    try:
        if not (args.output / "heads" / "config.json").exists():
            status("collecting_historical")
            execute(data, ["collect", *base, "--workers", str(args.workers)])
            status("retrying_historical_gaps")
            execute(
                data, ["collect", *base, "--workers", str(args.workers), "--retry-gaps"]
            )
        status("verifying_historical")
        execute(data, ["verify", *base])
        for mode in ("heads", "encoder"):
            run = args.output / mode
            if (run / "result.json").exists():
                continue
            status("waiting_for_idle_gpu", experiment=mode)
            gpu = free_gpu()
            while gpu is None:
                time.sleep(60)
                gpu = free_gpu()
            status("training", experiment=mode, gpu=gpu)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            command = [
                "train",
                *base,
                "--output",
                str(run),
                "--mode",
                mode,
                "--device",
                "cuda",
            ]
            if (run / "checkpoints/last.ckpt").exists():
                command.append("--resume")
            execute("geo2wf.historical.training", command, env)
        status("collecting_evaluation")
        execute(
            data, ["collect", *base, "--split", "test", "--workers", str(args.workers)]
        )
        execute(
            data,
            [
                "collect",
                *base,
                "--split",
                "test",
                "--workers",
                str(args.workers),
                "--retry-gaps",
            ],
        )
        execute(data, ["verify", *base, "--split", "test"])
        status("evaluating")
        execute(
            "geo2wf.historical.training",
            [
                "evaluate",
                *base,
                "--output",
                str(args.output / "evaluation"),
                "--heads-run",
                str(args.output / "heads"),
                "--encoder-run",
                str(args.output / "encoder"),
                "--device",
                "cpu",
            ],
        )
        status("complete")
    except BaseException as exc:
        status("stopped", error=str(exc))
        raise


if __name__ == "__main__":
    main()
