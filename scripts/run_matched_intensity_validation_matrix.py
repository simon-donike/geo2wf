#!/usr/bin/env python3
"""Run the complete matched IBTrACS/SAR validation matrix on two GPUs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-root",
        type=Path,
        default=Path("data/geotiff/geo_sar_10bands_era5_v2_pmw"),
    )
    parser.add_argument(
        "--ibtracs-file",
        type=Path,
        default=Path("data/IBTrACs/ibtracs.ALL.list.v04r01.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--joint-gpu", default="0")
    parser.add_argument("--pipeline-gpu", default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--correction-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--joint-epochs", type=int, default=None)
    parser.add_argument("--unet-epochs", type=int, default=None)
    parser.add_argument("--correction-epochs", type=int, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--wandb-project", default="geo2wf")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.joint_gpu == args.pipeline_gpu:
        parser.error("--joint-gpu and --pipeline-gpu must identify different GPUs")
    if args.bootstrap_repetitions < 0:
        parser.error("--bootstrap-repetitions must be non-negative")
    return args


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run(command: Sequence[str], *, log_path: Path) -> None:
    print(" ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            list(command),
            cwd=ROOT,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )


def _workflow_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError(f"comparison workflow did not complete: {path}")
    return payload


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    paired_root = args.paired_root.expanduser().resolve()
    ibtracs_file = args.ibtracs_file.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (ROOT / "logs" / "intensity-target-matrix" / _timestamp()).resolve()
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"matrix output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    group = args.wandb_group or f"matched-intensity-validation-{_timestamp()}"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "paired_root": str(paired_root),
        "ibtracs_file": str(ibtracs_file),
        "wandb_group": group,
        "runs": [],
    }
    manifest_path = output_root / "matrix-workflow.json"
    _write_json(manifest, manifest_path)

    common = [
        sys.executable,
        str(ROOT / "scripts" / "run_intensity_model_comparison.py"),
        "--paired-root",
        str(paired_root),
        "--ibtracs-file",
        str(ibtracs_file),
        "--joint-gpu",
        str(args.joint_gpu),
        "--pipeline-gpu",
        str(args.pipeline_gpu),
        "--split",
        "val",
        "--seed",
        str(args.seed),
        "--image-batch-size",
        str(args.image_batch_size),
        "--correction-batch-size",
        str(args.correction_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--bootstrap-repetitions",
        str(args.bootstrap_repetitions),
        "--wandb-group",
        group,
    ]
    for name in ("joint_epochs", "unet_epochs", "correction_epochs"):
        value = getattr(args, name)
        if value is not None:
            common.extend([f"--{name.replace('_', '-')}", str(value)])
    if args.smoke_test:
        common.append("--smoke-test")
    if args.disable_wandb:
        common.append("--disable-wandb")

    result_paths: list[Path] = []
    try:
        for era5 in ("with", "without"):
            shared_unet: dict[str, str] | None = None
            for target in ("ibtracs", "sar_robust_peak"):
                run_root = output_root / f"{era5}-era5" / target
                run_root.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    *common,
                    "--era5",
                    era5,
                    "--intensity-target-source",
                    target,
                    "--output-root",
                    str(run_root),
                ]
                if shared_unet is not None:
                    command.extend(
                        [
                            "--unet-checkpoint",
                            shared_unet["checkpoint"],
                            "--unet-config",
                            shared_unet["config"],
                        ]
                    )
                _run(command, log_path=output_root / f"{era5}-{target}.log")
                workflow_path = run_root / "workflow.json"
                workflow = _workflow_payload(workflow_path)
                if shared_unet is None:
                    shared_unet = {
                        "checkpoint": workflow["artifacts"]["unet_checkpoint"],
                        "config": workflow["artifacts"]["unet_config"],
                    }
                elif (
                    workflow["artifacts"]["unet_checkpoint"]
                    != shared_unet["checkpoint"]
                ):
                    raise RuntimeError(f"{era5} target runs did not share one U-Net")
                result_path = Path(workflow["artifacts"]["comparison_json"])
                result_paths.append(result_path)
                manifest["runs"].append(
                    {
                        "era5": era5,
                        "target": target,
                        "workflow": str(workflow_path),
                        "result": str(result_path),
                        "unet_checkpoint": shared_unet["checkpoint"],
                    }
                )
                _write_json(manifest, manifest_path)

        divergence_path = output_root / "sar-ibtracs-divergence.json"
        divergence_command = [
            sys.executable,
            str(ROOT / "scripts" / "analyze_sar_ibtracs_divergence.py"),
            "--paired-root",
            str(paired_root),
            "--ibtracs-file",
            str(ibtracs_file),
            "--output",
            str(divergence_path),
            "--bootstrap-repetitions",
            str(args.bootstrap_repetitions),
            "--bootstrap-seed",
            str(args.seed),
        ]
        _run(divergence_command, log_path=output_root / "divergence.log")

        consolidated_path = output_root / "matched-validation.json"
        combine_command = [
            sys.executable,
            str(ROOT / "scripts" / "combine_intensity_target_validation.py"),
            *(item for path in result_paths for item in ("--result", str(path))),
            "--divergence",
            str(divergence_path),
            "--output",
            str(consolidated_path),
            "--wandb-project",
            args.wandb_project,
            "--wandb-group",
            group,
        ]
        if args.disable_wandb:
            combine_command.append("--disable-wandb")
        _run(combine_command, log_path=output_root / "combine.log")
        manifest.update(
            {
                "status": "completed",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "artifacts": {
                    "divergence": str(divergence_path),
                    "consolidated_validation": str(consolidated_path),
                    "metrics_csv": str(consolidated_path.with_suffix(".csv")),
                    "predictions_csv": str(
                        consolidated_path.with_name(
                            consolidated_path.stem + "-predictions.csv"
                        )
                    ),
                    "markdown": str(consolidated_path.with_suffix(".md")),
                },
            }
        )
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _write_json(manifest, manifest_path)
        raise
    _write_json(manifest, manifest_path)
    return manifest


def main() -> None:
    run_matrix(parse_args())


if __name__ == "__main__":
    main()
