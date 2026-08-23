#!/usr/bin/env python3
"""Run the reproducible two-GPU intensity-model comparison workflow."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any, Sequence, TextIO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTECTED_STORMS = ("AL082025", "EP112025", "EP182023")


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
    parser.add_argument(
        "--era5",
        choices=("with", "without"),
        default="with",
        help="Run the matched-cohort comparison with or without ERA5 features.",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--intensity-target-source",
        choices=("ibtracs", "sar_robust_peak"),
        default="ibtracs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--correction-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--joint-epochs", type=int, default=None)
    parser.add_argument("--unet-epochs", type=int, default=None)
    parser.add_argument("--correction-epochs", type=int, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument(
        "--protected-storm",
        action="append",
        dest="protected_storms",
        default=None,
        metavar="ATCF_ID",
        help=(
            "Storm that must not occur in the training split. Repeat as needed; "
            "defaults to the inference-folder Humberto 2025, Kiko 2025, and "
            "Otis 2023 storms."
        ),
    )
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use one epoch and one train/validation batch per training stage.",
    )
    parser.add_argument(
        "--joint-checkpoint",
        type=Path,
        default=None,
        help="Reuse a joint checkpoint instead of retraining it.",
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=None,
        help="Reuse a field-only U-Net checkpoint instead of retraining it.",
    )
    parser.add_argument(
        "--unet-config",
        type=Path,
        default=None,
        help="Resolved config for --unet-checkpoint; required when it is reused.",
    )
    parser.add_argument(
        "--correction-checkpoint",
        type=Path,
        default=None,
        help="Reuse a correction checkpoint instead of retraining it.",
    )
    args = parser.parse_args()
    for name in ("image_batch_size", "correction_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    for name in ("joint_epochs", "unet_epochs", "correction_epochs"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.bootstrap_repetitions < 0:
        parser.error("--bootstrap-repetitions must be non-negative")
    for name in ("joint_gpu", "pipeline_gpu"):
        if not str(getattr(args, name)).isdigit():
            parser.error(f"--{name.replace('_', '-')} must be a numeric GPU index")
    args.protected_storms = tuple(
        dict.fromkeys(
            str(storm).strip().upper()
            for storm in (args.protected_storms or DEFAULT_PROTECTED_STORMS)
        )
    )
    if not all(args.protected_storms):
        parser.error("--protected-storm must be a non-empty ATCF ID")
    if args.unet_checkpoint is not None and args.unet_config is None:
        parser.error("--unet-config is required with --unet-checkpoint")
    if args.unet_config is not None and args.unet_checkpoint is None:
        parser.error("--unet-config is only valid with --unet-checkpoint")
    if (
        args.joint_checkpoint is None
        and args.unet_checkpoint is None
        and args.joint_gpu == args.pipeline_gpu
    ):
        parser.error("concurrent joint and U-Net training require different GPUs")
    return args


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def _training_command(
    experiment: str,
    stage_root: Path,
    gpu: str,
    overrides: Sequence[str],
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "geo2wf.training",
        f"experiment={experiment}",
        "trainer.accelerator=gpu",
        f"trainer.devices=[{gpu}]",
        f"trainer.default_root_dir={stage_root}",
        *overrides,
    ]


class RunningStage:
    def __init__(
        self,
        name: str,
        command: Sequence[str],
        *,
        environment: MappingLike,
        log_path: Path,
    ) -> None:
        self.name = name
        self.command = list(command)
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO = log_path.open("w", encoding="utf-8")
        print(f"[{name}] {_command_text(command)}")
        print(f"[{name}] log: {log_path}")
        self.process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=dict(environment),
            stdout=self._stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    def wait(self) -> None:
        return_code = self.process.wait()
        self._stream.close()
        if return_code:
            raise RuntimeError(
                f"stage {self.name!r} failed with exit code {return_code}; "
                f"see {self.log_path}"
            )
        print(f"[{self.name}] completed")

    def terminate(self) -> None:
        if self.process.poll() is not None:
            if not self._stream.closed:
                self._stream.close()
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        self.process.wait()
        if not self._stream.closed:
            self._stream.close()


MappingLike = dict[str, str]


def _run_stage(
    name: str,
    command: Sequence[str],
    *,
    environment: MappingLike,
    log_path: Path,
) -> None:
    stage = RunningStage(name, command, environment=environment, log_path=log_path)
    try:
        stage.wait()
    except BaseException:
        stage.terminate()
        raise


def _training_result(stage_root: Path) -> tuple[Path, Path]:
    results = sorted(stage_root.glob("*/result.json"))
    if len(results) != 1:
        raise RuntimeError(
            f"expected exactly one training result below {stage_root}, got {results}"
        )
    payload = json.loads(results[0].read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError(f"training did not complete successfully: {results[0]}")
    checkpoint = Path(str(payload.get("best_model_path", "")))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"best checkpoint is missing: {checkpoint}")
    config = results[0].parent / "resolved-config.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"resolved training config is missing: {config}")
    return checkpoint.resolve(), config.resolve()


def _checked_path(path: Path | None, name: str) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def _source_storm_counts(
    paired_root: Path, storm_ids: Sequence[str]
) -> dict[str, dict[str, int]]:
    requested = set(storm_ids)
    counts = {
        storm_id: {split: 0 for split in ("train", "val", "test")}
        for storm_id in storm_ids
    }
    for split in ("train", "val", "test"):
        manifest = paired_root / split / "manifest.csv"
        with manifest.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                storm_id = str(row.get("storm_id", "")).strip().upper()
                if storm_id in requested:
                    counts[storm_id][split] += 1
    return counts


def _protected_storm_audit(
    paired_root: Path,
    ibtracs_file: Path,
    storm_ids: Sequence[str],
    *,
    era5: str,
) -> dict[str, Any]:
    """Reject protected training storms and report their effective coverage."""
    from geo2wf.config import compose_config, instantiate_datamodule

    source_counts = _source_storm_counts(paired_root, storm_ids)
    source_training = [
        storm_id for storm_id in storm_ids if source_counts[storm_id]["train"]
    ]
    if source_training:
        raise ValueError(
            "protected storms occur in the source training split: " f"{source_training}"
        )

    experiment = (
        "intensity_comparison_unet"
        if era5 == "with"
        else "intensity_comparison_unet_no_era5"
    )
    config = compose_config(
        [
            f"experiment={experiment}",
            f"data.root={paired_root}",
            f"data.stats_file={paired_root / 'stats.json'}",
            f"data.ibtracs_file={ibtracs_file}",
            "data.num_workers=0",
        ]
    )
    datamodule = instantiate_datamodule(config)
    effective_counts = {
        storm_id: {split: 0 for split in ("train", "val", "test")}
        for storm_id in storm_ids
    }
    for split in ("train", "val", "test"):
        dataset = datamodule._make_dataset(split, augment=False)
        values = dataset.samples["storm_id"].astype(str).str.upper().value_counts()
        for storm_id in storm_ids:
            effective_counts[storm_id][split] = int(values.get(storm_id, 0))

    effective_training = [
        storm_id for storm_id in storm_ids if effective_counts[storm_id]["train"]
    ]
    if effective_training:
        raise ValueError(
            "protected storms occur in the effective training cohort: "
            f"{effective_training}"
        )

    storms = {}
    for storm_id in storm_ids:
        held_out_splits = [
            split for split in ("val", "test") if effective_counts[storm_id][split]
        ]
        storms[storm_id] = {
            "source_counts": source_counts[storm_id],
            "effective_counts": effective_counts[storm_id],
            "effective_held_out_splits": held_out_splits,
            "included_in_evaluation": bool(held_out_splits),
        }
        source_text = (
            ", ".join(
                f"{split}={count}"
                for split, count in source_counts[storm_id].items()
                if count
            )
            or "absent"
        )
        effective_text = (
            ", ".join(
                f"{split}={count}"
                for split, count in effective_counts[storm_id].items()
                if count
            )
            or "absent"
        )
        print(
            f"[protected storm] {storm_id}: source {source_text}; "
            f"effective {effective_text}"
        )
    return {
        "era5": era5,
        "protected_storms": list(storm_ids),
        "storms": storms,
    }


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    paired_root = args.paired_root.expanduser().resolve()
    ibtracs_file = args.ibtracs_file.expanduser().resolve()
    stats_file = paired_root / "stats.json"
    for path in (paired_root, ibtracs_file, stats_file):
        if not path.exists():
            raise FileNotFoundError(path)
    storm_audit = _protected_storm_audit(
        paired_root,
        ibtracs_file,
        args.protected_storms,
        era5=args.era5,
    )

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (
            ROOT
            / "logs"
            / "intensity-comparisons"
            / f"{_utc_timestamp()}-{args.era5}-era5-{args.intensity_target_source}"
        ).resolve()
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"workflow output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    joint_checkpoint = _checked_path(args.joint_checkpoint, "joint checkpoint")
    unet_checkpoint = _checked_path(args.unet_checkpoint, "U-Net checkpoint")
    unet_config = _checked_path(args.unet_config, "U-Net config")
    correction_checkpoint = _checked_path(
        args.correction_checkpoint, "correction checkpoint"
    )

    environment = os.environ.copy()
    environment.pop("GEO2WF_RUN_DIR", None)
    if args.disable_wandb:
        environment["WANDB_DISABLED"] = "true"
    else:
        environment.pop("WANDB_DISABLED", None)
    common = [
        f"seed={args.seed}",
        f"data.root={paired_root}",
        f"data.stats_file={stats_file}",
        f"data.ibtracs_file={ibtracs_file}",
        f"data.intensity_target_source={args.intensity_target_source}",
        "data.require_sar_valid_center=true",
        f"data.batch_size={args.image_batch_size}",
        f"data.num_workers={args.num_workers}",
    ]
    if args.wandb_group:
        common.append(f"logging.wandb.group={args.wandb_group}")
    smoke = (
        [
            "trainer.max_epochs=1",
            "trainer.limit_train_batches=1",
            "trainer.limit_val_batches=1",
        ]
        if args.smoke_test
        else []
    )
    timestamp = _utc_timestamp()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "split": args.split,
        "era5": args.era5,
        "intensity_target_source": args.intensity_target_source,
        "seed": args.seed,
        "gpus": {"joint": args.joint_gpu, "pipeline": args.pipeline_gpu},
        "paired_root": str(paired_root),
        "ibtracs_file": str(ibtracs_file),
        "storm_holdout_audit": storm_audit,
        "stages": {},
    }
    manifest_path = output_root / "workflow.json"
    _atomic_json(manifest, manifest_path)

    joint_stage: RunningStage | None = None
    try:
        joint_experiment = (
            "bottleneck_unet_mlp"
            if args.era5 == "with"
            else "bottleneck_unet_mlp_no_era5"
        )
        unet_experiment = (
            "intensity_comparison_unet"
            if args.era5 == "with"
            else "intensity_comparison_unet_no_era5"
        )
        if joint_checkpoint is None:
            overrides = [
                *common,
                "logging.wandb.name=intensity-comparison-"
                f"{args.era5}-era5-{args.intensity_target_source}-joint-{timestamp}",
                *smoke,
            ]
            if args.joint_epochs is not None and not args.smoke_test:
                overrides.append(f"trainer.max_epochs={args.joint_epochs}")
            command = _training_command(
                joint_experiment,
                output_root / "joint-runs",
                args.joint_gpu,
                overrides,
            )
            joint_environment = environment.copy()
            joint_stage = RunningStage(
                "joint training",
                command,
                environment=joint_environment,
                log_path=output_root / "joint-training.log",
            )
            manifest["stages"]["joint_training"] = {"command": command}
        else:
            manifest["stages"]["joint_training"] = {
                "reused_checkpoint": str(joint_checkpoint)
            }

        pipeline_environment = environment.copy()
        if unet_checkpoint is None:
            overrides = [
                *common,
                "logging.wandb.name=intensity-comparison-"
                f"{args.era5}-era5-matched-unet-{timestamp}",
                *smoke,
            ]
            if args.unet_epochs is not None and not args.smoke_test:
                overrides.append(f"trainer.max_epochs={args.unet_epochs}")
            command = _training_command(
                unet_experiment,
                output_root / "unet-runs",
                args.pipeline_gpu,
                overrides,
            )
            manifest["stages"]["unet_training"] = {"command": command}
            _run_stage(
                "U-Net training",
                command,
                environment=pipeline_environment,
                log_path=output_root / "unet-training.log",
            )
            unet_checkpoint, unet_config = _training_result(output_root / "unet-runs")
        else:
            assert unet_config is not None
            manifest["stages"]["unet_training"] = {
                "reused_checkpoint": str(unet_checkpoint),
                "resolved_config": str(unet_config),
            }

        assert unet_checkpoint is not None and unet_config is not None
        from geo2wf.config import load_config_file

        resolved_unet_config = load_config_file(unet_config)
        robust_peak_fraction = float(
            resolved_unet_config["data"]["sar_robust_peak_fraction"]
        )
        cache_root = output_root / "unet-intensity-cache"
        export_command = [
            sys.executable,
            str(ROOT / "scripts" / "export_joint_intensity_cache.py"),
            "--config",
            str(unet_config),
            "--checkpoint",
            str(unet_checkpoint),
            "--output-root",
            str(cache_root),
            "--intensity-target-source",
            args.intensity_target_source,
            "--batch-size",
            str(args.image_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            f"cuda:{args.pipeline_gpu}",
        ]
        manifest["stages"]["cache_export"] = {"command": export_command}
        _run_stage(
            "common-cohort cache export",
            export_command,
            environment=pipeline_environment,
            log_path=output_root / "cache-export.log",
        )

        if correction_checkpoint is None:
            overrides = [
                f"seed={args.seed}",
                f"data.root={cache_root}",
                f"data.batch_size={args.correction_batch_size}",
                f"data.num_workers={args.num_workers}",
                "model.anchor_statistic="
                + (
                    "max"
                    if args.intensity_target_source == "ibtracs"
                    else "robust_peak"
                ),
                f"model.robust_peak_fraction={robust_peak_fraction}",
                "logging.wandb.name=intensity-comparison-"
                f"{args.era5}-era5-{args.intensity_target_source}-correction-{timestamp}",
                *smoke,
            ]
            if args.wandb_group:
                overrides.append(f"logging.wandb.group={args.wandb_group}")
            if args.correction_epochs is not None and not args.smoke_test:
                overrides.append(f"trainer.max_epochs={args.correction_epochs}")
            command = _training_command(
                "unet_intensity_correction",
                output_root / "correction-runs",
                args.pipeline_gpu,
                overrides,
            )
            manifest["stages"]["correction_training"] = {"command": command}
            _run_stage(
                "correction training",
                command,
                environment=pipeline_environment,
                log_path=output_root / "correction-training.log",
            )
            correction_checkpoint, _ = _training_result(output_root / "correction-runs")
        else:
            manifest["stages"]["correction_training"] = {
                "reused_checkpoint": str(correction_checkpoint)
            }

        if joint_stage is not None:
            joint_stage.wait()
            joint_checkpoint, _ = _training_result(output_root / "joint-runs")
        assert joint_checkpoint is not None and correction_checkpoint is not None

        evaluation_output = output_root / f"{args.split}-comparison.json"
        evaluate_command = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_intensity_models.py"),
            "--data-config",
            str(unet_config),
            "--cache-root",
            str(cache_root),
            "--joint-checkpoint",
            str(joint_checkpoint),
            "--correction-checkpoint",
            str(correction_checkpoint),
            "--split",
            args.split,
            "--intensity-target-source",
            args.intensity_target_source,
            "--output",
            str(evaluation_output),
            "--batch-size",
            str(args.correction_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--bootstrap-repetitions",
            str(args.bootstrap_repetitions),
            "--bootstrap-seed",
            str(args.seed),
            "--device",
            f"cuda:{args.pipeline_gpu}",
        ]
        manifest["stages"]["evaluation"] = {"command": evaluate_command}
        _run_stage(
            "common-cohort evaluation",
            evaluate_command,
            environment=pipeline_environment,
            log_path=output_root / "evaluation.log",
        )
        manifest.update(
            {
                "status": "completed",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "artifacts": {
                    "unet_checkpoint": str(unet_checkpoint),
                    "unet_config": str(unet_config),
                    "joint_checkpoint": str(joint_checkpoint),
                    "correction_checkpoint": str(correction_checkpoint),
                    "cache_root": str(cache_root),
                    "comparison_json": str(evaluation_output),
                    "comparison_csv": str(evaluation_output.with_suffix(".csv")),
                    "comparison_markdown": str(evaluation_output.with_suffix(".md")),
                },
            }
        )
        _atomic_json(manifest, manifest_path)
        print(f"Workflow completed. Table: {evaluation_output.with_suffix('.md')}")
        return manifest
    except BaseException as error:
        if joint_stage is not None:
            joint_stage.terminate()
        manifest.update(
            {
                "status": "failed",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _atomic_json(manifest, manifest_path)
        raise


def main() -> None:
    run_workflow(parse_args())


if __name__ == "__main__":
    main()
