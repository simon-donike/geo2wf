#!/usr/bin/env python3
"""Evaluate completed current experiments with one comparable metric schema."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import pytorch_lightning as pl
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo2wf.config import (  # noqa: E402
    instantiate_datamodule,
    instantiate_model,
    load_config_file,
)
from geo2wf.metrics.image_quality import (  # noqa: E402
    masked_ssim_sum_count,
    psnr_db_from_mse,
)


CURRENT_EXPERIMENT_ROOTS = {
    "correction_image_radii": "logs/intensity-structure/unet-image-radii",
    "correction_mlp_radii": "logs/intensity-structure/mlp-radii",
    "latent_sar_era5_max_wind": "logs/latent-matrix/sar/era5/max-wind",
    "latent_sar_era5_max_wind_radii": ("logs/latent-matrix/sar/era5/max-wind-radii"),
    "latent_sar_no_era5_max_wind": "logs/latent-matrix/sar/no-era5/max-wind",
    "latent_sar_no_era5_max_wind_radii": (
        "logs/latent-matrix/sar/no-era5/max-wind-radii"
    ),
    "latent_no_sar_era5_max_wind": "logs/latent-matrix/no-sar/era5/max-wind",
    "latent_no_sar_era5_max_wind_radii": (
        "logs/latent-matrix/no-sar/era5/max-wind-radii"
    ),
    "latent_no_sar_no_era5_max_wind": ("logs/latent-matrix/no-sar/no-era5/max-wind"),
    "latent_no_sar_no_era5_max_wind_radii": (
        "logs/latent-matrix/no-sar/no-era5/max-wind-radii"
    ),
}

RADIUS_NAMES = ("rmw", "r34", "r50", "r64")
STRUCTURE_NAMES = (
    "eye_size",
    "rmw",
    "r34_equivalent",
    "r50_equivalent",
    "r64_equivalent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=RUN_DIR",
        help="Explicit completed run; repeat for multiple models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/current-experiment-evaluation"),
    )
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "gpu"),
        default="gpu" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Skip current experiments that have not completed yet.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parse_explicit_runs(values: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run must be NAME=RUN_DIR, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in runs:
            raise ValueError(f"invalid or duplicate experiment name: {name!r}")
        runs[name] = Path(raw_path).expanduser().resolve()
    return runs


def _latest_completed_run(root: Path) -> Path | None:
    completed: list[Path] = []
    if not root.is_dir():
        return None
    for result_path in root.glob("*/result.json"):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            completed.append(result_path.parent)
    return max(completed, key=lambda path: path.name) if completed else None


def _completed_run(run_dir: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    result_path = run_dir / "result.json"
    config_path = run_dir / "resolved-config.yaml"
    if not result_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"missing result/config in {run_dir}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError(f"run is not completed: {run_dir}")
    checkpoint = Path(result.get("best_model_path", "")).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing selected checkpoint: {checkpoint}")
    return load_config_file(config_path), checkpoint, result


def _float_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    result = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        number = float(value)
        if math.isfinite(number):
            result[str(key)] = number
    return result


def _validate_run(
    config: dict[str, Any],
    checkpoint: Path,
    *,
    accelerator: str,
    devices: int,
) -> dict[str, float]:
    datamodule = instantiate_datamodule(config)
    datamodule.setup("validate")
    model = instantiate_model(config)
    model.validate_data_spec(datamodule.data_spec)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
    )
    results = trainer.validate(model=model, datamodule=datamodule, verbose=False)
    if not results:
        raise RuntimeError("validation produced no metric result")
    return _float_metrics(results[0])


def _correction_image_metrics(config: dict[str, Any]) -> dict[str, float]:
    """Evaluate the frozen U-Net field carried by a correction experiment."""

    cache_root = Path(config["data"]["root"]).expanduser().resolve()
    metadata = json.loads(
        (cache_root / "cache-metadata.json").read_text(encoding="utf-8")
    )
    source_config_path = Path(metadata["unet_config"]["path"]).expanduser().resolve()
    source_config = load_config_file(source_config_path)
    paired_module = instantiate_datamodule(source_config)
    paired_module.setup("fit")
    cache_module = instantiate_datamodule(config)
    cache_module.setup("fit")
    paired_dataset = paired_module.val_dataset
    cache_dataset = cache_module.val_dataset
    paired_index = {
        str(sample_id): index
        for index, sample_id in enumerate(paired_dataset.samples["sample_id"])
    }
    if set(paired_index) != set(cache_dataset.samples["sample_id"].astype(str)):
        raise ValueError("correction cache and paired validation cohorts differ")

    # valid pixels, abs error, squared error, signed error, SSIM sum, SSIM scenes
    statistics = {
        "val": [0.0] * 6,
        "val_ri": [0.0] * 6,
    }
    for cache_index, sample_id in enumerate(
        cache_dataset.samples["sample_id"].astype(str)
    ):
        cached = cache_dataset[cache_index]
        paired = paired_dataset[paired_index[sample_id]]
        prediction = cached["wind_field"].reshape(1, 1, *cached["wind_field"].shape)
        target = paired["target_physical"].reshape(
            1, 1, *paired["target_physical"].shape[-2:]
        )
        mask = (
            cached["valid_mask"].reshape(1, 1, *cached["valid_mask"].shape)
            & paired["target_mask"].reshape(1, 1, *paired["target_mask"].shape[-2:])
            & paired["condition_mask"].reshape(
                1, 1, *paired["condition_mask"].shape[-2:]
            )
            & torch.isfinite(prediction)
            & torch.isfinite(target)
        )
        error = (prediction - target)[mask].to(torch.float64)
        if not error.numel():
            continue
        ssim_sum, ssim_scenes = masked_ssim_sum_count(prediction, target, mask)
        additions = (
            float(error.numel()),
            float(error.abs().sum()),
            float(error.square().sum()),
            float(error.sum()),
            float(ssim_sum),
            float(ssim_scenes),
        )
        prefixes = ["val"]
        if bool(cached.get("is_rapid_intensification", False)):
            prefixes.append("val_ri")
        for prefix in prefixes:
            statistics[prefix] = [
                current + addition
                for current, addition in zip(statistics[prefix], additions)
            ]

    metrics: dict[str, float] = {}
    for prefix, values in statistics.items():
        count, absolute, squared, signed, ssim_sum, ssim_scenes = values
        if count <= 0:
            continue
        mse = squared / count
        field_stem = "image" if prefix == "val" else "field"
        metrics[f"{prefix}/{field_stem}_mae_ms"] = absolute / count
        metrics[f"{prefix}/{field_stem}_rmse_ms"] = math.sqrt(mse)
        metrics[f"{prefix}/{field_stem}_bias_ms"] = signed / count
        metrics[f"{prefix}/{field_stem}_psnr_db"] = psnr_db_from_mse(mse)
        if ssim_scenes:
            metrics[f"{prefix}/{field_stem}_ssim"] = ssim_sum / ssim_scenes
    return metrics


def _first(metrics: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if value is not None and math.isfinite(value):
            return value
    return None


def _row(
    experiment: str,
    subset: str,
    output: str,
    target: str,
    metric: str,
    units: str,
    value: float | None,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "subset": subset,
        "output": output,
        "target": target,
        "metric": metric,
        "units": units,
        "value": value,
        "available": value is not None,
        "availability_reason": None if value is not None else reason,
    }


def _standard_rows(
    experiment: str, config: dict[str, Any], metrics: dict[str, float]
) -> list[dict[str, Any]]:
    target = str(config["model"]["_target_"])
    has_decoder = target.endswith("BottleneckUNetMLPRegressor")
    is_correction = target.endswith("UNetIntensityCorrection")
    has_image_output = has_decoder or is_correction
    no_image_reason = (
        None
        if has_image_output
        else "not_applicable: model has no jointly evaluated image decoder"
    )
    rows: list[dict[str, Any]] = []
    for subset, prefix in (("all_validation", "val"), ("ri_validation", "val_ri")):
        image_keys = {
            "l1": (
                f"{prefix}/image_mae_ms",
                f"{prefix}/field_mae_ms",
            ),
            "psnr": (
                f"{prefix}/image_psnr_db",
                f"{prefix}/field_psnr_db",
            ),
            "ssim": (f"{prefix}/image_ssim", f"{prefix}/field_ssim"),
        }
        for metric_name, keys in image_keys.items():
            units = (
                "m s-1"
                if metric_name == "l1"
                else ("dB" if metric_name == "psnr" else "1")
            )
            rows.append(
                _row(
                    experiment,
                    subset,
                    "image_reconstruction",
                    "wind_field",
                    metric_name,
                    units,
                    _first(metrics, *keys) if has_image_output else None,
                    reason=no_image_reason,
                )
            )

        intensity_stem = "intensity" if not is_correction else ""
        for metric_name in ("mae", "rmse"):
            candidates = [
                f"{prefix}/{intensity_stem + '_' if intensity_stem else ''}{metric_name}_ms"
            ]
            if subset == "ri_validation" and is_correction:
                candidates.insert(0, f"{prefix}/ibtracs_{metric_name}_ms")
            if subset == "ri_validation" and has_decoder:
                candidates.insert(0, f"{prefix}/ibtracs_{metric_name}_ms")
            rows.append(
                _row(
                    experiment,
                    subset,
                    "scalar_head",
                    "maximum_wind",
                    metric_name,
                    "m s-1",
                    _first(metrics, *candidates),
                    reason="metric_not_emitted",
                )
            )

        for structure_name in STRUCTURE_NAMES:
            for metric_name in ("mae", "rmse", "bias"):
                rows.append(
                    _row(
                        experiment,
                        subset,
                        "scalar_radius_head",
                        structure_name,
                        metric_name,
                        "km",
                        _first(
                            metrics,
                            f"{prefix}/structure_{structure_name}_{metric_name}_km",
                        ),
                        reason="not_applicable: radius head disabled or target unavailable",
                    )
                )

        for radius in RADIUS_NAMES:
            image_name = radius if radius == "rmw" else f"{radius}_equivalent"
            for metric_name in ("mae", "rmse", "bias"):
                rows.append(
                    _row(
                        experiment,
                        subset,
                        "image_derived_radius",
                        radius,
                        metric_name,
                        "km",
                        _first(
                            metrics,
                            f"{prefix}/ibtracs_{radius}_{metric_name}_km",
                            f"{prefix}/unet_image_{image_name}_{metric_name}_km",
                        ),
                        reason=(
                            "not_applicable: no image output or image-radius metric unavailable"
                        ),
                    )
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    explicit = _parse_explicit_runs(args.run)
    runs = explicit
    missing = []
    if not runs:
        runs = {}
        for name, relative_root in CURRENT_EXPERIMENT_ROOTS.items():
            run = _latest_completed_run(ROOT / relative_root)
            if run is None:
                missing.append(name)
            else:
                runs[name] = run
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            "current experiments are not all completed: " + ", ".join(missing)
        )
    if not runs:
        raise RuntimeError("no completed runs selected")

    rows: list[dict[str, Any]] = []
    run_payloads = {}
    for name, run_dir in runs.items():
        config, checkpoint, fit_result = _completed_run(run_dir)
        metrics = _validate_run(
            config,
            checkpoint,
            accelerator=args.accelerator,
            devices=args.devices,
        )
        if str(config["model"]["_target_"]).endswith("UNetIntensityCorrection"):
            metrics.update(_correction_image_metrics(config))
        rows.extend(_standard_rows(name, config, metrics))
        run_payloads[name] = {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "model_target": config["model"]["_target_"],
            "data_target": config["data"]["_target_"],
            "fit_result": fit_result,
            "validation_metrics": metrics,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "validation-metrics.csv"
    json_path = args.output_dir / "validation-results.json"
    _write_csv(csv_path, rows)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subset_definitions": {
            "all_validation": "configured validation split",
            "ri_validation": "validation samples with IBTrACS 24-hour wind increase >= 30 kt",
        },
        "image_metrics": {
            "l1": "pooled valid-pixel MAE in physical wind speed",
            "psnr": "physical-space PSNR with fixed 79.8 m/s data range",
            "ssim": "mean scene SSIM over complete valid 7x7 windows",
        },
        "missing_experiments": missing,
        "runs": run_payloads,
        "metrics": rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
