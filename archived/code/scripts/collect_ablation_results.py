#!/usr/bin/env python3
"""Collect training, inference, and calibration ablations into flat CSV/JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

INTENSITY_BINS_MS = (0.0, 17.0, 25.0, 33.0, 43.0, 60.0, math.inf)
INFERENCE_PREDICTORS = {
    "output_msw_ms_member_mean": ("msw", "member_mean"),
    "output_msw_ms_member_median": ("msw", "member_median"),
    "output_msw_ms_medoid": ("msw", "medoid_member"),
    "output_msw_ms_mean_field": ("msw", "mean_field"),
    "output_msw_ms_median_field": ("msw", "median_field"),
    "output_robust_peak_ms_member_mean": ("robust_peak", "member_mean"),
    "output_robust_peak_ms_member_median": ("robust_peak", "member_median"),
    "output_robust_peak_ms_medoid": ("robust_peak", "medoid_member"),
    "output_robust_peak_ms_mean_field": ("robust_peak", "mean_field"),
    "output_robust_peak_ms_median_field": ("robust_peak", "median_field"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(prediction) & np.isfinite(target)
    prediction = prediction[finite]
    target = target[finite]
    if not len(target):
        return {
            "count": 0,
            "bias_ms": None,
            "mae_ms": None,
            "rmse_ms": None,
            "correlation": None,
        }
    error = prediction - target
    correlation = None
    if (
        len(target) > 1
        and float(np.std(prediction)) > 0.0
        and float(np.std(target)) > 0.0
    ):
        candidate = float(np.corrcoef(prediction, target)[0, 1])
        correlation = candidate if math.isfinite(candidate) else None
    return {
        "count": int(len(error)),
        "bias_ms": float(error.mean()),
        "mae_ms": float(np.abs(error).mean()),
        "rmse_ms": float(np.sqrt(np.square(error).mean())),
        "correlation": correlation,
    }


def _metric_rows(
    base: dict[str, Any],
    prediction: np.ndarray,
    target: np.ndarray,
) -> list[dict[str, Any]]:
    rows = [
        {
            **base,
            "metric_scope": "overall",
            **{
                f"metric.{key}": value
                for key, value in _error_metrics(prediction, target).items()
            },
        }
    ]
    for lower, upper in zip(INTENSITY_BINS_MS[:-1], INTENSITY_BINS_MS[1:]):
        selected = (target >= lower) & (target < upper)
        if not selected.any():
            continue
        rows.append(
            {
                **base,
                "metric_scope": "target_intensity_bin",
                "target_intensity_lower_ms": lower,
                "target_intensity_upper_ms": upper if math.isfinite(upper) else None,
                **{
                    f"metric.{key}": value
                    for key, value in _error_metrics(
                        prediction[selected], target[selected]
                    ).items()
                },
            }
        )
    return rows


def collect_training(suite_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for manifest_path in sorted(suite_dir.glob("runs/*/run-manifest.json")):
        run_dir = manifest_path.parent
        manifest = _read_json(manifest_path)
        result_path = run_dir / "result.json"
        result = _read_json(result_path) if result_path.exists() else {}
        config_path = run_dir / "resolved-config.yaml"
        config = (
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        ablation = config.get("ablation", {})
        row = {
            "artifact_type": "training",
            "experiment": run_dir.name,
            "ablation_id": ablation.get("id"),
            "ablation_family": ablation.get("family"),
            "ablation_reference": ablation.get("reference"),
            "ablation_design": ablation.get("design"),
            "ablation_changes": json.dumps(ablation.get("changes", [])),
            "run_dir": str(run_dir.resolve()),
            "status": result.get("status", manifest.get("status", "unknown")),
            "best_model_path": result.get("best_model_path"),
            "best_model_score": result.get("best_model_score"),
            "epoch": result.get("epoch"),
            "global_step": result.get("global_step"),
            "model_type": config.get("model", {}).get("type"),
            "include_test_in_train": config.get("data", {}).get(
                "include_test_in_train"
            ),
            "metric_scope": "latest_validation",
        }
        row.update(
            {f"metric.{key}": value for key, value in result.get("metrics", {}).items()}
        )
        rows.append(row)
    return rows


def collect_inference(suite_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(suite_dir.rglob("run-metadata.json")):
        metadata = _read_json(metadata_path)
        if "ensemble_kind" not in metadata:
            continue
        output_root = metadata_path.parent
        base = {
            "artifact_type": "inference",
            "experiment": str(output_root.relative_to(suite_dir)),
            "run_dir": str(output_root.resolve()),
            "status": metadata.get("status", "unknown"),
            "guidance_scale": metadata.get("guidance_scale"),
            "ensemble_size": metadata.get("ensemble_size"),
            "ensemble_kind": metadata.get("ensemble_kind"),
            "summary_aggregation": metadata.get("aggregation", {}).get(
                "legacy_output_columns"
            ),
            "checkpoint_sha256": metadata.get("checkpoint", {}).get("sha256"),
            "baseline_checkpoint_sha256": (
                metadata.get("deterministic_baseline_checkpoint") or {}
            ).get("sha256"),
            "include_test_in_train": metadata.get("include_test_in_train"),
        }
        summary_paths = sorted(output_root.glob("*/inference-summary.csv"))
        if not summary_paths:
            rows.append({**base, "metric_scope": "run_metadata"})
            continue
        table = pd.concat(
            [pd.read_csv(path) for path in summary_paths], ignore_index=True
        )
        if "ibtracs_msw_ms" not in table:
            rows.append({**base, "metric_scope": "run_metadata"})
            continue
        target = table["ibtracs_msw_ms"].to_numpy(dtype=float)
        for column, (metric_family, aggregation) in INFERENCE_PREDICTORS.items():
            if column not in table:
                continue
            prediction = table[column].to_numpy(dtype=float)
            rows.extend(
                _metric_rows(
                    {
                        **base,
                        "metric_family": metric_family,
                        "aggregation": aggregation,
                        "prediction_column": column,
                    },
                    prediction,
                    target,
                )
            )
        lower_column = "output_msw_ms_member_p10"
        upper_column = "output_msw_ms_member_p90"
        if lower_column in table and upper_column in table:
            lower = table[lower_column].to_numpy(dtype=float)
            upper = table[upper_column].to_numpy(dtype=float)
            finite = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(target)
            rows.append(
                {
                    **base,
                    "metric_family": "msw_interval",
                    "aggregation": "member_p10_p90",
                    "prediction_column": f"{lower_column},{upper_column}",
                    "metric_scope": "overall",
                    "metric.count": int(finite.sum()),
                    "metric.coverage": (
                        float(
                            (
                                (target[finite] >= lower[finite])
                                & (target[finite] <= upper[finite])
                            ).mean()
                        )
                        if finite.any()
                        else None
                    ),
                    "metric.mean_width_ms": (
                        float((upper[finite] - lower[finite]).mean())
                        if finite.any()
                        else None
                    ),
                }
            )
    return rows


def _guidance_from_path(path: Path) -> float | None:
    for part in path.parts:
        if not part.startswith("guidance_"):
            continue
        label = part.removeprefix("guidance_").replace("m", "-").replace("p", ".")
        try:
            return float(label)
        except ValueError:
            return None
    return None


def collect_calibration(suite_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    protocol_labels = {
        "raw": "uncalibrated",
        "calibrated_in_sample": "in_sample_optimistic_do_not_rank",
        "calibrated_leave_one_storm_out": "leave_one_storm_out_diagnostic",
    }
    for path in sorted(suite_dir.rglob("calibration-evaluation.json")):
        payload = _read_json(path)
        calibration_path = path.parent / "calibration.json"
        calibration = _read_json(calibration_path) if calibration_path.exists() else {}
        base = {
            "artifact_type": "calibration",
            "experiment": str(path.parent.relative_to(suite_dir)),
            "run_dir": str(path.parent.resolve()),
            "status": "completed",
            "guidance_scale": _guidance_from_path(path),
            "calibration_method": calibration.get("model", {}).get("method"),
            "prediction_column": calibration.get("prediction_column"),
            "target_column": calibration.get("target_column"),
            "fit_count": calibration.get("fit_count"),
            "calibration_storm_count": len(calibration.get("storms", [])),
        }
        for variant, protocol in protocol_labels.items():
            metrics = payload.get(variant)
            if not metrics:
                continue
            overall = {
                **base,
                "calibration_variant": variant,
                "evaluation_protocol": protocol,
                "metric_scope": "overall",
                **{
                    f"metric.{key}": value
                    for key, value in metrics.items()
                    if key != "by_target_intensity"
                },
            }
            rows.append(overall)
            for intensity in metrics.get("by_target_intensity", []):
                rows.append(
                    {
                        **base,
                        "calibration_variant": variant,
                        "evaluation_protocol": protocol,
                        "metric_scope": "target_intensity_bin",
                        "target_intensity_lower_ms": intensity.get("lower_ms"),
                        "target_intensity_upper_ms": intensity.get("upper_ms"),
                        **{
                            f"metric.{key}": value
                            for key, value in intensity.items()
                            if key not in {"lower_ms", "upper_ms"}
                        },
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)
    rows = (
        collect_training(suite_dir)
        + collect_inference(suite_dir)
        + collect_calibration(suite_dir)
    )
    payload = {
        "schema_version": 2,
        "suite_dir": str(suite_dir),
        "artifact_count": len(rows),
        "artifacts": rows,
    }
    (suite_dir / "ablation-results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(suite_dir / "ablation-results.csv", index=False)
    print(f"Collected {len(rows)} artifacts in {suite_dir}")


if __name__ == "__main__":
    main()
