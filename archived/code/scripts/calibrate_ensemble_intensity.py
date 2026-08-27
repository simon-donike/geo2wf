#!/usr/bin/env python3
"""Fit and evaluate monotone ensemble-intensity calibration from summary CSVs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-column", default="output_msw_ms_member_median")
    parser.add_argument("--target-column", default="ibtracs_msw_ms")
    parser.add_argument("--method", choices=("isotonic", "affine"), default="isotonic")
    parser.add_argument(
        "--output-min-ms",
        type=float,
        default=0.0,
        help="Lower physical bound applied to every calibrated intensity.",
    )
    parser.add_argument(
        "--output-max-ms",
        type=float,
        default=80.0,
        help="Upper physical bound applied to every calibrated intensity.",
    )
    parser.add_argument(
        "--intensity-bins-ms",
        nargs="+",
        type=float,
        default=[0.0, 17.0, 25.0, 33.0, 43.0, 60.0, math.inf],
    )
    return parser.parse_args()


def _pava(y: np.ndarray, weight: np.ndarray) -> np.ndarray:
    values: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, (value, item_weight) in enumerate(zip(y, weight)):
        values.append(float(value))
        weights.append(float(item_weight))
        starts.append(index)
        ends.append(index + 1)
        while len(values) >= 2 and values[-2] > values[-1]:
            total = weights[-2] + weights[-1]
            merged = (values[-2] * weights[-2] + values[-1] * weights[-1]) / total
            values[-2:] = [merged]
            weights[-2:] = [total]
            ends[-2:] = [ends[-1]]
            starts.pop()
    fitted = np.empty_like(y, dtype=float)
    for value, start, end in zip(values, starts, ends):
        fitted[start:end] = value
    return fitted


def fit_isotonic(x: np.ndarray, y: np.ndarray) -> dict:
    order = np.argsort(x, kind="mergesort")
    x_sorted, y_sorted = x[order], y[order]
    unique_x, inverse, counts = np.unique(
        x_sorted, return_inverse=True, return_counts=True
    )
    summed = np.bincount(inverse, weights=y_sorted)
    means = summed / counts
    fitted = _pava(means, counts.astype(float))
    # Adjacent equal fitted values are harmless; retaining every unique x makes
    # the serialized interpolation explicit and reproducible.
    return {
        "method": "isotonic",
        "x_thresholds": unique_x.tolist(),
        "y_thresholds": fitted.tolist(),
        "out_of_bounds": "clip",
    }


def _has_meaningful_spread(values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or not np.isfinite(values).all():
        return False
    scale = max(1.0, float(np.max(np.abs(values))))
    return float(np.ptp(values)) > 1e-12 * scale


def fit_affine(x: np.ndarray, y: np.ndarray) -> dict:
    if not _has_meaningful_spread(x):
        return {
            "method": "affine",
            "slope": 0.0,
            "intercept": float(np.mean(y)),
        }
    slope, intercept = np.polyfit(x, y, 1)
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0:
        slope = 0.0
        intercept = float(np.mean(y))
    return {
        "method": "affine",
        "slope": float(slope),
        "intercept": float(intercept),
    }


def _validate_output_bounds(output_min_ms: float, output_max_ms: float) -> None:
    if (
        not math.isfinite(output_min_ms)
        or not math.isfinite(output_max_ms)
        or output_max_ms <= output_min_ms
    ):
        raise ValueError(
            "output bounds must be finite with maximum greater than minimum"
        )


def fit_calibrator(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    output_min_ms: float = 0.0,
    output_max_ms: float = 80.0,
) -> dict:
    if len(x) < 2:
        raise ValueError("calibration requires at least two finite observations")
    _validate_output_bounds(output_min_ms, output_max_ms)
    model = fit_isotonic(x, y) if method == "isotonic" else fit_affine(x, y)
    model["output_min_ms"] = float(output_min_ms)
    model["output_max_ms"] = float(output_max_ms)
    return model


def apply_calibrator(model: dict, x: np.ndarray) -> np.ndarray:
    if model["method"] == "affine":
        calibrated = model["slope"] * x + model["intercept"]
    else:
        calibrated = np.interp(
            x,
            np.asarray(model["x_thresholds"], dtype=float),
            np.asarray(model["y_thresholds"], dtype=float),
        )
    return np.clip(
        calibrated,
        float(model.get("output_min_ms", -math.inf)),
        float(model.get("output_max_ms", math.inf)),
    )


def error_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    bins: list[float],
) -> dict:
    error = prediction - target
    correlation = None
    if (
        len(error) > 1
        and _has_meaningful_spread(prediction)
        and _has_meaningful_spread(target)
    ):
        candidate = float(np.corrcoef(prediction, target)[0, 1])
        correlation = candidate if math.isfinite(candidate) else None
    result = {
        "count": int(len(error)),
        "bias_ms": float(error.mean()),
        "mae_ms": float(np.abs(error).mean()),
        "rmse_ms": float(np.sqrt(np.square(error).mean())),
        "correlation": correlation,
        "by_target_intensity": [],
    }
    for lower, upper in zip(bins[:-1], bins[1:]):
        selected = (target >= lower) & (target < upper)
        if not selected.any():
            continue
        selected_error = error[selected]
        result["by_target_intensity"].append(
            {
                "lower_ms": lower,
                "upper_ms": upper if math.isfinite(upper) else None,
                "count": int(selected.sum()),
                "bias_ms": float(selected_error.mean()),
                "mae_ms": float(np.abs(selected_error).mean()),
                "rmse_ms": float(np.sqrt(np.square(selected_error).mean())),
            }
        )
    return result


def interval_metrics(
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
) -> dict:
    valid = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(target)
    lower = lower[valid]
    upper = upper[valid]
    target = target[valid]
    if not len(target):
        return {
            "count": 0,
            "coverage": None,
            "mean_width_ms": None,
        }
    return {
        "count": int(len(target)),
        "coverage": float(np.mean((target >= lower) & (target <= upper))),
        "mean_width_ms": float(np.mean(upper - lower)),
    }


_ENSEMBLE_SUMMARY_SUFFIXES = (
    "member_p10",
    "member_p90",
    "member_mean",
    "member_median",
    "medoid",
    "mean_field",
    "median_field",
)


def _metric_prefix(prediction_column: str) -> str:
    for suffix in _ENSEMBLE_SUMMARY_SUFFIXES:
        token = "_" + suffix
        if prediction_column.endswith(token):
            return prediction_column[: -len(token)]
    return prediction_column


def _related_calibration_columns(
    columns: pd.Index,
    prediction_column: str,
) -> tuple[str, list[str]]:
    prefix = _metric_prefix(prediction_column)
    related = [
        f"{prefix}_{suffix}"
        for suffix in _ENSEMBLE_SUMMARY_SUFFIXES
        if f"{prefix}_{suffix}" in columns
    ]
    if prediction_column not in related:
        related.insert(0, prediction_column)
    return prefix, related


def _storm_label(path: Path, frame: pd.DataFrame) -> pd.Series:
    if "storm_id" in frame:
        return frame["storm_id"].astype(str)
    # Standard contract is <root>/<storm>/inference-summary.csv.
    return pd.Series([path.parent.name] * len(frame), index=frame.index)


def main() -> None:
    args = parse_args()
    _validate_output_bounds(args.output_min_ms, args.output_max_ms)
    if any(
        right <= left
        for left, right in zip(args.intensity_bins_ms[:-1], args.intensity_bins_ms[1:])
    ):
        raise ValueError("--intensity-bins-ms must be strictly increasing")
    frames = []
    for path in args.inputs:
        frame = pd.read_csv(path)
        frame["_source_path"] = str(path.resolve())
        frame["_storm_id"] = _storm_label(path, frame)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    required = {args.prediction_column, args.target_column}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError("missing calibration columns: " + ", ".join(missing))
    valid = np.isfinite(table[args.prediction_column]) & np.isfinite(
        table[args.target_column]
    )
    table = table.loc[valid].reset_index(drop=True)
    prediction = table[args.prediction_column].to_numpy(float)
    target = table[args.target_column].to_numpy(float)
    storms = table["_storm_id"].to_numpy(str)

    model = fit_calibrator(
        prediction,
        target,
        args.method,
        output_min_ms=args.output_min_ms,
        output_max_ms=args.output_max_ms,
    )
    metric_prefix, related_columns = _related_calibration_columns(
        table.columns, args.prediction_column
    )
    calibrated_by_column = {
        column: apply_calibrator(model, table[column].to_numpy(float))
        for column in related_columns
    }
    calibrated = calibrated_by_column[args.prediction_column]
    cross_validated_by_column = {
        column: np.full_like(prediction, np.nan) for column in related_columns
    }
    unique_storms = np.unique(storms)
    if len(unique_storms) >= 2:
        for storm in unique_storms:
            held_out = storms == storm
            train = ~held_out
            fold_model = fit_calibrator(
                prediction[train],
                target[train],
                args.method,
                output_min_ms=args.output_min_ms,
                output_max_ms=args.output_max_ms,
            )
            for column in related_columns:
                values = table[column].to_numpy(float)
                cross_validated_by_column[column][held_out] = apply_calibrator(
                    fold_model, values[held_out]
                )
    cross_validated = cross_validated_by_column[args.prediction_column]

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for column, values in calibrated_by_column.items():
        table[column + "_calibrated"] = values
    if np.isfinite(cross_validated).all():
        for column, values in cross_validated_by_column.items():
            table[column + "_calibrated_loso"] = values
    table.to_csv(output / "calibrated-predictions.csv", index=False)
    calibration_payload = {
        "schema_version": 1,
        "prediction_column": args.prediction_column,
        "target_column": args.target_column,
        "output_bounds_ms": {
            "minimum": float(args.output_min_ms),
            "maximum": float(args.output_max_ms),
        },
        "fit_count": int(len(table)),
        "storms": unique_storms.tolist(),
        "related_calibrated_columns": related_columns,
        "model": model,
    }
    (output / "calibration.json").write_text(
        json.dumps(calibration_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluation = {
        "schema_version": 1,
        "raw": error_metrics(prediction, target, args.intensity_bins_ms),
        "calibrated_in_sample": error_metrics(
            calibrated, target, args.intensity_bins_ms
        ),
        "calibrated_leave_one_storm_out": (
            error_metrics(cross_validated, target, args.intensity_bins_ms)
            if np.isfinite(cross_validated).all()
            else None
        ),
    }
    interval_lower = f"{metric_prefix}_member_p10"
    interval_upper = f"{metric_prefix}_member_p90"
    if interval_lower in table and interval_upper in table:
        interval_by_variant = {
            "raw": interval_metrics(
                table[interval_lower].to_numpy(float),
                table[interval_upper].to_numpy(float),
                target,
            ),
            "calibrated_in_sample": interval_metrics(
                calibrated_by_column[interval_lower],
                calibrated_by_column[interval_upper],
                target,
            ),
            "calibrated_leave_one_storm_out": (
                interval_metrics(
                    cross_validated_by_column[interval_lower],
                    cross_validated_by_column[interval_upper],
                    target,
                )
                if np.isfinite(cross_validated).all()
                else None
            ),
        }
        for variant, interval in interval_by_variant.items():
            if evaluation[variant] is None or interval is None:
                continue
            evaluation[variant].update(
                {
                    "p10_p90_interval_count": interval["count"],
                    "p10_p90_coverage": interval["coverage"],
                    "p10_p90_mean_width_ms": interval["mean_width_ms"],
                }
            )
    (output / "calibration-evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = []
    for variant, metrics in evaluation.items():
        if variant == "schema_version" or metrics is None:
            continue
        rows.append(
            {
                "variant": variant,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "by_target_intensity"
                },
            }
        )
    pd.DataFrame(rows).to_csv(output / "calibration-summary.csv", index=False)
    print(f"Wrote calibration artifacts to {output}")


if __name__ == "__main__":
    main()
