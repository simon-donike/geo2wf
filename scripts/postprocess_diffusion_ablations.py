#!/usr/bin/env python3
"""Evaluate residual gain, amplitude-cap, and smoothing post-processing.

This sweep never changes the diffusion checkpoint.  For each saved ensemble
member it reconstructs ``baseline + transform(member - baseline)`` and writes
the same member-first metrics as storm inference.  The baseline field is a
separate artifact produced by ``save_deterministic_baseline_fields.py``.
"""

from __future__ import annotations

import argparse
import sys
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import median_filter

from scripts.run_storm_diffusion_inference import _summarize_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="One guidance directory containing <storm>/inference-summary.csv and fields.",
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--storms", nargs="+", default=None)
    parser.add_argument(
        "--gains", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument(
        "--residual-caps-ms",
        type=float,
        nargs="+",
        default=[8.0, 16.0, 24.0],
        help="Symmetric physical residual caps. An uncapped control is always included.",
    )
    parser.add_argument(
        "--median-kernels", type=int, nargs="+", default=[0, 3]
    )
    parser.add_argument("--output-min-ms", type=float, default=0.0)
    parser.add_argument("--output-max-ms", type=float, default=80.0)
    parser.add_argument("--summary-aggregation", choices=("median", "mean", "medoid"), default="median")
    parser.add_argument("--member-quantiles", type=float, nargs="+", default=[0.1, 0.9])
    return parser.parse_args()


def _safe_component(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
    return safe[:180] or "variant"


def _number_label(value: float) -> str:
    if math.isinf(value):
        return "none"
    return format(value, ".8g").replace("-", "m").replace(".", "p")


def _variant_name(gain: float, cap: float, kernel: int) -> str:
    smoothing = "median" if kernel else "raw"
    return f"gain_{_number_label(gain)}_cap_{_number_label(cap)}_{smoothing}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _finite_error(prediction: pd.Series, target: pd.Series) -> dict[str, Any]:
    x = prediction.to_numpy(float)
    y = target.to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return {"count": 0, "bias_ms": None, "mae_ms": None, "rmse_ms": None}
    error = x[valid] - y[valid]
    return {
        "count": int(valid.sum()),
        "bias_ms": float(error.mean()),
        "mae_ms": float(np.abs(error).mean()),
        "rmse_ms": float(np.sqrt(np.square(error).mean())),
    }


def _metrics_for_table(table: pd.DataFrame) -> dict[str, Any]:
    target = table["ibtracs_msw_ms"]
    metrics: dict[str, Any] = {}
    baseline_msw = _finite_error(table["baseline_msw_ms"], target)
    baseline_robust = _finite_error(table["baseline_robust_peak_ms"], target)
    metrics["baseline_msw_vs_ibtracs_msw"] = baseline_msw
    metrics["baseline_robust_peak_vs_ibtracs_msw"] = baseline_robust
    for name, column in (
        ("msw", "output_msw_ms"),
        ("robust_peak", "output_robust_peak_ms"),
    ):
        result = _finite_error(table[column], target)
        metrics[f"{name}_vs_ibtracs_msw"] = result
        baseline = baseline_msw if name == "msw" else baseline_robust
        if result["mae_ms"] is not None and baseline["mae_ms"] not in (None, 0):
            result["mae_skill_vs_baseline"] = float(
                1.0 - result["mae_ms"] / baseline["mae_ms"]
            )
        high = table["ibtracs_msw_ms"] >= 33.0
        high_table = table.loc[high]
        metrics[f"{name}_high_wind_ge_33_ms"] = _finite_error(
            high_table[column], high_table["ibtracs_msw_ms"]
        )
    return metrics


def _load_baseline_index(root: Path) -> dict[tuple[str, str], Path]:
    manifest = pd.read_csv(root / "baseline-fields-manifest.csv")
    required = {"storm_id", "observation_id", "npz_path"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise KeyError("baseline manifest missing: " + ", ".join(missing))
    result = {}
    for row in manifest.itertuples(index=False):
        result[(str(row.storm_id), str(row.observation_id))] = root / str(row.npz_path)
    return result


def _load_field_index(root: Path, storms: list[str]) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for storm in storms:
        manifest_path = root / storm / "member-fields-manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = pd.read_csv(manifest_path)
        for row in manifest.itertuples(index=False):
            result[(storm, str(row.observation_id))] = root / str(row.npz_path)
    return result


def _smooth_residual(residual: np.ndarray, valid: np.ndarray, kernel: int) -> np.ndarray:
    if kernel == 0:
        return residual
    filled = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
    smoothed = median_filter(filled, size=(1, kernel, kernel), mode="nearest")
    return np.where(valid[None, ...], smoothed, np.nan)


def _validate_args(args: argparse.Namespace) -> tuple[list[float], list[float], list[int]]:
    if not args.gains or any(not math.isfinite(value) or value < 0 for value in args.gains):
        raise ValueError("--gains must contain finite non-negative values")
    caps = [float(value) for value in args.residual_caps_ms]
    if any(not math.isfinite(value) or value <= 0 for value in caps):
        raise ValueError("--residual-caps-ms must contain positive finite values")
    kernels = [int(value) for value in args.median_kernels]
    if not kernels or any(value not in (0, 3) for value in kernels):
        raise ValueError("--median-kernels currently supports only 0 and 3")
    if args.output_max_ms <= args.output_min_ms:
        raise ValueError("output bounds must be strictly increasing")
    if len(set(args.gains)) != len(args.gains) or len(set(caps)) != len(caps) or len(set(kernels)) != len(kernels):
        raise ValueError("sweep values must be unique")
    return list(args.gains), caps, kernels


def main() -> None:
    args = parse_args()
    gains, caps, kernels = _validate_args(args)
    summary_paths = sorted(args.input_root.glob("*/inference-summary.csv"))
    if args.storms is not None:
        wanted = set(args.storms)
        summary_paths = [path for path in summary_paths if path.parent.name in wanted]
    if not summary_paths:
        raise FileNotFoundError("no storm inference summaries below " + str(args.input_root))
    storms = [path.parent.name for path in summary_paths]
    baseline_index = _load_baseline_index(args.baseline_root)
    field_index = _load_field_index(args.input_root, storms)
    source_tables = {storm: pd.read_csv(path) for storm, path in ((p.parent.name, p) for p in summary_paths)}

    # Compute baseline diagnostics once. They are retained in every variant
    # table so skill scores are auditable without joining additional files.
    source_rows: list[tuple[str, pd.Series, np.ndarray, np.ndarray, np.ndarray]] = []
    baseline_rows: list[dict[str, float]] = []
    for storm, table in source_tables.items():
        for _, row in table.iterrows():
            observation_id = str(row["observation_id"])
            field_path = field_index[(storm, observation_id)]
            baseline_path = baseline_index[(storm, observation_id)]
            with np.load(field_path) as fields:
                members = np.asarray(fields["member_fields_ms"], dtype=np.float32)
                valid = np.asarray(fields["valid_mask"], dtype=bool)
                distance = np.asarray(fields["distance_km"], dtype=np.float32)
            with np.load(baseline_path) as baseline_file:
                baseline = np.asarray(baseline_file["baseline_field_ms"], dtype=np.float32)
            if members.ndim != 3 or baseline.shape != members.shape[1:] or valid.shape != baseline.shape:
                raise ValueError(f"shape mismatch for {storm}/{observation_id}")
            baseline_tensor = torch.from_numpy(baseline).unsqueeze(0)
            valid_tensor = torch.from_numpy(valid).unsqueeze(0)
            baseline_summary, *_ = _summarize_ensemble(
                baseline_tensor,
                valid_tensor,
                torch.from_numpy(distance),
                quantiles=[0.1, 0.9],
                summary_aggregation="median",
            )
            source_rows.append((storm, row, members, baseline, valid, distance))
            baseline_rows.append(
                {
                    "baseline_msw_ms": baseline_summary["output_msw_ms"],
                    "baseline_robust_peak_ms": baseline_summary["output_robust_peak_ms"],
                }
            )

    output_root = args.output_root
    variant_root = output_root / "variants"
    variant_root.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    variant_manifest: list[dict[str, Any]] = []
    all_variant_metadata: dict[str, Any] = {}
    for gain in gains:
        for cap in [math.inf, *caps]:
            for kernel in kernels:
                name = _variant_name(gain, cap, kernel)
                variant_dir = variant_root / name
                variant_dir.mkdir(parents=True, exist_ok=True)
                output_frames: dict[str, list[pd.DataFrame]] = {storm: [] for storm in storms}
                row_cursor = 0
                for storm, source_row, members, baseline, valid, distance in source_rows:
                    residual = members - baseline[None, ...]
                    transformed = residual * float(gain)
                    if math.isfinite(cap):
                        transformed = np.clip(transformed, -cap, cap)
                    transformed = _smooth_residual(transformed, valid, kernel)
                    transformed = np.clip(
                        baseline[None, ...] + transformed,
                        args.output_min_ms,
                        args.output_max_ms,
                    )
                    transformed[:, ~valid] = np.nan
                    summary, _, _, _, _, _ = _summarize_ensemble(
                        torch.from_numpy(transformed.astype(np.float32, copy=False)),
                        torch.from_numpy(valid).unsqueeze(0),
                        torch.from_numpy(distance),
                        quantiles=list(args.member_quantiles),
                        summary_aggregation=args.summary_aggregation,
                    )
                    frame = source_row.to_frame().T.copy()
                    frame["postprocess_variant"] = name
                    frame["postprocess_gain"] = float(gain)
                    frame["postprocess_residual_cap_ms"] = None if math.isinf(cap) else float(cap)
                    frame["postprocess_median_kernel"] = int(kernel)
                    frame["baseline_msw_ms"] = baseline_rows[row_cursor]["baseline_msw_ms"]
                    frame["baseline_robust_peak_ms"] = baseline_rows[row_cursor]["baseline_robust_peak_ms"]
                    for column, value in summary.items():
                        frame[column] = value
                    output_frames[storm].append(frame)
                    row_cursor += 1
                variant_table = pd.concat(
                    [pd.concat(frames, ignore_index=True) for frames in output_frames.values()],
                    ignore_index=True,
                )
                summary_path = variant_dir / "inference-summary.csv"
                variant_table.to_csv(summary_path, index=False)
                metrics = _metrics_for_table(variant_table)
                payload = {
                    "schema_version": 1,
                    "variant": {
                        "name": name,
                        "gain": float(gain),
                        "residual_cap_ms": None if math.isinf(cap) else float(cap),
                        "median_kernel": int(kernel),
                    },
                    "input_root": str(args.input_root.resolve()),
                    "rows": int(len(variant_table)),
                    "metrics": metrics,
                    "summary": str(summary_path.relative_to(output_root)),
                }
                (variant_dir / "postprocess-evaluation.json").write_text(
                    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                all_variant_metadata[name] = payload
                variant_manifest.append(
                    {
                        "variant": name,
                        "gain": float(gain),
                        "residual_cap_ms": None if math.isinf(cap) else float(cap),
                        "median_kernel": int(kernel),
                        "rows": int(len(variant_table)),
                        "summary_path": str(summary_path.relative_to(output_root)),
                    }
                )
                for metric_name, metric in metrics.items():
                    if isinstance(metric, dict) and "mae_ms" in metric:
                        result_rows.append(
                            {
                                "variant": name,
                                "gain": float(gain),
                                "residual_cap_ms": None if math.isinf(cap) else float(cap),
                                "median_kernel": int(kernel),
                                "metric": metric_name,
                                **metric,
                            }
                        )
    pd.DataFrame(variant_manifest).to_csv(output_root / "variants-manifest.csv", index=False)
    pd.DataFrame(result_rows).to_csv(output_root / "postprocess-results.csv", index=False)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(args.input_root.resolve()),
        "baseline_root": str(args.baseline_root.resolve()),
        "storms": storms,
        "gains": gains,
        "residual_caps_ms": caps,
        "uncapped_control_included": True,
        "median_kernels": kernels,
        "summary_aggregation": args.summary_aggregation,
        "member_quantiles": list(args.member_quantiles),
        "variant_count": len(variant_manifest),
        "outputs": {
            "results_csv": "postprocess-results.csv",
            "variants_manifest": "variants-manifest.csv",
            "variant_summaries": "variants/<variant>/inference-summary.csv",
            "variant_evaluations": "variants/<variant>/postprocess-evaluation.json",
        },
        "variants": all_variant_metadata,
    }
    (output_root / "postprocess-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(variant_manifest)} post-processing variants to {output_root}")


if __name__ == "__main__":
    main()
