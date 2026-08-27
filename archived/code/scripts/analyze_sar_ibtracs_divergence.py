#!/usr/bin/env python3
"""Describe disagreement between SAR scalar wind diagnostics and IBTrACS."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo2wf.config import compose_config, instantiate_datamodule  # noqa: E402
from geo2wf.data.joint_intensity import JointPairedIntensityDataModule  # noqa: E402


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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/intensity-comparisons/sar-ibtracs-divergence.json"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 0:
        parser.error("--bootstrap-repetitions must be non-negative")
    return args


def _interval(values: np.ndarray) -> list[float] | None:
    if not len(values):
        return None
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _bootstrap(
    frame: pd.DataFrame,
    error: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if repetitions == 0:
        return {"repetitions": 0, "seed": seed}
    storms = sorted(frame["storm_id"].astype(str).unique())
    by_storm = {
        storm: np.flatnonzero(frame["storm_id"].astype(str).to_numpy() == storm)
        for storm in storms
    }
    rng = np.random.default_rng(seed)
    mae = np.empty(repetitions, dtype=float)
    bias = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        selected = rng.choice(storms, size=len(storms), replace=True)
        indices = np.concatenate([by_storm[str(storm)] for storm in selected])
        values = error[indices]
        mae[index] = np.mean(np.abs(values))
        bias[index] = np.mean(values)
    return {
        "repetitions": repetitions,
        "seed": seed,
        "mae_ms_95ci": _interval(mae),
        "bias_ms_95ci": _interval(bias),
    }


def summarize_divergence(
    frame: pd.DataFrame,
    sar_column: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    selected = frame.loc[
        np.isfinite(pd.to_numeric(frame[sar_column], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["ibtracs_target_ms"], errors="coerce"))
    ].copy()
    if selected.empty:
        raise ValueError("divergence subset has no finite paired references")
    sar = selected[sar_column].to_numpy(dtype=float)
    ibtracs = selected["ibtracs_target_ms"].to_numpy(dtype=float)
    error = sar - ibtracs
    absolute = np.abs(error)
    pearson = float(np.corrcoef(sar, ibtracs)[0, 1]) if len(selected) > 1 else None
    spearman = (
        float(pd.Series(sar).corr(pd.Series(ibtracs), method="spearman"))
        if len(selected) > 1
        else None
    )
    if pearson is not None and not math.isfinite(pearson):
        pearson = None
    if spearman is not None and not math.isfinite(spearman):
        spearman = None
    return {
        "samples": len(selected),
        "storms": int(selected["storm_id"].nunique()),
        "bias_ms": float(np.mean(error)),
        "mae_ms": float(np.mean(absolute)),
        "rmse_ms": float(np.sqrt(np.mean(np.square(error)))),
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "signed_error_quantiles_ms": {
            f"p{int(100 * q):02d}": float(np.quantile(error, q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "absolute_error_quantiles_ms": {
            f"p{int(100 * q):02d}": float(np.quantile(absolute, q))
            for q in (0.5, 0.75, 0.9, 0.95)
        },
        "storm_bootstrap": _bootstrap(
            selected,
            error,
            repetitions=repetitions,
            seed=seed,
        ),
    }


def _rows(datamodule: JointPairedIntensityDataModule) -> tuple[pd.DataFrame, dict]:
    rows = []
    coverage = {}
    for split in (datamodule.train_split, datamodule.val_split, datamodule.test_split):
        dataset = datamodule._make_dataset(split, augment=False)
        for label in dataset._labels:
            rows.append(
                {
                    "sample_id": str(label["source_sample_id"]),
                    "storm_id": str(label["storm_id"]),
                    "split": split,
                    "observation_timestamp": str(label["observation_timestamp"]),
                    "ibtracs_target_ms": float(label["ibtracs_wind_ms"]),
                    "sar_max_wind_ms": float(label["sar_max_wind_ms"]),
                    "sar_robust_peak_target_ms": float(label["sar_robust_peak_ms"]),
                    "sar_has_valid_center": bool(label["sar_has_valid_center"]),
                    "is_rapid_intensification": bool(label["is_rapid_intensification"]),
                    "ri_24h_change_ms": float(label["ri_24h_change_ms"]),
                }
            )
        center_valid = sum(
            bool(label["sar_has_valid_center"]) for label in dataset._labels
        )
        eligible = len(dataset)
        coverage[split] = {
            "source_samples": dataset.paired_dataset.manifest_sample_count,
            "usable_ibtracs_sar_matches": eligible,
            "matched_center_valid_samples": center_valid,
            "filtered_unbracketed": dataset.filtered_unbracketed_count,
            "filtered_invalid_sar_center": eligible - center_valid,
            "filtered_unusable_sar": dataset.filtered_unusable_sar_count,
            "center_valid_fraction_of_usable_matches": (
                center_valid / eligible if eligible else None
            ),
        }
    total_eligible = sum(
        values["matched_center_valid_samples"] + values["filtered_invalid_sar_center"]
        for values in coverage.values()
    )
    coverage["all"] = {
        key: sum(values[key] for values in coverage.values())
        for key in (
            "source_samples",
            "usable_ibtracs_sar_matches",
            "matched_center_valid_samples",
            "filtered_unbracketed",
            "filtered_invalid_sar_center",
            "filtered_unusable_sar",
        )
    }
    coverage["all"]["center_valid_fraction_of_usable_matches"] = (
        coverage["all"]["matched_center_valid_samples"] / total_eligible
        if total_eligible
        else None
    )
    frame = pd.DataFrame(rows)
    for name, selected in (
        ("rapid_intensification", frame["is_rapid_intensification"]),
        ("non_rapid_intensification", ~frame["is_rapid_intensification"]),
    ):
        subset = frame.loc[selected]
        valid = int(subset["sar_has_valid_center"].sum())
        coverage[name] = {
            "usable_ibtracs_sar_matches": len(subset),
            "matched_center_valid_samples": valid,
            "filtered_invalid_sar_center": len(subset) - valid,
            "center_valid_fraction_of_usable_matches": (
                valid / len(subset) if len(subset) else None
            ),
        }
    return frame, coverage


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SAR–IBTrACS scalar intensity divergence",
        "",
        "Positive bias means the SAR diagnostic exceeds interpolated IBTrACS USA_WIND.",
        "",
        "| Subset | SAR diagnostic | Center-valid / usable | Rate | Samples | Storms | Bias (m/s) | MAE (m/s) | RMSE (m/s) | Pearson | Spearman |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["table"]:
        pearson = row["pearson_correlation"]
        spearman = row["spearman_correlation"]
        lines.append(
            "| {subset} | {diagnostic} | {matched_center_valid_samples}/{usable_ibtracs_sar_matches} | {center_valid_rate:.1%} | {samples} | {storms} | {bias_ms:.3f} | "
            "{mae_ms:.3f} | {rmse_ms:.3f} | {pearson} | {spearman} |".format(
                **row,
                pearson="—" if pearson is None else f"{pearson:.3f}",
                spearman="—" if spearman is None else f"{spearman:.3f}",
            )
        )
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config = compose_config(
        [
            "experiment=intensity_comparison_unet",
            f"data.root={args.paired_root.expanduser().resolve()}",
            f"data.stats_file={(args.paired_root / 'stats.json').expanduser().resolve()}",
            f"data.ibtracs_file={args.ibtracs_file.expanduser().resolve()}",
            "data.intensity_target_source=ibtracs",
            # Build the pre-center-filter matched population so center-valid
            # rates can also be reported for RI and non-RI subgroups.
            "data.require_sar_valid_center=false",
            "data.num_workers=0",
        ]
    )
    datamodule = instantiate_datamodule(config)
    if not isinstance(datamodule, JointPairedIntensityDataModule):
        raise TypeError("divergence analysis requires JointPairedIntensityDataModule")
    candidate_frame, coverage = _rows(datamodule)
    frame = candidate_frame.loc[candidate_frame["sar_has_valid_center"]].reset_index(
        drop=True
    )
    table = []
    details = {}
    subsets: list[tuple[str, pd.DataFrame]] = [("all", frame)]
    subsets.extend(
        (split, frame.loc[frame["split"] == split])
        for split in ("train", "val", "test")
    )
    subsets.extend(
        [
            ("rapid_intensification", frame.loc[frame["is_rapid_intensification"]]),
            (
                "non_rapid_intensification",
                frame.loc[~frame["is_rapid_intensification"]],
            ),
        ]
    )
    for subset_name, subset in subsets:
        if subset.empty:
            continue
        details[subset_name] = {}
        for diagnostic, column in (
            ("sar_max", "sar_max_wind_ms"),
            ("sar_robust_peak", "sar_robust_peak_target_ms"),
        ):
            summary = summarize_divergence(
                subset,
                column,
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed,
            )
            details[subset_name][diagnostic] = summary
            table.append(
                {
                    "subset": subset_name,
                    "diagnostic": diagnostic,
                    "usable_ibtracs_sar_matches": coverage[subset_name][
                        "usable_ibtracs_sar_matches"
                    ],
                    "matched_center_valid_samples": coverage[subset_name][
                        "matched_center_valid_samples"
                    ],
                    "center_valid_rate": coverage[subset_name][
                        "center_valid_fraction_of_usable_matches"
                    ],
                    **summary,
                }
            )
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "SAR diagnostic minus interpolated IBTrACS USA_WIND",
        "sar_robust_peak_fraction": datamodule.sar_robust_peak_fraction,
        "coverage": coverage,
        "details": details,
        "table": table,
        "rows": frame.astype(object)
        .where(pd.notna(frame), None)
        .to_dict(orient="records"),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    pd.DataFrame(table).drop(
        columns=[
            "storm_bootstrap",
            "signed_error_quantiles_ms",
            "absolute_error_quantiles_ms",
        ]
    ).to_csv(output.with_suffix(".csv"), index=False)
    output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload))
    return payload


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
