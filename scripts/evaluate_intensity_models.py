#!/usr/bin/env python3
"""Evaluate field, correction, joint, and encoder-only models on one cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo2wf.config import instantiate_datamodule, load_config_file  # noqa: E402
from geo2wf.data.collation import collate_wind_field_samples  # noqa: E402
from geo2wf.data.encoder_intensity import EncoderIBTrACSDataset  # noqa: E402
from geo2wf.data.intensity import (  # noqa: E402
    UNetIntensityDataset,
    tropical_category_from_wind_ms,
)
from geo2wf.data.joint_intensity import (  # noqa: E402
    JointPairedIntensityDataModule,
)
from geo2wf.models.bottleneck_unet_mlp import (  # noqa: E402
    BottleneckEncoderMLPRegressor,
    BottleneckUNetMLPRegressor,
)
from geo2wf.models.intensity_correction import (  # noqa: E402
    UNetIntensityCorrection,
    rows_for_intensity_reference,
    summarize_intensity_rows,
)


MODEL_LABELS = {
    "unet_raw_max": "U-Net raw field diagnostic",
    "unet_correction": "U-Net + correction",
    "joint_unet_mlp": "Joint U-Net + MLP",
    "encoder_mlp_ibtracs": "U-Net encoder + MLP (IBTrACS only)",
}
MODEL_ORDER = (
    "unet_raw_max",
    "unet_correction",
    "joint_unet_mlp",
    "encoder_mlp_ibtracs",
)


def _json_compatible(value: Any) -> Any:
    """Replace non-finite numeric values with JSON null recursively.

    A missing 24-hour IBTrACS history is represented as NaN inside tensors and
    data frames so metric code can distinguish it from a real zero change. JSON
    has no portable NaN representation, so optional diagnostics are normalized
    only at the report boundary.
    """
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config",
        type=Path,
        required=True,
        help="Resolved field-only U-Net config defining the common paired cohort.",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--correction-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=None,
        help="Optional IBTrACS-only U-Net encoder + MLP checkpoint.",
    )
    parser.add_argument(
        "--intensity-target-source",
        choices=("ibtracs", "sar_robust_peak"),
        default=None,
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.bootstrap_repetitions < 0:
        parser.error("--bootstrap-repetitions must be non-negative")
    return args


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _cohort_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [
        "sample_id",
        "storm_id",
        "split",
        "target_timestamp",
    ]
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"comparison manifest is missing columns: {sorted(missing)}")
    selected = frame.loc[:, columns].sort_values("sample_id")
    serialized = selected.to_csv(
        index=False, lineterminator="\n", float_format="%.9g"
    ).encode("utf-8")
    return {
        "samples": len(selected),
        "storms": int(selected["storm_id"].nunique()),
        "columns": columns,
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }


def _target_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [
        "sample_id",
        "intensity_target_source",
        "target_wind_ms",
        "ibtracs_target_ms",
        "sar_robust_peak_target_ms",
    ]
    missing = set(columns).difference(frame.columns)
    if missing:
        legacy_columns = ["sample_id", "target_wind_ms"]
        legacy_missing = set(legacy_columns).difference(frame.columns)
        if legacy_missing:
            raise ValueError(
                f"comparison manifest is missing columns: {sorted(missing)}"
            )
        selected = frame.loc[:, legacy_columns].sort_values("sample_id")
        serialized = selected.to_csv(
            index=False, lineterminator="\n", float_format="%.9g"
        ).encode("utf-8")
        return {
            "source": "ibtracs",
            "samples": len(selected),
            "columns": legacy_columns,
            "sha256": hashlib.sha256(serialized).hexdigest(),
            "legacy_schema": True,
        }
    selected = frame.loc[:, columns].sort_values("sample_id")
    serialized = selected.to_csv(
        index=False, lineterminator="\n", float_format="%.9g"
    ).encode("utf-8")
    return {
        "source": str(selected["intensity_target_source"].iloc[0]),
        "samples": len(selected),
        "columns": columns,
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }


def _field_statistics() -> np.ndarray:
    # valid pixels, absolute error, squared error, signed error
    return np.zeros(4, dtype=np.float64)


def _accumulate_field(
    statistics: np.ndarray,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    valid = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    error = (prediction - target)[valid].double()
    if not error.numel():
        return
    statistics += np.asarray(
        [
            error.numel(),
            float(error.abs().sum()),
            float(error.square().sum()),
            float(error.sum()),
        ],
        dtype=np.float64,
    )


def _summarize_field(statistics: np.ndarray) -> dict[str, float | int]:
    count = int(statistics[0])
    if count <= 0:
        raise ValueError("field evaluation has no valid pixels")
    return {
        "valid_pixels": count,
        "mae_ms": float(statistics[1] / count),
        "rmse_ms": float(math.sqrt(statistics[2] / count)),
        "bias_ms": float(statistics[3] / count),
    }


def _correction_rows(
    dataset: UNetIntensityDataset,
    checkpoint: Path,
    *,
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict[str, dict[str, Any]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model = UNetIntensityCorrection.load_from_checkpoint(
        checkpoint, map_location="cpu"
    ).eval()
    model.validate_data_spec(dataset.data_spec)
    expected_anchors = set(dataset.samples.get("anchor_statistic", ["max"]))
    if expected_anchors != {model.anchor_statistic}:
        raise ValueError(
            "correction checkpoint anchor does not match the cache target: "
            f"checkpoint={model.anchor_statistic!r}, cache={sorted(expected_anchors)}"
        )
    cache_fraction = float(
        dataset.cache_metadata.get("target", {}).get("sar_robust_peak_fraction", 0.005)
    )
    if not math.isclose(
        model.robust_peak_fraction,
        cache_fraction,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "correction checkpoint robust-peak fraction does not match cache: "
            f"checkpoint={model.robust_peak_fraction}, cache={cache_fraction}"
        )
    model.to(device)
    rows: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for batch in tqdm(loader, desc="evaluate correction", unit="batch"):
            device_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            prediction = model.predict_intensity(device_batch)
            values = {
                "prediction_ms": prediction.output_msw_ms.detach().cpu().tolist(),
                "raw_unet_anchor_ms": prediction.raw_unet_anchor_ms.detach()
                .cpu()
                .tolist(),
                "raw_unet_ms": prediction.raw_unet_max_wind_ms.detach().cpu().tolist(),
                "raw_unet_max_ms": prediction.raw_unet_max_wind_ms.detach()
                .cpu()
                .tolist(),
                "raw_unet_robust_peak_ms": prediction.raw_unet_robust_peak_ms.detach()
                .cpu()
                .tolist(),
                "correction_ms": prediction.correction_ms.detach().cpu().tolist(),
                "prediction_category": prediction.output_category.detach()
                .cpu()
                .tolist(),
                "target_ms": batch["target_wind_ms"].tolist(),
                "target_category": batch["target_category"].tolist(),
                "ibtracs_target_ms": batch["ibtracs_target_ms"].tolist(),
                "sar_robust_peak_target_ms": batch[
                    "sar_robust_peak_target_ms"
                ].tolist(),
                "sar_max_wind_ms": batch["sar_max_wind_ms"].tolist(),
                "is_rapid_intensification": batch["is_rapid_intensification"].tolist(),
                "ri_24h_change_ms": batch["ri_24h_change_ms"].tolist(),
            }
            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                if sample_id in rows:
                    raise ValueError(f"duplicate correction sample ID: {sample_id}")
                rows[sample_id] = {
                    "sample_id": sample_id,
                    "storm_id": str(batch["storm_id"][index]),
                    "observation_timestamp": str(batch["observation_timestamp"][index]),
                    "intensity_target_source": str(
                        batch["intensity_target_source"][index]
                    ),
                    **{name: value[index] for name, value in values.items()},
                }
                rows[sample_id]["raw_unet_ms"] = rows[sample_id]["raw_unet_anchor_ms"]
    return rows


def _assert_common_cohort(
    joint_dataset,
    cache_dataset: UNetIntensityDataset,
    correction_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    joint_ids = set(joint_dataset.samples["sample_id"].astype(str))
    cache_ids = set(cache_dataset.samples["sample_id"].astype(str))
    correction_ids = set(correction_rows)
    if joint_ids != cache_ids or joint_ids != correction_ids:
        detail = {
            "joint_only": sorted(joint_ids - cache_ids)[:5],
            "cache_only": sorted(cache_ids - joint_ids)[:5],
            "missing_correction": sorted(joint_ids - correction_ids)[:5],
        }
        raise ValueError(f"model evaluation cohorts differ: {detail}")
    if len(joint_dataset) != len(joint_ids):
        raise ValueError("joint evaluation sample IDs are not unique")


def _joint_and_field_rows(
    joint_dataset,
    cache_dataset: UNetIntensityDataset,
    correction_rows: Mapping[str, Mapping[str, Any]],
    checkpoint: Path,
    *,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | int]]]:
    loader = DataLoader(
        joint_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_wind_field_samples,
    )
    model = BottleneckUNetMLPRegressor.load_from_checkpoint(
        checkpoint, map_location="cpu"
    ).eval()
    model.validate_data_spec(joint_dataset.data_spec)
    model.to(device)
    cache_indices = {
        str(sample_id): index
        for index, sample_id in enumerate(cache_dataset.samples["sample_id"])
    }
    joint_rows: list[dict[str, Any]] = []
    unet_field = _field_statistics()
    joint_field = _field_statistics()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="evaluate joint model", unit="batch"):
            device_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            prediction = model.predict_joint(device_batch)
            joint_prediction = prediction.central_physical.detach().cpu()
            intensity_prediction = prediction.intensity_prediction_ms.detach().cpu()
            target_field = batch["target_physical"]
            common_mask = batch["target_mask"].bool() & batch["condition_mask"].bool()

            cached_samples = [
                cache_dataset[cache_indices[str(sample_id)]]
                for sample_id in batch["sample_id"]
            ]
            cached_field = torch.stack(
                [sample["wind_field"] for sample in cached_samples]
            ).unsqueeze(1)
            cached_mask = torch.stack(
                [sample["valid_mask"] for sample in cached_samples]
            ).unsqueeze(1)
            _accumulate_field(
                unet_field, cached_field, target_field, common_mask & cached_mask
            )
            _accumulate_field(joint_field, joint_prediction, target_field, common_mask)

            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                reference = correction_rows[sample_id]
                target_ms = float(batch["intensity_target_ms"][index])
                if not math.isclose(
                    target_ms,
                    float(reference["target_ms"]),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-4,
                ):
                    raise ValueError(
                        f"target mismatch for {sample_id}: joint={target_ms}, "
                        f"cache={reference['target_ms']}"
                    )
                storm_id = str(batch["meta"][index]["storm_id"])
                if storm_id != str(reference["storm_id"]):
                    raise ValueError(f"storm mismatch for sample {sample_id}")
                predicted_ms = float(intensity_prediction[index])
                joint_rows.append(
                    {
                        "sample_id": sample_id,
                        "storm_id": storm_id,
                        "observation_timestamp": reference["observation_timestamp"],
                        "prediction_ms": predicted_ms,
                        "target_ms": target_ms,
                        "raw_unet_ms": float(reference["raw_unet_ms"]),
                        "raw_unet_anchor_ms": float(reference["raw_unet_anchor_ms"]),
                        "raw_unet_max_ms": float(reference["raw_unet_max_ms"]),
                        "raw_unet_robust_peak_ms": float(
                            reference["raw_unet_robust_peak_ms"]
                        ),
                        "ibtracs_target_ms": float(reference["ibtracs_target_ms"]),
                        "sar_robust_peak_target_ms": float(
                            reference["sar_robust_peak_target_ms"]
                        ),
                        "sar_max_wind_ms": float(reference["sar_max_wind_ms"]),
                        "is_rapid_intensification": bool(
                            reference["is_rapid_intensification"]
                        ),
                        "ri_24h_change_ms": float(reference["ri_24h_change_ms"]),
                        "intensity_target_source": str(
                            reference["intensity_target_source"]
                        ),
                        "correction_ms": predicted_ms - float(reference["raw_unet_ms"]),
                        "prediction_category": tropical_category_from_wind_ms(
                            predicted_ms
                        ),
                        "target_category": int(reference["target_category"]),
                    }
                )
    return joint_rows, {
        "unet_raw_max": _summarize_field(unet_field),
        "joint_unet_mlp": _summarize_field(joint_field),
    }


def _encoder_rows(
    dataset: EncoderIBTrACSDataset,
    correction_rows: Mapping[str, Mapping[str, Any]],
    checkpoint: Path,
    *,
    batch_size: int,
    num_workers: int,
    device: str,
) -> list[dict[str, Any]]:
    """Evaluate the scalar-only encoder on the exact comparison cohort."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_wind_field_samples,
    )
    model = BottleneckEncoderMLPRegressor.load_from_checkpoint(
        checkpoint, map_location="cpu"
    ).eval()
    model.validate_data_spec(dataset.data_spec)
    model.to(device)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="evaluate encoder-only model", unit="batch"):
            device_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            predictions = (
                model.predict_batch(device_batch).intensity_prediction_ms.detach().cpu()
            )
            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                reference = correction_rows[sample_id]
                ibtracs_target = float(reference["ibtracs_target_ms"])
                dataset_target = float(batch["intensity_target_ms"][index])
                if not math.isclose(
                    dataset_target,
                    ibtracs_target,
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-4,
                ):
                    raise ValueError(
                        f"encoder IBTrACS target mismatch for {sample_id}: "
                        f"dataset={dataset_target}, cache={ibtracs_target}"
                    )
                predicted_ms = float(predictions[index])
                rows.append(
                    {
                        **dict(reference),
                        "prediction_ms": predicted_ms,
                        "target_ms": ibtracs_target,
                        "target_category": tropical_category_from_wind_ms(
                            ibtracs_target
                        ),
                        "correction_ms": predicted_ms - float(reference["raw_unet_ms"]),
                        "prediction_category": tropical_category_from_wind_ms(
                            predicted_ms
                        ),
                    }
                )
    expected = set(dataset.samples["sample_id"].astype(str))
    actual = {str(row["sample_id"]) for row in rows}
    if actual != expected or actual != set(correction_rows):
        raise ValueError(
            "encoder-only evaluation cohort differs from comparison cohort"
        )
    return rows


def _raw_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        raw = float(row["raw_unet_ms"])
        category = tropical_category_from_wind_ms(raw)
        result.append(
            {
                **dict(row),
                "prediction_ms": raw,
                "correction_ms": 0.0,
                "prediction_category": category,
            }
        )
    return result


def _rows_by_reference(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]], reference: str
) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for model_name, rows in rows_by_model.items():
        projected = rows_for_intensity_reference(rows, reference)
        if model_name == "unet_raw_max":
            for row in projected:
                row["prediction_ms"] = float(row["raw_unet_ms"])
                row["correction_ms"] = 0.0
                row["prediction_category"] = tropical_category_from_wind_ms(
                    float(row["prediction_ms"])
                )
        result[model_name] = projected
    return result


def _reference_evaluation(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    result = {}
    for reference in ("ibtracs", "sar_robust_peak"):
        reference_rows = _rows_by_reference(rows_by_model, reference)
        if not all(reference_rows.values()):
            continue
        overall = {
            name: summarize_intensity_rows(rows)
            for name, rows in reference_rows.items()
        }
        overall_bootstrap = _cluster_bootstrap(
            reference_rows,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        )
        ri_rows = {
            name: [
                row for row in rows if bool(row.get("is_rapid_intensification", False))
            ]
            for name, rows in reference_rows.items()
        }
        ri = None
        ri_bootstrap = None
        if all(ri_rows.values()):
            ri = {
                name: summarize_intensity_rows(rows) for name, rows in ri_rows.items()
            }
            ri_bootstrap = _cluster_bootstrap(
                ri_rows,
                repetitions=bootstrap_repetitions,
                seed=bootstrap_seed,
            )
        result[reference] = {
            "overall": overall,
            "overall_storm_bootstrap": overall_bootstrap,
            "rapid_intensification": ri,
            "rapid_intensification_storm_bootstrap": ri_bootstrap,
        }
    return result


def _percentile_interval(values: np.ndarray) -> list[float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return [float(lower), float(upper)]


def _cluster_bootstrap(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if repetitions == 0:
        return {"repetitions": 0, "seed": seed, "models": {}}
    reference_rows = rows_by_model["unet_raw_max"]
    sample_order = [str(row["sample_id"]) for row in reference_rows]
    storms = sorted({str(row["storm_id"]) for row in reference_rows})
    indices_by_storm = {
        storm: np.asarray(
            [
                index
                for index, row in enumerate(reference_rows)
                if str(row["storm_id"]) == storm
            ],
            dtype=int,
        )
        for storm in storms
    }
    errors: dict[str, np.ndarray] = {}
    for model_name, rows in rows_by_model.items():
        indexed = {str(row["sample_id"]): row for row in rows}
        if set(indexed) != set(sample_order):
            raise ValueError("bootstrap model rows do not share exact sample IDs")
        errors[model_name] = np.asarray(
            [
                abs(
                    float(indexed[sample_id]["prediction_ms"])
                    - float(indexed[sample_id]["target_ms"])
                )
                for sample_id in sample_order
            ],
            dtype=float,
        )

    rng = np.random.default_rng(seed)
    mae_samples = {name: np.empty(repetitions, dtype=float) for name in rows_by_model}
    macro_samples = {name: np.empty(repetitions, dtype=float) for name in rows_by_model}
    for repetition in range(repetitions):
        selected_storms = rng.choice(storms, size=len(storms), replace=True)
        selected_indices = np.concatenate(
            [indices_by_storm[str(storm)] for storm in selected_storms]
        )
        for model_name, absolute_error in errors.items():
            mae_samples[model_name][repetition] = absolute_error[
                selected_indices
            ].mean()
            macro_samples[model_name][repetition] = np.mean(
                [
                    absolute_error[indices_by_storm[str(storm)]].mean()
                    for storm in selected_storms
                ]
            )

    baseline = mae_samples["unet_raw_max"]
    models = {}
    for model_name in rows_by_model:
        models[model_name] = {
            "mae_ms_95ci": _percentile_interval(mae_samples[model_name]),
            "storm_macro_mae_ms_95ci": _percentile_interval(macro_samples[model_name]),
            "mae_delta_vs_unet_raw_max_ms_95ci": _percentile_interval(
                mae_samples[model_name] - baseline
            ),
        }
    return {"repetitions": repetitions, "seed": seed, "models": models}


def _table_rows(
    summaries: Mapping[str, Mapping[str, Any]],
    field_metrics: Mapping[str, Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    bootstrap_models = bootstrap.get("models", {})
    baseline_mae = summaries["unet_raw_max"]["regression"]["mae_ms"]
    for model_name in MODEL_ORDER:
        if model_name not in summaries:
            continue
        summary = summaries[model_name]
        field = field_metrics.get(model_name, {})
        interval = bootstrap_models.get(model_name, {}).get("mae_ms_95ci", [None, None])
        delta_interval = bootstrap_models.get(model_name, {}).get(
            "mae_delta_vs_unet_raw_max_ms_95ci", [None, None]
        )
        rows.append(
            {
                "model": MODEL_LABELS[model_name],
                "model_key": model_name,
                "samples": summary["samples"],
                "storms": summary["storms"],
                "intensity_mae_ms": summary["regression"]["mae_ms"],
                "intensity_mae_95ci_low_ms": interval[0],
                "intensity_mae_95ci_high_ms": interval[1],
                "intensity_mae_delta_vs_unet_raw_max_ms": summary["regression"][
                    "mae_ms"
                ]
                - baseline_mae,
                "intensity_mae_delta_95ci_low_ms": delta_interval[0],
                "intensity_mae_delta_95ci_high_ms": delta_interval[1],
                "intensity_rmse_ms": summary["regression"]["rmse_ms"],
                "intensity_bias_ms": summary["regression"]["bias_ms"],
                "storm_macro_mae_ms": summary["storm_macro_mae_ms"],
                "category_accuracy": summary["category"]["accuracy"],
                "category_macro_f1": summary["category"]["macro_f1"],
                "within_one_category_accuracy": summary["category"][
                    "within_one_accuracy"
                ],
                "field_mae_ms": field.get("mae_ms"),
                "field_rmse_ms": field.get("rmse_ms"),
                "field_bias_ms": field.get("bias_ms"),
            }
        )
    return rows


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _markdown_result_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Model | Samples | Storms | Intensity MAE (m/s; 95% CI) | Δ MAE vs raw U-Net (m/s; 95% CI) | Intensity RMSE (m/s) | Intensity bias (m/s) | Storm-macro MAE (m/s) | Category accuracy | Category macro F1 | Within one category | Field MAE (m/s) | Field RMSE (m/s) | Field bias (m/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        low = row["intensity_mae_95ci_low_ms"]
        high = row["intensity_mae_95ci_high_ms"]
        mae = _format_number(row["intensity_mae_ms"])
        mae_ci = (
            f"{mae} ({_format_number(low)}–{_format_number(high)})"
            if low is not None and high is not None
            else mae
        )
        delta_low = row["intensity_mae_delta_95ci_low_ms"]
        delta_high = row["intensity_mae_delta_95ci_high_ms"]
        delta = _format_number(row["intensity_mae_delta_vs_unet_raw_max_ms"])
        delta_ci = (
            f"{delta} ({_format_number(delta_low)}–{_format_number(delta_high)})"
            if delta_low is not None and delta_high is not None
            else delta
        )
        lines.append(
            "| {model} | {samples} | {storms} | {mae_ci} | {delta_ci} | {rmse} | "
            "{bias} | {macro} | {accuracy} | {f1} | {within} | {field_mae} | "
            "{field_rmse} | {field_bias} |".format(
                model=row["model"],
                samples=row["samples"],
                storms=row["storms"],
                mae_ci=mae_ci,
                delta_ci=delta_ci,
                rmse=_format_number(row["intensity_rmse_ms"]),
                bias=_format_number(row["intensity_bias_ms"]),
                macro=_format_number(row["storm_macro_mae_ms"]),
                accuracy=_format_number(row["category_accuracy"]),
                f1=_format_number(row["category_macro_f1"]),
                within=_format_number(row["within_one_category_accuracy"]),
                field_mae=_format_number(row["field_mae_ms"]),
                field_rmse=_format_number(row["field_rmse_ms"]),
                field_bias=_format_number(row["field_bias_ms"]),
            )
        )
    return lines


def _methodology_markdown(
    *,
    split: str,
    samples: int,
    storms: int,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> list[str]:
    interpretation = (
        "These are validation/model-selection results, not a final held-out "
        "generalization estimate. Hyperparameters and model choices should be frozen "
        "before evaluating the test split."
        if split == "val"
        else "These are held-out test results and should not be used for further model selection."
    )
    return [
        "## What the models predict",
        "",
        "- **U-Net raw field maximum** is the largest predicted wind speed over all "
        "finite, valid pixels in the separately trained U-Net field.",
        "- **U-Net + correction** starts from that same raw maximum and adds the "
        "output of a separately trained correction network. The correction network "
        "sees the frozen U-Net field, its validity mask, distance to the storm center, "
        "and contemporaneously available metadata. Its final intensity is clamped to "
        "be non-negative.",
        "- **Joint U-Net + MLP** directly predicts scalar maximum wind from an MLP "
        "attached to the U-Net bottleneck. Its field reconstruction and scalar head "
        "are optimized jointly and share the encoder.",
        "",
        "The correction model emits only a scalar, so field MAE, RMSE, and bias are "
        "not applicable and are shown as —.",
        "",
        "## Evaluation cohort and target",
        "",
        f"The `{split}` cohort contains **{samples} samples from {storms} storms**. "
        "Every row in every table uses the exact same sample IDs and storm-disjoint "
        "split. The scalar target is IBTrACS `USA_WIND`, expressed in m/s and "
        "linearly interpolated to the image timestamp only when the surrounding "
        "IBTrACS fixes are no more than three hours apart.",
        "",
        "## Metric definitions",
        "",
        "Let `prediction - target` be the signed scalar intensity error for one sample.",
        "",
        "| Metric | How it is calculated | How to read it |",
        "|---|---|---|",
        "| **Samples / Storms** | Counts of image observations and unique storm IDs in the evaluation cohort. | These counts must match across model rows for a paired comparison. |",
        "| **Intensity MAE** | Mean of `abs(prediction - target)` over samples. | Typical absolute scalar-intensity miss; lower is better. The parenthesized range is the 95% storm-bootstrap interval. |",
        "| **Δ MAE vs raw U-Net** | A model's MAE minus the raw U-Net MAE, computed on the same bootstrap resample. | Negative values favor the model; a 95% interval excluding zero indicates a consistent paired improvement at the sampled-storm level. |",
        "| **Intensity RMSE** | Square root of the mean squared scalar error over samples. | Penalizes large intensity misses more heavily than MAE; lower is better. |",
        "| **Intensity bias** | Mean of `prediction - target` over samples. | Zero is ideal; negative means systematic underprediction and positive means overprediction. Opposite errors can cancel. |",
        "| **Storm-macro MAE** | Compute MAE separately for every storm, then take the unweighted mean over storms. | Gives each storm equal weight regardless of how many images it contributes; lower is better. |",
        "| **Category accuracy** | Fraction whose predicted and target intensity categories match exactly. | Higher is better; range 0–1. |",
        "| **Category macro F1** | For each target category present, calculate the harmonic mean of precision and recall, then average categories equally. | Rewards balanced category performance rather than dominance by frequent categories; higher is better. |",
        "| **Within one category** | Fraction for which the numeric predicted category differs from the target category by at most one. | Measures tolerance to a one-bin miss; higher is better. |",
        "| **Field MAE** | Mean absolute U-Net-versus-SAR wind error over all valid pixels. | Typical pixel-level wind-field miss; lower is better. |",
        "| **Field RMSE** | Square root of mean squared U-Net-versus-SAR error over all valid pixels. | More sensitive than field MAE to large pixel errors; lower is better. |",
        "| **Field bias** | Mean signed U-Net-minus-SAR error over all valid pixels. | Zero is ideal; negative means field underprediction and positive means overprediction. |",
        "",
        "### Intensity categories",
        "",
        "Continuous one-minute wind is converted without rounding using these "
        "thresholds: tropical depression `<34 kt`; tropical storm `34–<64 kt`; "
        "Category 1 `64–<83 kt`; Category 2 `83–<96 kt`; Category 3 `96–<113 kt`; "
        "Category 4 `113–<137 kt`; and Category 5 `≥137 kt` "
        "(`1 kt = 0.514444 m/s`).",
        "",
        "### Valid pixels and uncertainty",
        "",
        "Field metrics pool pixels over the intersection of the SAR target mask, "
        "condition mask, and finite predictions/targets; the cached raw U-Net also "
        "uses its exported validity mask. They are therefore pixel-weighted, not "
        "sample- or storm-macro metrics.",
        "",
        f"The reported 95% intervals use **{bootstrap_repetitions:,} paired "
        f"cluster-bootstrap repetitions over storms** with seed `{bootstrap_seed}`. "
        "Each repetition samples storm IDs with replacement and evaluates all models "
        "on that identical resample, preserving within-storm dependence and the "
        "pairing between models. Bounds are the 2.5th and 97.5th percentiles.",
        "",
        interpretation,
        "",
    ]


def _markdown_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 42,
) -> str:
    if not rows:
        raise ValueError("cannot render an empty comparison table")
    lines = [
        f"# Intensity model comparison ({split})",
        "",
        *_markdown_result_table(rows),
        "",
        *_methodology_markdown(
            split=split,
            samples=int(rows[0]["samples"]),
            storms=int(rows[0]["storms"]),
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_seed=bootstrap_seed,
        ),
    ]
    return "\n".join(lines)


def evaluate_models(args: argparse.Namespace) -> dict[str, Any]:
    required_paths = [
        args.data_config,
        args.cache_root,
        args.joint_checkpoint,
        args.correction_checkpoint,
    ]
    if args.encoder_checkpoint is not None:
        required_paths.append(args.encoder_checkpoint)
    for path in required_paths:
        if not Path(path).expanduser().exists():
            raise FileNotFoundError(path)

    cache_dataset = UNetIntensityDataset(args.cache_root, args.split)
    cache_metadata = cache_dataset.cache_metadata
    target_source = (
        "ibtracs"
        if int(cache_metadata.get("schema_version", 1)) == 1
        else str(cache_metadata.get("target", {}).get("source", "ibtracs"))
    )
    if args.encoder_checkpoint is not None and target_source != "ibtracs":
        raise ValueError(
            "the encoder-only checkpoint is trained only for the IBTrACS target"
        )
    if (
        args.intensity_target_source is not None
        and args.intensity_target_source != target_source
    ):
        raise ValueError(
            "requested target source does not match cache: "
            f"{args.intensity_target_source} != {target_source}"
        )
    config = load_config_file(args.data_config)
    config["data"]["intensity_target_source"] = target_source
    config["data"]["require_sar_valid_center"] = True
    datamodule = instantiate_datamodule(config)
    if not isinstance(datamodule, JointPairedIntensityDataModule):
        raise TypeError("comparison requires JointPairedIntensityDataModule")
    joint_dataset = datamodule._make_dataset(args.split, augment=False)
    if cache_metadata.get("source_kind") != "joint_paired_intensity_cohort":
        raise ValueError(
            "comparison requires a cache exported from the exact joint cohort"
        )
    configured_sha = str(cache_metadata.get("unet_config", {}).get("sha256", ""))
    if configured_sha != _sha256(args.data_config):
        raise ValueError(
            "comparison data config does not match the config that created the "
            "U-Net cache"
        )

    correction = _correction_rows(
        cache_dataset,
        args.correction_checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    _assert_common_cohort(joint_dataset, cache_dataset, correction)
    joint_rows, field_metrics = _joint_and_field_rows(
        joint_dataset,
        cache_dataset,
        correction,
        args.joint_checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    correction_rows = [correction[str(row["sample_id"])] for row in joint_rows]
    raw_rows = _raw_rows(correction_rows)
    rows_by_model = {
        "unet_raw_max": raw_rows,
        "unet_correction": correction_rows,
        "joint_unet_mlp": joint_rows,
    }
    if args.encoder_checkpoint is not None:
        encoder_dataset = EncoderIBTrACSDataset(
            joint_dataset.paired_dataset,
            datamodule.ibtracs_tracks,
            joint_dataset.samples["sample_id"].astype(str).tolist(),
            max_bracket_hours=datamodule.max_ibtracs_bracket_hours,
        )
        rows_by_model["encoder_mlp_ibtracs"] = _encoder_rows(
            encoder_dataset,
            correction,
            args.encoder_checkpoint,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
    summaries = {
        name: summarize_intensity_rows(rows) for name, rows in rows_by_model.items()
    }
    reference_evaluation = _reference_evaluation(
        rows_by_model,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    if int(cache_metadata.get("schema_version", 1)) >= 2 and not any(
        details.get("rapid_intensification")
        for details in reference_evaluation.values()
    ):
        raise ValueError("final validation evaluation has no RI samples")
    bootstrap = _cluster_bootstrap(
        rows_by_model,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    table_rows = _table_rows(summaries, field_metrics, bootstrap)
    cache_manifest = pd.read_csv(
        args.cache_root / args.split / "manifest.csv", keep_default_na=False
    )
    payload = _json_compatible(
        {
            "schema_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "split": args.split,
            "interpretation": (
                "validation_model_selection_diagnostic"
                if args.split == "val"
                else "held_out_test_evaluation"
            ),
            "conditioning": {
                "use_era5": bool(config["data"].get("use_era5", True)),
                "label": (
                    "with_era5"
                    if config["data"].get("use_era5", True)
                    else "without_era5"
                ),
            },
            "cohort": _cohort_fingerprint(cache_manifest),
            "target_fingerprint": _target_fingerprint(cache_manifest),
            "data_config": {
                "path": str(args.data_config.resolve()),
                "sha256": _sha256(args.data_config),
            },
            "cache": {
                "path": str(args.cache_root.resolve()),
                "metadata": cache_metadata,
            },
            "checkpoints": {
                "unet": cache_metadata["unet_checkpoint"],
                "correction": {
                    "path": str(args.correction_checkpoint.resolve()),
                    "sha256": _sha256(args.correction_checkpoint),
                },
                "joint": {
                    "path": str(args.joint_checkpoint.resolve()),
                    "sha256": _sha256(args.joint_checkpoint),
                },
                **(
                    {
                        "encoder": {
                            "path": str(args.encoder_checkpoint.resolve()),
                            "sha256": _sha256(args.encoder_checkpoint),
                        }
                    }
                    if args.encoder_checkpoint is not None
                    else {}
                ),
            },
            "models": {
                name: {
                    "label": MODEL_LABELS[name],
                    "intensity": summaries[name],
                    "field": field_metrics.get(name),
                }
                for name in rows_by_model
            },
            "reference_evaluation": reference_evaluation,
            "prediction_rows": {name: rows for name, rows in rows_by_model.items()},
            "paired_storm_bootstrap": bootstrap,
            "table": table_rows,
        }
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    pd.DataFrame(table_rows).to_csv(output.with_suffix(".csv"), index=False)
    output.with_suffix(".md").write_text(
        _markdown_table(
            table_rows,
            split=args.split,
            bootstrap_repetitions=args.bootstrap_repetitions,
            bootstrap_seed=args.bootstrap_seed,
        ),
        encoding="utf-8",
    )
    print(
        _markdown_table(
            table_rows,
            split=args.split,
            bootstrap_repetitions=args.bootstrap_repetitions,
            bootstrap_seed=args.bootstrap_seed,
        )
    )
    print(
        f"Wrote {output}, {output.with_suffix('.csv')}, and {output.with_suffix('.md')}"
    )
    return payload


def main() -> None:
    evaluate_models(parse_args())


if __name__ == "__main__":
    main()
