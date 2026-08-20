#!/usr/bin/env python3
"""Evaluate raw U-Net, corrected U-Net, and joint U-Net+MLP on one cohort."""

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
from geo2wf.data.intensity import (  # noqa: E402
    UNetIntensityDataset,
    tropical_category_from_wind_ms,
)
from geo2wf.data.joint_intensity import (  # noqa: E402
    JointPairedIntensityDataModule,
)
from geo2wf.models.bottleneck_unet_mlp import (  # noqa: E402
    BottleneckUNetMLPRegressor,
)
from geo2wf.models.intensity_correction import (  # noqa: E402
    UNetIntensityCorrection,
    summarize_intensity_rows,
)


MODEL_LABELS = {
    "unet_raw_max": "U-Net raw field maximum",
    "unet_correction": "U-Net + correction",
    "joint_unet_mlp": "Joint U-Net + MLP",
}


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
        "target_wind_ms",
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
                "raw_unet_ms": prediction.raw_unet_max_wind_ms.detach().cpu().tolist(),
                "correction_ms": prediction.correction_ms.detach().cpu().tolist(),
                "prediction_category": prediction.output_category.detach()
                .cpu()
                .tolist(),
                "target_ms": batch["target_wind_ms"].tolist(),
                "target_category": batch["target_category"].tolist(),
            }
            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                if sample_id in rows:
                    raise ValueError(f"duplicate correction sample ID: {sample_id}")
                rows[sample_id] = {
                    "sample_id": sample_id,
                    "storm_id": str(batch["storm_id"][index]),
                    "observation_timestamp": str(batch["observation_timestamp"][index]),
                    **{name: value[index] for name, value in values.items()},
                }
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
            intensity_prediction = prediction.ibtracs_max_wind_ms.detach().cpu()
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
    for model_name in ("unet_raw_max", "unet_correction", "joint_unet_mlp"):
        summary = summaries[model_name]
        field = field_metrics.get(model_name, {})
        interval = bootstrap_models.get(model_name, {}).get("mae_ms_95ci", [None, None])
        rows.append(
            {
                "model": MODEL_LABELS[model_name],
                "model_key": model_name,
                "samples": summary["samples"],
                "storms": summary["storms"],
                "intensity_mae_ms": summary["regression"]["mae_ms"],
                "intensity_mae_95ci_low_ms": interval[0],
                "intensity_mae_95ci_high_ms": interval[1],
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


def _markdown_table(rows: Sequence[Mapping[str, Any]], *, split: str) -> str:
    lines = [
        f"# Intensity model comparison ({split})",
        "",
        "All rows use the exact same sample IDs, storm-disjoint split, and "
        "interpolated IBTrACS USA_WIND targets. MAE confidence intervals use a "
        "paired cluster bootstrap over storms.",
        "",
        "| Model | N | Intensity MAE m/s (95% CI) | RMSE | Bias | Storm-macro MAE | Category acc. | Macro F1 | Within one | Field MAE | Field RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
        lines.append(
            "| {model} | {samples} | {mae_ci} | {rmse} | {bias} | {macro} | "
            "{accuracy} | {f1} | {within} | {field_mae} | {field_rmse} |".format(
                model=row["model"],
                samples=row["samples"],
                mae_ci=mae_ci,
                rmse=_format_number(row["intensity_rmse_ms"]),
                bias=_format_number(row["intensity_bias_ms"]),
                macro=_format_number(row["storm_macro_mae_ms"]),
                accuracy=_format_number(row["category_accuracy"]),
                f1=_format_number(row["category_macro_f1"]),
                within=_format_number(row["within_one_category_accuracy"]),
                field_mae=_format_number(row["field_mae_ms"]),
                field_rmse=_format_number(row["field_rmse_ms"]),
            )
        )
    lines.extend(
        [
            "",
            "The correction model has no field metrics because it emits only a scalar. "
            "Validation results are model-selection diagnostics; use the untouched test "
            "split once after choices are frozen for the final generalization claim.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_models(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.data_config,
        args.cache_root,
        args.joint_checkpoint,
        args.correction_checkpoint,
    ):
        if not Path(path).expanduser().exists():
            raise FileNotFoundError(path)

    config = load_config_file(args.data_config)
    datamodule = instantiate_datamodule(config)
    if not isinstance(datamodule, JointPairedIntensityDataModule):
        raise TypeError("comparison requires JointPairedIntensityDataModule")
    joint_dataset = datamodule._make_dataset(args.split, augment=False)
    cache_dataset = UNetIntensityDataset(args.cache_root, args.split)
    cache_metadata = cache_dataset.cache_metadata
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
    summaries = {
        name: summarize_intensity_rows(rows) for name, rows in rows_by_model.items()
    }
    bootstrap = _cluster_bootstrap(
        rows_by_model,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    table_rows = _table_rows(summaries, field_metrics, bootstrap)
    cache_manifest = pd.read_csv(
        args.cache_root / args.split / "manifest.csv", keep_default_na=False
    )
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "interpretation": (
            "validation_model_selection_diagnostic"
            if args.split == "val"
            else "held_out_test_evaluation"
        ),
        "cohort": _cohort_fingerprint(cache_manifest),
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
        },
        "models": {
            name: {
                "label": MODEL_LABELS[name],
                "intensity": summaries[name],
                "field": field_metrics.get(name),
            }
            for name in MODEL_LABELS
        },
        "paired_storm_bootstrap": bootstrap,
        "table": table_rows,
    }

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
        _markdown_table(table_rows, split=args.split), encoding="utf-8"
    )
    print(_markdown_table(table_rows, split=args.split))
    print(
        f"Wrote {output}, {output.with_suffix('.csv')}, and {output.with_suffix('.md')}"
    )
    return payload


def main() -> None:
    evaluate_models(parse_args())


if __name__ == "__main__":
    main()
