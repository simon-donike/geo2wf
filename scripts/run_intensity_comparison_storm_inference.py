#!/usr/bin/env python3
"""Run the final intensity comparison over complete validation storms.

One invocation evaluates one conditioning regime (with or without ERA5). It
loads every GEO observation for the requested storms from the inference
manifest, predicts with the separately trained U-Net and correction head and
the jointly trained U-Net+MLP, and joins a linearly interpolated IBTrACS
``USA_WIND`` target at the GEO timestamp.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo2wf.data.intensity import tropical_category_from_wind_ms  # noqa: E402
from geo2wf.data.joint_intensity import (  # noqa: E402
    _interpolate_ibtracs_wind,
    _load_ibtracs_tracks,
    _ri_diagnostics,
)
from geo2wf.models.bottleneck_unet_mlp import (  # noqa: E402
    BottleneckUNetMLPRegressor,
)
from geo2wf.models.deterministic_residual import (  # noqa: E402
    ERA5ResidualRegressor,
)
from geo2wf.models.intensity_correction import (  # noqa: E402
    UNetIntensityCorrection,
)
from scripts.export_geo_sar_geotiffs import ERA5_CHANNELS, _read_manifest  # noqa: E402
from scripts.run_storm_unet_inference import (  # noqa: E402
    _corrected_intensity,
    _prepare_sample,
    _sha256,
    _storm_intensity_context,
    _validate_correction_provenance,
)


STORMS = ("AL082025", "EP112025", "EP182023")
STORM_NAMES = {
    "AL082025": "Humberto",
    "EP112025": "Kiko",
    "EP182023": "Otis",
}
DATA_ROOT = ROOT / "inference" / "inf_data"
PAIRED_ROOT = ROOT / "data" / "geotiff" / "geo_sar_10bands_era5_v2_pmw"
IBTRACS_FILE = ROOT / "data" / "IBTrACs" / "ibtracs.ALL.list.v04r01.csv"
COMPARISON_ROOT = ROOT / "logs" / "intensity-comparisons"
DEFAULT_OUTPUT_ROOT = COMPARISON_ROOT / "three-storm-inference"
REGIMES = ("with", "without")
MODEL_COLUMNS = {
    "unet_raw_max": "unet_raw_max_ms",
    "unet_correction": "unet_correction_ms",
    "joint_unet_mlp": "joint_unet_mlp_ms",
}
ABLATION_COLUMNS = {
    "ablation_max_wind_only": "ablation_max_wind_only_ms",
    "ablation_max_wind_radii": "ablation_max_wind_radii_ms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5", choices=REGIMES, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATA_ROOT / "index-files" / "observation_manifest_v6.csv",
    )
    parser.add_argument("--stats", type=Path, default=PAIRED_ROOT / "stats.json")
    parser.add_argument("--ibtracs-file", type=Path, default=IBTRACS_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--comparison-run",
        type=Path,
        help=(
            "Completed intensity-comparison workflow directory (or workflow.json). "
            "Its checkpoint artifacts are used unless explicitly overridden."
        ),
    )
    parser.add_argument("--unet-checkpoint", type=Path, default=None)
    parser.add_argument("--correction-checkpoint", type=Path, default=None)
    parser.add_argument("--joint-checkpoint", type=Path, default=None)
    parser.add_argument("--intensity-cache-metadata", type=Path, default=None)
    parser.add_argument(
        "--ablation-max-wind-checkpoint",
        type=Path,
        help="Optional ERA5 joint-model checkpoint from the max-wind-only arm.",
    )
    parser.add_argument(
        "--ablation-radii-checkpoint",
        type=Path,
        help="Optional ERA5 joint-model checkpoint from the radii-supervised arm.",
    )
    parser.add_argument("--storms", nargs="+", default=list(STORMS))
    parser.add_argument("--max-ibtracs-bracket-hours", type=float, default=3.0)
    parser.add_argument("--ri-threshold-kt", type=float, default=30.0)
    parser.add_argument("--ri-window-hours", type=float, default=24.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug rows per storm")
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Zero-based deterministic observation shard (requires --num-shards)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of deterministic observation shards",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_ibtracs_bracket_hours <= 0:
        parser.error("--max-ibtracs-bracket-hours must be positive")
    if args.ri_threshold_kt <= 0 or args.ri_window_hours <= 0:
        parser.error("RI threshold and window must be positive")
    if (args.shard_index is None) != (args.num_shards is None):
        parser.error("--shard-index and --num-shards must be provided together")
    if args.num_shards is not None and args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if args.shard_index is not None and not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= index < --num-shards")
    if args.era5 == "without" and (
        args.ablation_max_wind_checkpoint is not None
        or args.ablation_radii_checkpoint is not None
    ):
        parser.error("radii-ablation checkpoints are ERA5-conditioned")
    return args


def _regime_paths(args: argparse.Namespace) -> dict[str, Path]:
    artifacts: Mapping[str, Any] = {}
    if args.comparison_run is not None:
        workflow_path = Path(args.comparison_run).expanduser().resolve()
        if workflow_path.is_dir():
            workflow_path = workflow_path / "workflow.json"
        if not workflow_path.is_file():
            raise FileNotFoundError(workflow_path)
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        if workflow.get("status") != "completed":
            raise ValueError(f"comparison workflow is not completed: {workflow_path}")
        if workflow.get("era5") != args.era5:
            raise ValueError(
                "comparison workflow conditioning does not match --era5: "
                f"{workflow.get('era5')!r} != {args.era5!r}"
            )
        artifacts = workflow.get("artifacts", {})

    cache_root = artifacts.get("cache_root")
    paths = {
        "unet": args.unet_checkpoint or artifacts.get("unet_checkpoint"),
        "correction": args.correction_checkpoint
        or artifacts.get("correction_checkpoint"),
        "joint": args.joint_checkpoint or artifacts.get("joint_checkpoint"),
        "cache_metadata": args.intensity_cache_metadata
        or (Path(cache_root) / "cache-metadata.json" if cache_root else None),
    }
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        raise ValueError(
            "provide --comparison-run or explicit paths for: " + ", ".join(missing)
        )
    if args.ablation_max_wind_checkpoint is not None:
        paths["ablation_max_wind_only"] = args.ablation_max_wind_checkpoint
    if args.ablation_radii_checkpoint is not None:
        paths["ablation_max_wind_radii"] = args.ablation_radii_checkpoint
    return {
        name: Path(path).expanduser().resolve()
        for name, path in paths.items()
        if path is not None
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            frame.to_csv(stream, index=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _cohort_fingerprint(frame: pd.DataFrame) -> str:
    columns = [
        "observation_id",
        "storm_id",
        "observation_timestamp",
        "target_ms",
        "ri_24h_change_ms",
        "is_rapid_intensification",
    ]
    data = (
        frame.loc[:, columns]
        .sort_values("observation_id")
        .to_csv(index=False, lineterminator="\n", float_format="%.9g")
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "mae_ms": float(np.mean(np.abs(error))),
        "rmse_ms": float(np.sqrt(np.mean(np.square(error)))),
        "bias_ms": float(np.mean(error)),
    }


def _summarize(frame: pd.DataFrame) -> dict[str, Any]:
    model_columns = {
        **MODEL_COLUMNS,
        **{
            name: column
            for name, column in ABLATION_COLUMNS.items()
            if column in frame.columns
        },
    }
    result: dict[str, Any] = {
        "samples": len(frame),
        "evaluated_samples": int(
            frame[list(model_columns.values())].notna().all(axis=1).sum()
        ),
        "storms": int(frame["storm_id"].nunique()),
        "models": {},
    }
    for model, column in model_columns.items():
        valid_frame = frame.loc[
            np.isfinite(frame[column].to_numpy(float))
            & np.isfinite(frame["target_ms"].to_numpy(float))
        ]
        prediction = valid_frame[column].to_numpy(float)
        target = valid_frame["target_ms"].to_numpy(float)
        per_storm = {}
        for storm_id, storm in valid_frame.groupby("storm_id", sort=True):
            per_storm[str(storm_id)] = {
                "samples": len(storm),
                **_metrics(
                    storm[column].to_numpy(float), storm["target_ms"].to_numpy(float)
                ),
            }
        result["models"][model] = {
            "samples": len(valid_frame),
            **_metrics(prediction, target),
            "storm_macro_mae_ms": float(
                np.mean([values["mae_ms"] for values in per_storm.values()])
            ),
            "per_storm": per_storm,
        }
    return result


def _valid_maximum(field: torch.Tensor, mask: torch.Tensor) -> float:
    field = field.squeeze()
    valid = mask.squeeze().bool() & torch.isfinite(field)
    if not valid.any():
        raise ValueError("predicted field contains no valid finite pixel")
    return float(field[valid].max().item())


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    storms = tuple(dict.fromkeys(str(value).strip().upper() for value in args.storms))
    paths = _regime_paths(args)
    required = [args.manifest, args.stats, args.ibtracs_file, *paths.values()]
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    correction_cache = _validate_correction_provenance(
        paths["unet"], paths["cache_metadata"]
    )

    use_era5 = args.era5 == "with"
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    records = _read_manifest(args.manifest, args.data_root)
    geo_records = {
        record.observation_id: record
        for record in records
        if record.source_type == "geo" and record.storm_id in storms
    }
    missing = sorted(set(storms) - {record.storm_id for record in geo_records.values()})
    if missing:
        raise ValueError(f"inference manifest has no GEO observations for {missing}")

    era5_by_storm: dict[str, xr.Dataset] = {}
    if use_era5:
        for storm in storms:
            matches = [
                record
                for record in records
                if record.storm_id == storm and record.source_type == "era5"
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one ERA5 record for {storm}, got {len(matches)}"
                )
            with xr.open_dataset(
                matches[0].path,
                group="rectilinear",
                engine="h5netcdf",
                decode_times=True,
            ) as source:
                era5_by_storm[storm] = source[list(ERA5_CHANNELS)].load()

    tracks = _load_ibtracs_tracks(args.ibtracs_file, set(storms))
    intensity_context = _storm_intensity_context(args.ibtracs_file, list(storms))
    unet = ERA5ResidualRegressor.load_from_checkpoint(
        paths["unet"], map_location="cpu"
    ).eval()
    if bool(unet.use_era5) != use_era5:
        raise ValueError(
            f"U-Net use_era5={unet.use_era5} does not match --era5 {args.era5}"
        )
    correction = UNetIntensityCorrection.load_from_checkpoint(
        paths["correction"], map_location="cpu"
    ).eval()
    joint = BottleneckUNetMLPRegressor.load_from_checkpoint(
        paths["joint"], map_location="cpu"
    ).eval()
    ablation_models = {
        name: BottleneckUNetMLPRegressor.load_from_checkpoint(
            paths[name], map_location="cpu"
        ).eval()
        for name in ABLATION_COLUMNS
        if name in paths
    }
    expected_channels = 23 if use_era5 else 14
    if (
        unet.condition_channels != expected_channels
        or joint.condition_channels != expected_channels
        or any(
            model.condition_channels != expected_channels
            for model in ablation_models.values()
        )
    ):
        raise ValueError(
            "checkpoint condition-channel contract does not match inference regime: "
            f"U-Net={unet.condition_channels}, joint={joint.condition_channels}, "
            f"expected={expected_channels}"
        )
    unet.to(args.device)
    correction.to(args.device)
    joint.to(args.device)
    for model in ablation_models.values():
        model.to(args.device)

    rows: list[dict[str, Any]] = []
    selected_count = 0
    with torch.inference_mode():
        for storm in storms:
            selected = sorted(
                (record for record in geo_records.values() if record.storm_id == storm),
                key=lambda record: (record.timestamp, record.observation_id),
            )
            if args.limit is not None:
                selected = selected[: args.limit]
            if args.num_shards is not None:
                selected = selected[args.shard_index :: args.num_shards]
            selected_count += len(selected)
            iterator = tqdm(
                selected, desc=f"{STORM_NAMES.get(storm, storm)} {args.era5}"
            )
            for geo in iterator:
                label = _interpolate_ibtracs_wind(
                    tracks[storm],
                    geo.timestamp,
                    max_bracket_hours=args.max_ibtracs_bracket_hours,
                )
                if label is None:
                    raise ValueError(
                        f"IBTrACS cannot bracket {storm} at {geo.timestamp}"
                    )
                ri_change_ms, is_ri = _ri_diagnostics(
                    tracks[storm],
                    label["observation_timestamp"],
                    current_wind_ms=float(label["target_wind_ms"]),
                    max_bracket_hours=args.max_ibtracs_bracket_hours,
                    threshold_kt=args.ri_threshold_kt,
                    window_hours=args.ri_window_hours,
                )
                batch, _ = _prepare_sample(
                    geo,
                    era5_by_storm.get(storm),
                    stats,
                    use_era5=use_era5,
                )
                device_batch = {
                    key: value.to(args.device) for key, value in batch.items()
                }
                base_row = {
                    "observation_id": geo.observation_id,
                    "storm_id": storm,
                    "storm_name": STORM_NAMES.get(storm, storm),
                    "source_split": geo.split,
                    "observation_timestamp": pd.Timestamp(geo.timestamp).isoformat(),
                    "sensor": geo.sensor,
                    "input_path": str(geo.path.resolve()),
                    "center_lat": float(geo.ibtracs_center[0]),
                    "center_lon": float(geo.ibtracs_center[1]),
                    "target_ms": float(label["target_wind_ms"]),
                    "target_category": tropical_category_from_wind_ms(
                        float(label["target_wind_ms"])
                    ),
                    "ri_24h_change_ms": ri_change_ms,
                    "is_rapid_intensification": is_ri,
                    "ibtracs_lower_fix_timestamp": label["lower_fix_timestamp"],
                    "ibtracs_upper_fix_timestamp": label["upper_fix_timestamp"],
                }
                if not bool(device_batch["condition_mask"].any().item()):
                    invalid_predictions: dict[str, Any] = {}
                    for name, column in ABLATION_COLUMNS.items():
                        if name in ablation_models:
                            invalid_predictions[column] = np.nan
                            invalid_predictions[f"{name}_category"] = None
                    rows.append(
                        {
                            **base_row,
                            "inference_valid": False,
                            "inference_issue": "no_valid_pixel_after_center_crop",
                            "unet_raw_max_ms": np.nan,
                            "unet_raw_category": None,
                            "unet_correction_ms": np.nan,
                            "unet_correction_delta_ms": np.nan,
                            "unet_correction_category": None,
                            "joint_unet_mlp_ms": np.nan,
                            "joint_unet_mlp_category": None,
                            **invalid_predictions,
                        }
                    )
                    continue
                raw_field = unet.predict_physical(device_batch)
                corrected = _corrected_intensity(
                    correction,
                    raw_field,
                    device_batch["condition_mask"],
                    geo,
                    intensity_context[storm],
                )
                joint_output = joint.predict_normalized(device_batch)
                ablation_predictions = {
                    name: float(
                        model.predict_normalized(
                            device_batch
                        ).ibtracs_max_wind_ms.item()
                    )
                    for name, model in ablation_models.items()
                }
                raw_ms = _valid_maximum(raw_field, device_batch["condition_mask"])
                corrected_ms = float(corrected.output_msw_ms.item())
                joint_ms = float(joint_output.ibtracs_max_wind_ms.item())
                extra_predictions: dict[str, Any] = {}
                for name, prediction_ms in ablation_predictions.items():
                    extra_predictions[ABLATION_COLUMNS[name]] = prediction_ms
                    extra_predictions[f"{name}_category"] = (
                        tropical_category_from_wind_ms(prediction_ms)
                    )
                rows.append(
                    {
                        **base_row,
                        "inference_valid": True,
                        "inference_issue": None,
                        "unet_raw_max_ms": raw_ms,
                        "unet_raw_category": tropical_category_from_wind_ms(raw_ms),
                        "unet_correction_ms": corrected_ms,
                        "unet_correction_delta_ms": float(
                            corrected.correction_ms.item()
                        ),
                        "unet_correction_category": tropical_category_from_wind_ms(
                            corrected_ms
                        ),
                        "joint_unet_mlp_ms": joint_ms,
                        "joint_unet_mlp_category": tropical_category_from_wind_ms(
                            joint_ms
                        ),
                        **extra_predictions,
                    }
                )

    frame = pd.DataFrame(rows).sort_values(
        ["storm_id", "observation_timestamp", "observation_id"]
    )
    if len(frame) != selected_count:
        raise RuntimeError(f"expected {selected_count} rows, produced {len(frame)}")
    label = "with-era5" if use_era5 else "without-era5"
    csv_path = args.output_root.expanduser().resolve() / f"{label}.csv"
    json_path = csv_path.with_suffix(".json")
    _atomic_csv(frame, csv_path)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "conditioning": label,
        "interpretation": "dense_validation_storm_case_study",
        "cohort": {
            "samples": len(frame),
            "storms": int(frame["storm_id"].nunique()),
            "storm_ids": list(storms),
            "source_splits": sorted(frame["source_split"].unique().tolist()),
            "sha256": _cohort_fingerprint(frame),
            "evaluated_samples": int(frame["inference_valid"].sum()),
            "invalid_samples": int((~frame["inference_valid"]).sum()),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        },
        "target": {
            "source": "IBTrACS USA_WIND",
            "units": "m s-1",
            "interpolation": "linear at GEO observation time",
            "maximum_bracket_hours": args.max_ibtracs_bracket_hours,
            "rapid_intensification": {
                "threshold_kt": args.ri_threshold_kt,
                "window_hours": args.ri_window_hours,
                "comparison": "greater_than_or_equal",
            },
        },
        "model_columns": {
            **MODEL_COLUMNS,
            **{
                name: column
                for name, column in ABLATION_COLUMNS.items()
                if name in ablation_models
            },
        },
        "checkpoints": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
            if name != "cache_metadata"
        },
        "correction_cache_scientific_evaluation": correction_cache.get(
            "scientific_evaluation", "unspecified"
        ),
        "metrics": _summarize(frame),
    }
    _atomic_json(payload, json_path)
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
    print(f"Wrote {csv_path} and {json_path}")
    return csv_path, json_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
