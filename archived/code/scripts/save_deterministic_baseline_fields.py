#!/usr/bin/env python3
"""Materialize Stage-1 baseline fields for residual post-processing ablations.

The residual diffusion checkpoint does not need to be sampled for this pass.
The deterministic baseline is evaluated with the exact checkpoint referenced by
the diffusion config, then one compact NPZ is written per observation.  Keeping
this as a separate artifact makes the post-processing sweep reproducible and
avoids duplicating the expensive stochastic ensemble fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_geo_sar_geotiffs import (  # noqa: E402
    ERA5_CHANNELS,
    _era5_coordinate_values,
    _fix_longitudes,
    _load_geo_channels,
    _make_grid,
    _nearest_time_index,
    _read_manifest,
    _regrid_continuous,
    _regrid,
    _append_native_era5_derived_fields,
)
from scripts.run_storm_diffusion_inference import (  # noqa: E402
    CROP_SIZE,
    GRID_RESOLUTION_DEGREES,
    GRID_SIZE,
    ROBUST_CLIP,
    _bounds,
    _center_crop,
    _normalize,
)
from geo2wf.training import build_model, resolve_runtime_config  # noqa: E402
from scripts.pmw_conditioning import (
    nearest_supported_pmw,
    pmw_audit_row,
    pmw_condition_settings,
    prepare_pmw_condition_features,
    supported_pmw_by_storm,
)
from geo2wf.data.features import (
    normalized_distance_to_center as _normalized_distance_to_center,
    solar_time_features as _solar_time_features,
)  # noqa: E402

DEFAULT_DATA_ROOT = ROOT / "inference" / "inf_data"
DEFAULT_REFERENCE_ROOT = ROOT / "inference" / "inf_vit"
DEFAULT_OUTPUT_ROOT = ROOT / "logs" / "ablation-suites" / "postprocess-baseline"
DEFAULT_STATS = ROOT / "data" / "geotiff" / "geo_sar_10bands_era5" / "stats.json"
DEFAULT_CONFIG = ROOT / "configs" / "config_geo_sar_10bands_era5_residual.yaml"
DEFAULT_CHECKPOINT = (
    ROOT
    / "logs"
    / "20260730-132206_config_geo_sar_10bands_era5_residual"
    / "checkpoints"
    / "epoch=038-step=4758.ckpt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_DATA_ROOT / "index-files" / "observation_manifest_v6.csv",
    )
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--storms",
        nargs="+",
        default=None,
        help="Optional subset of manifest storm IDs (default: all manifest storms).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _safe_component(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
    return safe[:180] or "observation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _era5_fields(dataset: xr.Dataset, timestamp: pd.Timestamp):
    times = pd.DatetimeIndex(dataset["time"].values)
    times = times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
    index = _nearest_time_index(times, timestamp)
    if index is None:
        raise ValueError("ERA5 dataset has no time step for " + str(timestamp))
    lat = _era5_coordinate_values(dataset, "latitude", "lat", index)
    lon = _fix_longitudes(_era5_coordinate_values(dataset, "longitude", "lon", index))
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    return {
        channel: (
            np.asarray(dataset[channel].isel(time=index).values, dtype=np.float32),
            lat,
            lon,
        )
        for channel in ERA5_CHANNELS
    }


def _prepare_sample(geo, era5_dataset: xr.Dataset, stats: dict):
    if geo.center is None or geo.ibtracs_center is None:
        raise ValueError(f"{geo.observation_id} has no finite center")
    grid_lat, grid_lon = _make_grid(
        geo.center[0], geo.center[1], GRID_SIZE, GRID_RESOLUTION_DEGREES
    )
    geo_channels = list(
        {
            "abi": [
                "CMI_C07",
                "CMI_C08",
                "CMI_C09",
                "CMI_C10",
                "CMI_C11",
                "CMI_C12",
                "CMI_C13",
                "CMI_C14",
                "CMI_C15",
                "CMI_C16",
            ],
            "ahi": [
                "B07",
                "B08",
                "B09",
                "B10",
                "B11",
                "B12",
                "B13",
                "B14",
                "B15",
                "B16",
            ],
        }[geo.sensor.lower()]
    )
    geo_fields = _load_geo_channels(geo, geo_channels)
    geo_regridded = [
        _regrid(*geo_fields[channel], grid_lat, grid_lon) for channel in geo_channels
    ]
    geo_array = np.stack([item[0] for item in geo_regridded]).astype(np.float32)
    geo_mask = np.logical_and.reduce(
        [item[1] & np.isfinite(item[0]) for item in geo_regridded]
    )
    era5_fields = _append_native_era5_derived_fields(
        _era5_fields(era5_dataset, geo.timestamp)
    )
    era5_channels = list(era5_fields)
    era5_regridded = [
        _regrid_continuous(*era5_fields[channel], grid_lat, grid_lon)
        for channel in era5_channels
    ]
    era5_array = np.stack([item[0] for item in era5_regridded]).astype(np.float32)
    era5_mask = np.logical_and.reduce(
        [item[1] & np.isfinite(item[0]) for item in era5_regridded]
    )
    valid = torch.from_numpy(geo_mask & era5_mask).unsqueeze(0)
    geo_tensor = torch.from_numpy(geo_array)
    era5_tensor = torch.from_numpy(era5_array)
    wind_index = era5_channels.index("wind_speed_10m")
    era5_wind_physical = era5_tensor[wind_index : wind_index + 1].clone()
    geo_tensor = _normalize(
        geo_tensor,
        "geo",
        geo_channels,
        stats,
        normalization="robust-zscore",
        robust_clip=ROBUST_CLIP,
    )
    era5_tensor = _normalize(
        era5_tensor,
        "era5",
        [f"era5_{channel}" for channel in era5_channels],
        stats,
        normalization="robust-zscore",
        robust_clip=ROBUST_CLIP,
    )
    era5_wind = _normalize(
        era5_wind_physical, "sar", ["wind_speed"], stats, normalization="min-max"
    )
    condition = torch.cat([geo_tensor, era5_tensor], dim=0)
    condition = _center_crop(torch.nan_to_num(condition) * valid)
    era5_wind = _center_crop(torch.nan_to_num(era5_wind) * valid)
    era5_wind_physical = _center_crop(torch.nan_to_num(era5_wind_physical) * valid)
    valid = _center_crop(valid)
    bounds = _bounds(*geo.center)
    distance = _normalized_distance_to_center(
        bounds, (CROP_SIZE, CROP_SIZE), torch.tensor(geo.ibtracs_center)
    )
    solar = _solar_time_features(bounds, (CROP_SIZE, CROP_SIZE), geo.timestamp)
    condition = torch.cat([condition, distance, solar], dim=0)
    return {
        "condition": condition.unsqueeze(0),
        "condition_mask": valid.unsqueeze(0),
        "era5_wind_speed": era5_wind.unsqueeze(0),
        "era5_wind_speed_physical": era5_wind_physical.unsqueeze(0),
        "era5_wind_speed_mask": valid.unsqueeze(0),
    }, valid.squeeze(0)


def _manifest_storm_ids(manifest: Path) -> list[str]:
    """Return every unique non-empty storm ID declared by the manifest."""
    return sorted(
        pd.read_csv(manifest, usecols=["storm_id"])["storm_id"]
        .dropna()
        .astype(str)
        .unique()
    )


def main() -> None:
    args = parse_args()
    manifest_storms = _manifest_storm_ids(args.manifest)
    if args.storms is None:
        args.storms = manifest_storms
    else:
        unknown = sorted(set(args.storms) - set(manifest_storms))
        if unknown:
            raise ValueError(f"Storms are not present in the manifest: {unknown}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    for path in (args.config, args.checkpoint, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = resolve_runtime_config(
        yaml.safe_load(args.config.read_text(encoding="utf-8"))
    )
    stats_path = args.stats or Path(config["data"]["stats_file"])
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    records = _read_manifest(args.manifest, args.data_root)
    by_id = {record.observation_id: record for record in records}
    era5_by_storm = {}
    for storm in args.storms:
        record = next(
            item
            for item in records
            if item.storm_id == storm and item.source_type == "era5"
        )
        with xr.open_dataset(
            record.path, group="rectilinear", engine="h5netcdf", decode_times=True
        ) as source:
            era5_by_storm[storm] = source[list(ERA5_CHANNELS)].load()
    pmw_enabled, pmw_max_gap_hours, pmw_include_offset = pmw_condition_settings(config)
    pmw_records = supported_pmw_by_storm(records)
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(args.device)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    counts = {}
    with torch.inference_mode():
        for storm in args.storms:
            reference = pd.read_csv(
                args.reference_root / storm / "inference-summary.csv"
            )
            available = reference["observation_id"].isin(by_id)
            if not available.all():
                print(
                    f"Skipping {(~available).sum()} stale ViT observations for {storm} "
                    "that are absent from the current manifest"
                )
                reference = reference.loc[available].copy()
            if args.limit is not None:
                reference = reference.head(args.limit).copy()
            storm_dir = output_root / storm / "baseline-fields"
            storm_dir.mkdir(parents=True, exist_ok=True)
            storm_rows = list(reference.itertuples(index=False))
            audit_rows = []
            matched_count = 0
            for start in tqdm(
                range(0, len(storm_rows), args.batch_size),
                desc=f"baseline {storm}",
                total=math.ceil(len(storm_rows) / args.batch_size),
            ):
                chunk = storm_rows[start : start + args.batch_size]
                prepared = []
                matched_chunk = []
                for row in chunk:
                    geo = by_id[row.observation_id]
                    selected_pmw = None
                    selected_gap = None
                    status = "disabled"
                    if pmw_enabled:
                        selected_pmw, selected_gap, status = nearest_supported_pmw(
                            geo, pmw_records, max_time_gap_hours=pmw_max_gap_hours
                        )
                        if status != "matched":
                            audit_rows.append(
                                pmw_audit_row(
                                    geo,
                                    selected_pmw,
                                    selected_gap,
                                    "skipped",
                                    reason=status,
                                )
                            )
                            continue
                    sample = _prepare_sample(geo, era5_by_storm[storm], stats)
                    if pmw_enabled:
                        grid_lat, grid_lon = _make_grid(
                            geo.center[0],
                            geo.center[1],
                            GRID_SIZE,
                            GRID_RESOLUTION_DEGREES,
                        )
                        try:
                            pmw_features, _, selected_gap = (
                                prepare_pmw_condition_features(
                                    geo,
                                    selected_pmw,
                                    grid_lat,
                                    grid_lon,
                                    stats,
                                    max_time_gap_hours=pmw_max_gap_hours,
                                    include_time_offset=pmw_include_offset,
                                    crop_size=CROP_SIZE,
                                )
                            )
                        except (KeyError, OSError, ValueError) as error:
                            audit_rows.append(
                                pmw_audit_row(
                                    geo,
                                    selected_pmw,
                                    selected_gap,
                                    "skipped",
                                    reason=str(error),
                                )
                            )
                            continue
                        sample[0]["condition"] = torch.cat(
                            [sample[0]["condition"], pmw_features.unsqueeze(0)], dim=1
                        )
                        audit_rows.append(
                            pmw_audit_row(geo, selected_pmw, selected_gap, "matched")
                        )
                    prepared.append(sample)
                    matched_chunk.append(row)
                chunk = matched_chunk
                if not prepared:
                    continue
                matched_count += len(chunk)
                batch = {
                    key: torch.cat([item[0][key] for item in prepared], dim=0).to(
                        args.device
                    )
                    for key in prepared[0][0]
                }
                baseline = model.predict_physical(batch).detach().float().cpu()
                valid = torch.stack([item[1] for item in prepared])
                for index, row in enumerate(chunk):
                    observation_id = str(row.observation_id)
                    filename = f"{_safe_component(observation_id)}.npz"
                    path = storm_dir / filename
                    np.savez_compressed(
                        path,
                        observation_id=np.asarray(observation_id),
                        baseline_field_ms=baseline[index, 0]
                        .numpy()
                        .astype(np.float32, copy=False),
                        valid_mask=valid[index].numpy().astype(np.uint8, copy=False),
                    )
                    rows.append(
                        {
                            "storm_id": storm,
                            "observation_id": observation_id,
                            "observation_timestamp": getattr(
                                row, "observation_timestamp", None
                            ),
                            "npz_path": str(path.relative_to(output_root)),
                            "baseline_array": "baseline_field_ms",
                            "shape": "x".join(
                                str(value) for value in baseline[index, 0].shape
                            ),
                            "dtype": "float32",
                        }
                    )
            if pmw_enabled:
                pd.DataFrame(audit_rows).to_csv(
                    output_root / storm / "pmw-inference-audit.csv", index=False
                )
            counts[storm] = matched_count
    pd.DataFrame(rows).to_csv(output_root / "baseline-fields-manifest.csv", index=False)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "storms": list(args.storms),
        "records_by_storm": counts,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": _sha256(args.checkpoint),
        },
        "config": str(args.config.resolve()),
        "manifest": str(args.manifest.resolve()),
        "device": str(args.device),
        "include_test_in_train": config.get("data", {}).get("include_test_in_train"),
        "outputs": {
            "manifest": "baseline-fields-manifest.csv",
            "fields": "<storm>/baseline-fields/<observation>.npz",
        },
    }
    (output_root / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} baseline fields to {output_root}")


if __name__ == "__main__":
    main()
