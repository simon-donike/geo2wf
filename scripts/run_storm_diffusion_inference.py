"""Run the deterministic-baseline residual diffusion model for the explorer storms.

The output deliberately mirrors the CSV contract in ``inference/inf_anna``.
Per-observation tensor bundles are not written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import xarray as xr
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (  # noqa: E402
    _normalize,
    _normalized_distance_to_center,
    _solar_time_features,
)
from scripts.export_geo_sar_geotiffs import (  # noqa: E402
    ERA5_CHANNELS,
    GEO_CHANNEL_SETS,
    _append_native_era5_derived_fields,
    _era5_coordinate_values,
    _fix_longitudes,
    _load_geo_channels,
    _make_grid,
    _nearest_time_index,
    _read_manifest,
    _regrid,
    _regrid_continuous,
)
from src.ERA5Residual import ERA5ResidualRegressor  # noqa: E402
from train import build_model

DEFAULT_DATA_ROOT = ROOT / "inference" / "inf_data"
DEFAULT_REFERENCE_ROOT = ROOT / "inference" / "inf_anna"
DEFAULT_OUTPUT_ROOT = ROOT / "inference" / "inf_model_c"
DEFAULT_STATS = ROOT / "data" / "geotiff" / "geo_sar_10bands_era5" / "stats.json"
DEFAULT_CONFIG = ROOT / "configs" / "config_geo_sar_10bands_era5_diffusion_residual_deterministic.yaml"
DEFAULT_CHECKPOINT = (
    ROOT
    / "logs"
    / "20260730-150623_config_geo_sar_10bands_era5_diffusion_residual_deterministic"
    / "checkpoints"
    / "epoch=055-step=6832.ckpt"
)
GRID_SIZE = 256
GRID_RESOLUTION_DEGREES = 0.027
CROP_SIZE = 192
ROBUST_CLIP = 4.0
R64_THRESHOLD_MS = 64.0 * 0.514444


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
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--storms", nargs="+", default=["AL082025", "EP112025"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Debug: rows per storm.")
    return parser.parse_args()


def _bounds(center_lat: float, center_lon: float) -> torch.Tensor:
    half_extent = GRID_SIZE * GRID_RESOLUTION_DEGREES / 2.0
    full = torch.tensor(
        [
            center_lon - half_extent,
            center_lon + half_extent,
            center_lat - half_extent,
            center_lat + half_extent,
        ],
        dtype=torch.float64,
    )
    crop_offset = (GRID_SIZE - CROP_SIZE) // 2 * GRID_RESOLUTION_DEGREES
    return full + torch.tensor(
        [crop_offset, -crop_offset, crop_offset, -crop_offset],
        dtype=torch.float64,
    )


def _center_crop(tensor: torch.Tensor) -> torch.Tensor:
    offset = (tensor.shape[-1] - CROP_SIZE) // 2
    return tensor[..., offset : offset + CROP_SIZE, offset : offset + CROP_SIZE]


def _physical_distance_km(
    bounds: torch.Tensor, center: tuple[float, float]
) -> torch.Tensor:
    normalized = _normalized_distance_to_center(
        bounds, (CROP_SIZE, CROP_SIZE), torch.tensor(center)
    ).squeeze(0)
    left, right, bottom, top = bounds.tolist()
    corners = [
        (bottom, left),
        (bottom, right),
        (top, left),
        (top, right),
    ]

    def haversine_km(point: tuple[float, float]) -> float:
        lat1, lon1 = map(math.radians, center)
        lat2, lon2 = map(math.radians, point)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        value = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        return 6371.0 * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))

    return normalized * max(haversine_km(corner) for corner in corners)


def _output_metrics(
    output: torch.Tensor, valid: torch.Tensor, distance_km: torch.Tensor
) -> dict[str, float]:
    """Calculate physical wind metrics over valid pixels only."""
    field = output.squeeze().cpu()
    valid = valid.squeeze().cpu().bool() & torch.isfinite(field)
    if int(valid.sum()) < math.ceil(0.05 * field.numel()):
        return {key: math.nan for key in ("msw", "r64", "p90", "core_mean", "rmw", "mean")}
    values = field[valid]
    core = field[valid & (distance_km <= 100.0)]
    bin_width = float(
        torch.nanmedian(torch.abs(torch.diff(distance_km[:, CROP_SIZE // 2])))
    )
    bin_width = bin_width if math.isfinite(bin_width) and bin_width > 0 else 3.0
    radii = torch.arange(
        0.0, float(distance_km[valid].max()) + bin_width, bin_width
    )
    radial_distance, radial_wind = [], []
    for radius in radii:
        annulus = valid & (distance_km >= radius) & (distance_km < radius + bin_width)
        if annulus.any():
            radial_distance.append(float(radius))
            radial_wind.append(float(field[annulus].mean()))
    r64 = [radius for radius, wind in zip(radial_distance, radial_wind) if wind >= R64_THRESHOLD_MS]
    rmw = radial_distance[radial_wind.index(max(radial_wind))] if radial_wind else math.nan
    return {
        "msw": float(values.max()),
        "r64": max(r64) if r64 else math.nan,
        "p90": float(torch.quantile(values, 0.9, interpolation="linear")),
        "core_mean": float(core.mean()) if core.numel() else math.nan,
        "rmw": rmw,
        "mean": float(values.mean()),
    }


def _era5_fields(
    dataset: xr.Dataset, timestamp: pd.Timestamp
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    times = pd.DatetimeIndex(dataset["time"].values)
    times = times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
    index = _nearest_time_index(times, timestamp)
    if index is None:
        raise ValueError("ERA5 dataset has no time steps")
    selected_time = times[index]
    lat = _era5_coordinate_values(dataset, "latitude", "lat", index)
    lon = _fix_longitudes(
        _era5_coordinate_values(dataset, "longitude", "lon", index)
    )
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


def _prepare_sample(
    geo, era5_dataset: xr.Dataset, stats: dict
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if geo.center is None or geo.ibtracs_center is None:
        raise ValueError(f"{geo.observation_id} has no finite center")
    grid_lat, grid_lon = _make_grid(
        geo.center[0],
        geo.center[1],
        GRID_SIZE,
        GRID_RESOLUTION_DEGREES,
    )
    geo_channels = list(GEO_CHANNEL_SETS["common10"][geo.sensor])
    geo_fields = _load_geo_channels(geo, geo_channels)
    geo_regridded = [
        _regrid(*geo_fields[channel], grid_lat, grid_lon) for channel in geo_channels
    ]
    geo_array = np.stack([item[0] for item in geo_regridded]).astype(np.float32)
    geo_mask = np.logical_and.reduce(
        [item[1] & np.isfinite(item[0]) for item in geo_regridded]
    )

    era5_fields = _era5_fields(era5_dataset, geo.timestamp)
    era5_fields = _append_native_era5_derived_fields(era5_fields)
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
        era5_wind_physical,
        "sar",
        ["wind_speed"],
        stats,
        normalization="min-max",
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
    batch = {
        "condition": condition.unsqueeze(0),
        "condition_mask": valid.unsqueeze(0),
        "era5_wind_speed": era5_wind.unsqueeze(0),
        "era5_wind_speed_physical": era5_wind_physical.unsqueeze(0),
        "era5_wind_speed_mask": valid.unsqueeze(0),
    }
    return batch, _physical_distance_km(bounds, geo.ibtracs_center)

def main() -> None:
    args = parse_args()
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    records = _read_manifest(args.manifest, args.data_root)
    by_id = {record.observation_id: record for record in records}
    era5_records = {
        storm: next(record for record in records if record.storm_id == storm and record.source_type == "era5")
        for storm in args.storms
    }
    era5_by_storm = {}
    for storm, record in era5_records.items():
        with xr.open_dataset(record.path, group="rectilinear", engine="h5netcdf", decode_times=True) as source:
            era5_by_storm[storm] = source[list(ERA5_CHANNELS)].load()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(args.device)
    target_stats = stats["channels"]["sar"]["wind_speed"]
    target_offset = float(target_stats["min"])
    target_scale = float(target_stats["max"]) - target_offset
    args.output_root.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for storm in args.storms:
            table = pd.read_csv(args.reference_root / storm / "inference-summary.csv")
            if args.limit is not None:
                table = table.head(args.limit).copy()
            rows = list(table.itertuples(index=False))
            metric_rows, local_paths = [], []
            starts = range(0, len(rows), args.batch_size)
            iterator = tqdm(starts, total=math.ceil(len(rows) / args.batch_size), desc=storm)
            for start in iterator:
                chunk = rows[start : start + args.batch_size]
                prepared = [
                    _prepare_sample(by_id[row.observation_id], era5_by_storm[storm], stats)
                    for row in chunk
                ]
                batch = {
                    key: torch.cat([sample[0][key] for sample in prepared], dim=0).to(args.device)
                    for key in prepared[0][0]
                }
                batch["sample_id"] = [row.observation_id for row in chunk]
                batch["target"] = torch.zeros_like(batch["era5_wind_speed"])
                batch["target_mask"] = batch["condition_mask"]
                batch["target_norm_offset"] = torch.full((len(chunk), 1), target_offset, device=args.device)
                batch["target_norm_scale"] = torch.full((len(chunk), 1), target_scale, device=args.device)
                _, normalized = model._predict_batch(batch, start // args.batch_size)
                prediction = normalized * target_scale + target_offset
                for index, (row, (_, distance_km)) in enumerate(zip(chunk, prepared)):
                    metric_rows.append(_output_metrics(prediction[index : index + 1], batch["condition_mask"][index : index + 1], distance_km))
                    local_paths.append(str(by_id[row.observation_id].path))
            table["original_input_path"] = local_paths
            for column, key in {
                "output_msw_ms": "msw", "output_r64_km": "r64",
                "output_p90_ms": "p90", "output_core_mean_ms": "core_mean",
                            "output_rmw_km": "rmw", "output_mean_ms": "mean",
            }.items():
                table[column] = [metrics[key] for metrics in metric_rows]
            storm_dir = args.output_root / storm
            storm_dir.mkdir(parents=True, exist_ok=True)
            output_path = storm_dir / "inference-summary.csv"
            temporary_path = output_path.with_suffix(".csv.tmp")
            table.to_csv(temporary_path, index=False)
            temporary_path.replace(output_path)
            print(f"Wrote {output_path.relative_to(ROOT)} ({len(table)} rows)")


if __name__ == "__main__":
    main()
