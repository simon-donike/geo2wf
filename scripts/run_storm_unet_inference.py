"""Run the deterministic ERA5-residual U-Net for the explorer storms.

All raw GEO, ERA5, PMW, and manifest inputs come from ``inference/inf_data``;
the ViT summaries only define the dashboard observation IDs. The output mirrors
their CSV contract under ``inference/inf_unet``; tensor bundles are not written.
When a correction checkpoint is supplied, the maximum-wind column and category
are written as a distinct UNet+MLP result series.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo2wf.data.features import (  # noqa: E402
    normalized_distance_to_center as _normalized_distance_to_center,
    solar_time_features as _solar_time_features,
)
from geo2wf.config import load_config_file  # noqa: E402
from geo2wf.data.normalization import (  # noqa: E402
    normalization_affine_parameters,
    normalize as _normalize,
)
from geo2wf.data.intensity import encode_intensity_metadata  # noqa: E402
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
from geo2wf.models.deterministic_residual import ERA5ResidualRegressor  # noqa: E402
from geo2wf.models.intensity_correction import UNetIntensityCorrection  # noqa: E402
from scripts.pmw_conditioning import (
    nearest_supported_pmw,
    pmw_audit_row,
    pmw_condition_settings,
    prepare_pmw_condition_features,
    supported_pmw_by_storm,
)

DEFAULT_DATA_ROOT = ROOT / "inference" / "inf_data"
DEFAULT_REFERENCE_ROOT = ROOT / "inference" / "inf_vit"
DEFAULT_OUTPUT_ROOT = ROOT / "inference" / "inf_unet"
DEFAULT_CONFIG = ROOT / "configs" / "modular.yaml"
DEFAULT_STATS = ROOT / "data" / "geotiff" / "geo_sar_10bands_era5" / "stats.json"
DEFAULT_IBTRACS_FILE = ROOT / "data" / "IBTrACs" / "ibtracs.ALL.list.v04r01.csv"
DEFAULT_INTENSITY_CACHE_METADATA = (
    ROOT / "data" / "unet_intensity_geostat_nopmw_v2" / "cache-metadata.json"
)
GRID_SIZE = 256
GRID_RESOLUTION_DEGREES = 0.027
CROP_SIZE = 192
ROBUST_CLIP = 4.0
R64_THRESHOLD_MS = 64.0 * 0.514444
WIND_RADIUS_THRESHOLDS_MS = {
    "r34": 34.0 * 0.514444,
    "r50": 50.0 * 0.514444,
    "r64": R64_THRESHOLD_MS,
}
EARTH_RADIUS_KM = 6371.0
RADIAL_BIN_KM = 10.0


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--correction-checkpoint",
        type=Path,
        default=None,
        help="Optional single-field intensity correction checkpoint.",
    )
    parser.add_argument(
        "--intensity-cache-metadata",
        type=Path,
        default=DEFAULT_INTENSITY_CACHE_METADATA,
        help=(
            "Cache provenance used to verify the frozen U-Net used for "
            "correction training."
        ),
    )
    parser.add_argument("--ibtracs-file", type=Path, default=DEFAULT_IBTRACS_FILE)
    parser.add_argument(
        "--storms",
        nargs="+",
        default=None,
        help="Optional subset of manifest storm IDs (default: all manifest storms).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Debug: rows per storm."
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_correction_provenance(
    unet_checkpoint: Path, cache_metadata_path: Path
) -> dict:
    metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    expected = str(metadata.get("unet_checkpoint", {}).get("sha256", ""))
    actual = _sha256(unet_checkpoint)
    if not expected or actual != expected:
        raise ValueError(
            "correction U-Net provenance mismatch: cache expects "
            f"{expected or 'no hash'}, selected checkpoint is {actual}"
        )
    return metadata


def _storm_intensity_context(
    ibtracs_file: Path, storm_ids: list[str]
) -> dict[str, dict[str, object]]:
    tracks = pd.read_csv(
        ibtracs_file,
        skiprows=[1],
        usecols=["USA_ATCF_ID", "ISO_TIME", "BASIN"],
        keep_default_na=False,
        low_memory=False,
    )
    tracks["storm_id"] = tracks["USA_ATCF_ID"].astype(str).str.strip().str.upper()
    tracks["timestamp"] = pd.to_datetime(tracks["ISO_TIME"], errors="coerce", utc=True)
    tracks["basin"] = tracks["BASIN"].astype(str).str.strip().str.upper()
    tracks = tracks.loc[
        tracks["storm_id"].isin(storm_ids) & tracks["timestamp"].notna()
    ].sort_values(["storm_id", "timestamp"])
    context = {
        storm_id: {
            "start": frame.iloc[0]["timestamp"],
            "basin": frame.loc[frame["basin"].ne(""), "basin"].iloc[0],
        }
        for storm_id, frame in tracks.groupby("storm_id")
    }
    missing = sorted(set(storm_ids) - set(context))
    if missing:
        raise ValueError(f"IBTrACS has no track context for storms: {missing}")
    return context


def _corrected_intensity(
    correction_model: UNetIntensityCorrection,
    wind_field: torch.Tensor,
    valid_mask: torch.Tensor,
    geo,
    context: dict[str, object],
):
    bounds = _bounds(*geo.center)
    normalized_distance = _normalized_distance_to_center(
        bounds, (CROP_SIZE, CROP_SIZE), torch.tensor(geo.ibtracs_center)
    ).to(device=wind_field.device, dtype=wind_field.dtype)
    finite_valid = (
        valid_mask.squeeze(1).bool()
        & torch.isfinite(wind_field.squeeze(1))
        & torch.isfinite(normalized_distance)
    )
    valid_fraction = float(finite_valid.float().mean())
    elapsed_hours = max(
        0.0,
        (pd.Timestamp(geo.timestamp) - pd.Timestamp(context["start"])).total_seconds()
        / 3600.0,
    )
    metadata = encode_intensity_metadata(
        {
            "observation_timestamp": geo.timestamp,
            "center_lat": geo.ibtracs_center[0],
            "center_lon": geo.ibtracs_center[1],
            "basin": context["basin"],
            "storm_elapsed_hours": elapsed_hours,
            "valid_fraction": valid_fraction,
        }
    ).unsqueeze(0)
    return correction_model.predict_intensity(
        {
            "wind_field": wind_field.squeeze(1),
            "valid_mask": finite_valid,
            "distance_to_center": normalized_distance,
            "metadata": metadata.to(wind_field.device),
        }
    )


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
    output: torch.Tensor,
    valid: torch.Tensor,
    distance_km: torch.Tensor,
    bounds: torch.Tensor,
    center: tuple[float, float],
) -> dict[str, float]:
    """Calculate physical metrics directly from a predicted wind image.

    R34/R50/R64 are equivalent-circle radii of threshold-exceedance area in
    the largest complete storm-centred circle supported by the image. RMW is
    the peak 10-km annular-mean bin. The centre locates the polar geometry; no
    IBTrACS radius value is used in these estimates.
    """
    field = output.squeeze().cpu()
    valid = valid.squeeze().cpu().bool() & torch.isfinite(field)
    if int(valid.sum()) < math.ceil(0.05 * field.numel()):
        return {
            key: math.nan
            for key in (
                "msw",
                "r34",
                "r50",
                "r64",
                "p90",
                "core_mean",
                "rmw",
                "mean",
            )
        }
    values = field[valid]
    core = field[valid & (distance_km <= 100.0)]
    left, right, bottom, top = bounds.cpu().to(torch.float64).tolist()
    center_lat, center_lon = center
    height, width = field.shape
    row_fraction = (torch.arange(height, dtype=torch.float64) + 0.5) / height
    column_fraction = (torch.arange(width, dtype=torch.float64) + 0.5) / width
    latitudes = top - row_fraction * (top - bottom)
    longitudes = left + column_fraction * (right - left)
    latitude_grid, longitude_grid = torch.meshgrid(latitudes, longitudes, indexing="ij")
    delta_lon = torch.remainder(longitude_grid - center_lon + 180.0, 360.0) - 180.0
    north_km = torch.deg2rad(latitude_grid - center_lat) * EARTH_RADIUS_KM
    east_km = (
        torch.deg2rad(delta_lon) * EARTH_RADIUS_KM * math.cos(math.radians(center_lat))
    )
    radius_km = torch.sqrt(north_km.square() + east_km.square())
    geometry_valid = valid & torch.isfinite(radius_km)
    directional_extents = torch.stack(
        (
            north_km[geometry_valid].max(),
            -north_km[geometry_valid].min(),
            east_km[geometry_valid].max(),
            -east_km[geometry_valid].min(),
        )
    )
    complete_radius = float(directional_extents.min())
    if complete_radius <= 0.0:
        return {
            "msw": float(values.max()),
            "r34": math.nan,
            "r50": math.nan,
            "r64": math.nan,
            "p90": float(torch.quantile(values, 0.9, interpolation="linear")),
            "core_mean": float(core.mean()) if core.numel() else math.nan,
            "rmw": math.nan,
            "mean": float(values.mean()),
        }
    complete_domain = geometry_valid & (radius_km <= complete_radius)

    radial_distance, radial_wind = [], []
    for lower in torch.arange(0.0, complete_radius, RADIAL_BIN_KM):
        upper = float(lower) + RADIAL_BIN_KM
        if upper > complete_radius:
            continue
        annulus = complete_domain & (radius_km >= lower) & (radius_km < upper)
        if int(annulus.sum()) >= 4:
            radial_distance.append(float(lower) + RADIAL_BIN_KM / 2.0)
            radial_wind.append(float(field[annulus].mean()))
    rmw = radial_distance[int(np.argmax(radial_wind))] if radial_wind else math.nan

    latitude_edges = torch.linspace(top, bottom, height + 1, dtype=torch.float64)
    longitude_width_rad = math.radians(abs(right - left) / width)
    row_area_km2 = (
        EARTH_RADIUS_KM**2
        * longitude_width_rad
        * (
            torch.sin(torch.deg2rad(latitude_edges[:-1]))
            - torch.sin(torch.deg2rad(latitude_edges[1:]))
        ).abs()
    )
    pixel_area_km2 = row_area_km2[:, None].expand(height, width)
    wind_radii = {}
    for name, threshold_ms in WIND_RADIUS_THRESHOLDS_MS.items():
        area_km2 = pixel_area_km2[complete_domain & (field >= threshold_ms)].sum()
        wind_radii[name] = math.sqrt(float(area_km2) / math.pi)
    return {
        "msw": float(values.max()),
        **wind_radii,
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


def _prepare_sample(
    geo,
    era5_dataset: xr.Dataset | None,
    stats: dict,
    *,
    use_era5: bool = True,
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

    geo_tensor = torch.from_numpy(geo_array)
    geo_tensor = _normalize(
        geo_tensor,
        "geo",
        geo_channels,
        stats,
        normalization="robust-zscore",
        robust_clip=ROBUST_CLIP,
    )
    valid = torch.from_numpy(geo_mask).unsqueeze(0)
    condition = geo_tensor
    era5_wind = None
    era5_wind_physical = None
    if use_era5:
        if era5_dataset is None:
            raise ValueError("ERA5 data are required when use_era5=True")
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
        era5_tensor = torch.from_numpy(era5_array)
        wind_index = era5_channels.index("wind_speed_10m")
        era5_wind_physical = era5_tensor[wind_index : wind_index + 1].clone()
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
    }
    target_offset, target_scale = normalization_affine_parameters(
        "sar", ["wind_speed"], stats, normalization="min-max"
    )
    batch["target_norm_offset"] = target_offset.unsqueeze(0)
    batch["target_norm_scale"] = target_scale.unsqueeze(0)
    if use_era5:
        assert era5_wind is not None and era5_wind_physical is not None
        era5_wind = _center_crop(torch.nan_to_num(era5_wind)) * valid
        era5_wind_physical = _center_crop(torch.nan_to_num(era5_wind_physical)) * valid
        batch.update(
            {
                "era5_wind_speed": era5_wind.unsqueeze(0),
                "era5_wind_speed_physical": era5_wind_physical.unsqueeze(0),
                "era5_wind_speed_mask": valid.unsqueeze(0),
            }
        )
    return batch, _physical_distance_km(bounds, geo.ibtracs_center)


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
    correction_model = None
    correction_provenance = None
    intensity_context = {}
    if args.correction_checkpoint is not None:
        for required in (
            args.checkpoint,
            args.correction_checkpoint,
            args.intensity_cache_metadata,
            args.ibtracs_file,
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        correction_provenance = _validate_correction_provenance(
            args.checkpoint, args.intensity_cache_metadata
        )
        intensity_context = _storm_intensity_context(args.ibtracs_file, args.storms)
    config = load_config_file(args.config)
    stats_path = args.stats or Path(config["data"]["stats_file"])
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    records = _read_manifest(args.manifest, args.data_root)
    pmw_enabled, pmw_max_gap_hours, pmw_include_offset = pmw_condition_settings(config)
    pmw_records = supported_pmw_by_storm(records)
    by_id = {record.observation_id: record for record in records}
    era5_records = {
        storm: next(
            record
            for record in records
            if record.storm_id == storm and record.source_type == "era5"
        )
        for storm in args.storms
    }
    era5_by_storm = {}
    for storm, record in era5_records.items():
        with xr.open_dataset(
            record.path, group="rectilinear", engine="h5netcdf", decode_times=True
        ) as source:
            era5_by_storm[storm] = source[list(ERA5_CHANNELS)].load()
    model = (
        ERA5ResidualRegressor.load_from_checkpoint(args.checkpoint, map_location="cpu")
        .eval()
        .to(args.device)
    )
    if args.correction_checkpoint is not None:
        correction_model = (
            UNetIntensityCorrection.load_from_checkpoint(
                args.correction_checkpoint, map_location="cpu"
            )
            .eval()
            .to(args.device)
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for storm in args.storms:
            reference_path = args.reference_root / storm / "inference-summary.csv"
            table = pd.read_csv(reference_path)
            available = table["observation_id"].isin(by_id)
            if not available.all():
                print(
                    f"Skipping {(~available).sum()} stale ViT observations for {storm} "
                    "that are absent from the current manifest"
                )
                table = table.loc[available].copy()
            if args.limit is not None:
                table = table.head(args.limit).copy()
            iterator = tqdm(table.itertuples(index=False), total=len(table), desc=storm)
            metric_rows, local_paths = [], []
            kept_indices, audit_rows = [], []
            for table_index, row in enumerate(iterator):
                geo = by_id[row.observation_id]
                selected_pmw = None
                selected_gap = None
                pmw_features = None
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
                    grid_lat, grid_lon = _make_grid(
                        geo.center[0], geo.center[1], GRID_SIZE, GRID_RESOLUTION_DEGREES
                    )
                    try:
                        pmw_features, _, selected_gap = prepare_pmw_condition_features(
                            geo,
                            selected_pmw,
                            grid_lat,
                            grid_lon,
                            stats,
                            max_time_gap_hours=pmw_max_gap_hours,
                            include_time_offset=pmw_include_offset,
                            crop_size=CROP_SIZE,
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
                batch, distance_km = _prepare_sample(geo, era5_by_storm[storm], stats)
                if pmw_features is not None:
                    batch["condition"] = torch.cat(
                        [batch["condition"], pmw_features.unsqueeze(0)], dim=1
                    )
                batch = {key: value.to(args.device) for key, value in batch.items()}
                prediction = model.predict_physical(batch)
                bounds = _bounds(*geo.center)
                metrics = _output_metrics(
                    prediction,
                    batch["condition_mask"],
                    distance_km,
                    bounds,
                    geo.ibtracs_center,
                )
                if correction_model is not None:
                    corrected = _corrected_intensity(
                        correction_model,
                        prediction,
                        batch["condition_mask"],
                        geo,
                        intensity_context[storm],
                    )
                    metrics.update(
                        {
                            "raw_unet_max_wind_ms": float(
                                corrected.raw_unet_max_wind_ms.item()
                            ),
                            "correction_ms": float(corrected.correction_ms.item()),
                            "msw": float(corrected.output_msw_ms.item()),
                            "category": int(corrected.output_category.item()),
                        }
                    )
                metric_rows.append(metrics)
                local_paths.append(str(geo.path))
                kept_indices.append(table_index)
                if pmw_enabled:
                    audit_rows.append(
                        pmw_audit_row(geo, selected_pmw, selected_gap, "matched")
                    )
            table = table.iloc[kept_indices].copy().reset_index(drop=True)
            table["original_input_path"] = local_paths
            output_columns = {
                "output_msw_ms": "msw",
                "output_r34_km": "r34",
                "output_r50_km": "r50",
                "output_r64_km": "r64",
                "output_p90_ms": "p90",
                "output_core_mean_ms": "core_mean",
                "output_rmw_km": "rmw",
                "output_mean_ms": "mean",
            }
            if correction_model is not None:
                output_columns.update(
                    {
                        "raw_unet_max_wind_ms": "raw_unet_max_wind_ms",
                        "correction_ms": "correction_ms",
                        "output_category": "category",
                    }
                )
            for column, key in output_columns.items():
                table[column] = [metrics[key] for metrics in metric_rows]
            storm_dir = args.output_root / storm
            storm_dir.mkdir(parents=True, exist_ok=True)
            if correction_model is not None:
                provenance_path = storm_dir / "correction-provenance.json"
                provenance_path.write_text(
                    json.dumps(
                        {
                            "model": "UNet+MLP",
                            "storm_id": storm,
                            "samples": len(table),
                            "unet_checkpoint": {
                                "path": str(args.checkpoint.resolve()),
                                "sha256": _sha256(args.checkpoint),
                            },
                            "correction_checkpoint": {
                                "path": str(args.correction_checkpoint.resolve()),
                                "sha256": _sha256(args.correction_checkpoint),
                            },
                            "intensity_cache_metadata": str(
                                args.intensity_cache_metadata.resolve()
                            ),
                            "scientific_evaluation": correction_provenance.get(
                                "scientific_evaluation", "unspecified"
                            ),
                            "single_timestep": True,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            if pmw_enabled:
                pd.DataFrame(audit_rows).to_csv(
                    storm_dir / "pmw-inference-audit.csv", index=False
                )
            output_path = storm_dir / "inference-summary.csv"
            temporary_path = output_path.with_suffix(".csv.tmp")
            table.to_csv(temporary_path, index=False)
            temporary_path.replace(output_path)
            display_path = (
                output_path.relative_to(ROOT)
                if output_path.is_relative_to(ROOT)
                else output_path
            )
            print(f"Wrote {display_path} ({len(table)} rows)")


if __name__ == "__main__":
    main()
