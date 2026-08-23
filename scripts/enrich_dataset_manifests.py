#!/usr/bin/env python3
"""Add physical wind and storm-geometry diagnostics to exported manifests.

The exporter manifest contains pairing and coordinate metadata, but the target
and ERA5 wind diagnostics live in the rasters.  This utility computes those
diagnostics once and writes them back to the root and split manifests.  It is
idempotent: rerunning it replaces only the derived columns and preserves the
original columns and row order.

IBTrACS intensity/category fields are intentionally not inferred here.  The
local exported manifests contain the IBTrACS center but not the upstream track
intensity table.  An upstream join can be added later without changing the
raster-derived schema.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio

EARTH_RADIUS_KM = 6371.0
R64_THRESHOLD_MS = 64.0 * 0.514444
ROBUST_PEAK_FRACTION = 0.005
RADIAL_BIN_KM = 10.0
RADIAL_MAX_RADIUS_KM = 200.0
RADIAL_MIN_PIXELS = 8
MATURE_PEAK_THRESHOLD_MS = 33.0
HIGH_INTENSITY_PEAK_THRESHOLD_MS = 43.0
WEAK_PEAK_THRESHOLD_MS = 25.0
CORE_RADIUS_KM = 100.0
RING_INNER_RADIUS_KM = 20.0
RING_OUTER_RADIUS_KM = 60.0
MATURE_MIN_REGIONAL_COVERAGE = 0.5
MATURE_MIN_RING_MEAN_MS = 17.0
MATURE_MIN_EYE_CONTRAST_MS = 5.0

U_CHANNELS = {"u_wind_10m", "era5_u_wind_10m", "u10", "era5_u10"}
V_CHANNELS = {"v_wind_10m", "era5_v_wind_10m", "v10", "era5_v10"}

DERIVED_COLUMNS = (
    "metadata_schema_version",
    "storm_sample_index_all",
    "storm_sample_count_all",
    "storm_span_hours_all",
    "storm_elapsed_hours",
    "storm_remaining_hours",
    "era5_time_gap_hours",
    "ibtracs_center_offset_km",
    "sar_valid_pixels",
    "sar_valid_fraction",
    "sar_max_wind_ms",
    "sar_robust_peak_ms",
    "sar_p95_wind_ms",
    "sar_p99_wind_ms",
    "sar_mean_wind_ms",
    "sar_core_valid_pixels_0_100km",
    "sar_core_valid_fraction_0_100km",
    "sar_core_mean_wind_ms",
    "sar_ring_valid_pixels_20_60km",
    "sar_ring_valid_fraction_20_60km",
    "sar_ring_mean_wind_ms",
    "sar_ring_minus_core_mean_ms",
    "sar_eye_valid_pixels_0_20km",
    "sar_eye_valid_fraction_0_20km",
    "sar_eye_mean_wind_ms",
    "sar_eye_contrast_ms",
    "sar_radial_profile_valid_bins",
    "sar_radial_profile_peak_mean_ms",
    "sar_rmw_km",
    "sar_r64_km",
    "era5_valid_pixels",
    "era5_valid_fraction",
    "era5_max_wind_ms",
    "era5_robust_peak_ms",
    "era5_p95_wind_ms",
    "era5_p99_wind_ms",
    "era5_mean_wind_ms",
    "era5_core_valid_pixels_0_100km",
    "era5_core_valid_fraction_0_100km",
    "era5_core_mean_wind_ms",
    "era5_ring_valid_pixels_20_60km",
    "era5_ring_valid_fraction_20_60km",
    "era5_ring_mean_wind_ms",
    "era5_ring_minus_core_mean_ms",
    "era5_eye_valid_pixels_0_20km",
    "era5_eye_valid_fraction_0_20km",
    "era5_eye_mean_wind_ms",
    "era5_eye_contrast_ms",
    "era5_radial_profile_valid_bins",
    "era5_radial_profile_peak_mean_ms",
    "era5_rmw_km",
    "era5_r64_km",
    "sar_minus_era5_max_ms",
    "sar_minus_era5_robust_peak_ms",
    "sar_minus_era5_p95_ms",
    "sar_era5_common_valid_pixels",
    "sar_era5_common_valid_fraction",
    "sar_era5_common_mean_difference_ms",
    "sar_has_valid_center",
    "sar_radial_profile_usable_v1",
    "sar_quality_usable_v1",
    "sar_well_developed_v1",
    "sar_high_intensity_v1",
    "sar_weak_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/geotiff/geo_sar_10bands_era5"),
        help="Export root containing manifest.csv and split directories.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        dest="manifests",
        help="Manifest path relative to --root; repeat to override discovery.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and summarize diagnostics without modifying manifests.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Summary JSON path; defaults to <root>/manifest-metadata-summary.json.",
    )
    return parser.parse_args()


def _manifest_paths(root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        paths = [Path(value) for value in requested]
        return [path if path.is_absolute() else root / path for path in paths]
    paths = [root / "manifest.csv"]
    paths.extend(sorted(root.glob("*/manifest.csv")))
    return [path for path in paths if path.is_file()]


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _json_channels(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _read_raster(path: Path) -> tuple[np.ndarray, rasterio.coords.BoundingBox]:
    with rasterio.open(path) as source:
        values = source.read(masked=True).astype("float32")
        bounds = source.bounds
    return np.asarray(values.filled(np.nan), dtype=np.float64), bounds


def _radius_grid(
    bounds: rasterio.coords.BoundingBox,
    shape: tuple[int, int],
    center_lat: float,
    center_lon: float,
) -> np.ndarray:
    height, width = shape
    lat = (
        bounds.top
        - (np.arange(height, dtype=np.float64) + 0.5)
        * (bounds.top - bounds.bottom)
        / height
    )[:, None]
    lon = (
        bounds.left
        + (np.arange(width, dtype=np.float64) + 0.5)
        * (bounds.right - bounds.left)
        / width
    )
    lat_radians = np.deg2rad(lat)
    center_lat_radians = math.radians(center_lat)
    delta_latitude = lat_radians - center_lat_radians
    delta_longitude = np.deg2rad((lon - center_lon + 180.0) % 360.0 - 180.0)
    haversine = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(lat_radians)
        * math.cos(center_lat_radians)
        * np.sin(delta_longitude / 2.0) ** 2
    )
    return (
        2.0
        * EARTH_RADIUS_KM
        * np.arctan2(
            np.sqrt(np.clip(haversine, 0.0, 1.0)),
            np.sqrt(np.clip(1.0 - haversine, 0.0, 1.0)),
        )
    )


def _robust_peak(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    count = max(1, int(math.ceil(len(values) * ROBUST_PEAK_FRACTION)))
    return float(np.mean(np.partition(values, -count)[-count:]))


def _summary(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "valid_pixels": 0,
            "valid_fraction": 0.0,
            "max": math.nan,
            "robust_peak": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "mean": math.nan,
        }
    return {
        "valid_pixels": int(len(values)),
        "valid_fraction": float(len(values)),
        "max": float(np.max(values)),
        "robust_peak": _robust_peak(values),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "mean": float(np.mean(values)),
    }


def _regional_summary(
    field: np.ndarray,
    valid: np.ndarray,
    radius_km: np.ndarray,
    lower_km: float,
    upper_km: float,
) -> dict[str, float]:
    region = (radius_km >= lower_km) & (radius_km < upper_km)
    values = field[region & valid]
    total = int(region.sum())
    return {
        "valid_pixels": int(len(values)),
        "valid_fraction": float(len(values) / total) if total else math.nan,
        "mean": float(np.mean(values)) if len(values) else math.nan,
    }


def _radial_summary(
    field: np.ndarray,
    valid: np.ndarray,
    radius_km: np.ndarray,
) -> dict[str, float]:
    means: list[tuple[float, float]] = []
    for lower in np.arange(0.0, RADIAL_MAX_RADIUS_KM, RADIAL_BIN_KM):
        upper = lower + RADIAL_BIN_KM
        values = field[(radius_km >= lower) & (radius_km < upper) & valid]
        if len(values) >= RADIAL_MIN_PIXELS:
            means.append((lower + RADIAL_BIN_KM / 2.0, float(np.mean(values))))
    if not means:
        return {
            "valid_bins": 0,
            "peak_mean": math.nan,
            "rmw": math.nan,
            "r64": math.nan,
        }
    radii = np.asarray([item[0] for item in means])
    values = np.asarray([item[1] for item in means])
    return {
        "valid_bins": int(len(means)),
        "peak_mean": float(np.max(values)),
        "rmw": float(radii[int(np.argmax(values))]),
        "r64": (
            float(np.max(radii[values >= R64_THRESHOLD_MS]))
            if np.any(values >= R64_THRESHOLD_MS)
            else math.nan
        ),
    }


def _field_metrics(
    field: np.ndarray,
    radius_km: np.ndarray | None,
    *,
    prefix: str,
) -> dict[str, Any]:
    valid = np.isfinite(field)
    values = field[valid]
    summary = _summary(values)
    total_pixels = int(field.size)
    result: dict[str, Any] = {
        f"{prefix}_valid_pixels": summary["valid_pixels"],
        f"{prefix}_valid_fraction": (
            float(summary["valid_pixels"] / total_pixels) if total_pixels else 0.0
        ),
        f"{prefix}_max_wind_ms": summary["max"],
        f"{prefix}_robust_peak_ms": summary["robust_peak"],
        f"{prefix}_p95_wind_ms": summary["p95"],
        f"{prefix}_p99_wind_ms": summary["p99"],
        f"{prefix}_mean_wind_ms": summary["mean"],
    }
    if radius_km is None or radius_km.shape != field.shape:
        return result
    core = _regional_summary(field, valid, radius_km, 0.0, CORE_RADIUS_KM)
    eye = _regional_summary(field, valid, radius_km, 0.0, 20.0)
    ring = _regional_summary(
        field,
        valid,
        radius_km,
        RING_INNER_RADIUS_KM,
        RING_OUTER_RADIUS_KM,
    )
    radial = _radial_summary(field, valid, radius_km)
    result.update(
        {
            f"{prefix}_core_valid_pixels_0_100km": core["valid_pixels"],
            f"{prefix}_core_valid_fraction_0_100km": core["valid_fraction"],
            f"{prefix}_core_mean_wind_ms": core["mean"],
            f"{prefix}_ring_valid_pixels_20_60km": ring["valid_pixels"],
            f"{prefix}_ring_valid_fraction_20_60km": ring["valid_fraction"],
            f"{prefix}_ring_mean_wind_ms": ring["mean"],
            f"{prefix}_ring_minus_core_mean_ms": (
                ring["mean"] - core["mean"]
                if math.isfinite(ring["mean"]) and math.isfinite(core["mean"])
                else math.nan
            ),
            f"{prefix}_eye_valid_pixels_0_20km": eye["valid_pixels"],
            f"{prefix}_eye_valid_fraction_0_20km": eye["valid_fraction"],
            f"{prefix}_eye_mean_wind_ms": eye["mean"],
            f"{prefix}_eye_contrast_ms": (
                ring["mean"] - eye["mean"]
                if math.isfinite(ring["mean"]) and math.isfinite(eye["mean"])
                else math.nan
            ),
            f"{prefix}_radial_profile_valid_bins": radial["valid_bins"],
            f"{prefix}_radial_profile_peak_mean_ms": radial["peak_mean"],
            f"{prefix}_rmw_km": radial["rmw"],
            f"{prefix}_r64_km": radial["r64"],
        }
    )
    return result


def _center_offset_km(row: pd.Series) -> float:
    center_lat = _finite_float(row.get("ibtracs_center_lat"))
    center_lon = _finite_float(row.get("ibtracs_center_lon"))
    image_lat = _finite_float(row.get("center_lat"))
    image_lon = _finite_float(row.get("center_lon"))
    if not all(
        math.isfinite(value) for value in (center_lat, center_lon, image_lat, image_lon)
    ):
        return math.nan
    lat1, lat2 = math.radians(center_lat), math.radians(image_lat)
    dlat = lat2 - lat1
    dlon = math.radians((image_lon - center_lon + 180.0) % 360.0 - 180.0)
    haversine = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return float(
        2.0
        * EARTH_RADIUS_KM
        * math.atan2(math.sqrt(haversine), math.sqrt(max(0.0, 1.0 - haversine)))
    )


def _era5_wind(
    path: Path, channels: Iterable[str]
) -> tuple[np.ndarray | None, rasterio.coords.BoundingBox | None]:
    channel_names = list(channels)
    u_index = next(
        (index for index, name in enumerate(channel_names) if name in U_CHANNELS), None
    )
    v_index = next(
        (index for index, name in enumerate(channel_names) if name in V_CHANNELS), None
    )
    if u_index is None or v_index is None:
        return None, None
    with rasterio.open(path) as source:
        values = source.read(masked=True).astype("float32")
        bounds = source.bounds
    data = np.asarray(values.filled(np.nan), dtype=np.float64)
    u = data[u_index]
    v = data[v_index]
    valid = np.isfinite(u) & np.isfinite(v)
    wind = np.full_like(u, np.nan, dtype=np.float64)
    wind[valid] = np.hypot(u[valid], v[valid])
    return wind, bounds


def _row_metrics(root: Path, row: pd.Series) -> dict[str, Any]:
    target, target_bounds = _read_raster(root / str(row["target_path"]))
    target_field = target[0]
    center_lat = _finite_float(row.get("ibtracs_center_lat"))
    center_lon = _finite_float(row.get("ibtracs_center_lon"))
    target_radius = (
        _radius_grid(target_bounds, target_field.shape, center_lat, center_lon)
        if math.isfinite(center_lat) and math.isfinite(center_lon)
        else None
    )
    result = _field_metrics(target_field, target_radius, prefix="sar")
    center_valid = False
    if target_radius is not None:
        height, width = target_field.shape
        left, bottom, right, top = (
            target_bounds.left,
            target_bounds.bottom,
            target_bounds.right,
            target_bounds.top,
        )
        target_row = math.floor((top - center_lat) * height / (top - bottom))
        target_column = math.floor((center_lon - left) * width / (right - left))
        center_valid = bool(
            0 <= target_row < height
            and 0 <= target_column < width
            and math.isfinite(float(target_field[target_row, target_column]))
        )
    result.update(
        {
            "sar_has_valid_center": center_valid,
            "ibtracs_center_offset_km": _center_offset_km(row),
            "era5_time_gap_hours": _time_gap_hours(row),
        }
    )

    context_path = str(row.get("context_path", "")).strip()
    context = None
    context_bounds = None
    if context_path:
        context, context_bounds = _era5_wind(
            root / context_path,
            _json_channels(row.get("context_channels", "[]")),
        )
    if context is None:
        result.update(_empty_field_metrics("era5"))
        result.update(_empty_difference_metrics())
        return _finish_flags(result)
    context_radius = (
        _radius_grid(context_bounds, context.shape, center_lat, center_lon)
        if context_bounds is not None
        and math.isfinite(center_lat)
        and math.isfinite(center_lon)
        else None
    )
    result.update(_field_metrics(context, context_radius, prefix="era5"))
    result.update(
        {
            "sar_minus_era5_max_ms": _difference(
                result["sar_max_wind_ms"], result["era5_max_wind_ms"]
            ),
            "sar_minus_era5_robust_peak_ms": _difference(
                result["sar_robust_peak_ms"], result["era5_robust_peak_ms"]
            ),
            "sar_minus_era5_p95_ms": _difference(
                result["sar_p95_wind_ms"], result["era5_p95_wind_ms"]
            ),
        }
    )
    target_valid = np.isfinite(target_field)
    context_valid = np.isfinite(context)
    common = target_valid & context_valid
    result["sar_era5_common_valid_pixels"] = int(common.sum())
    result["sar_era5_common_valid_fraction"] = float(common.mean())
    result["sar_era5_common_mean_difference_ms"] = (
        float(np.mean(target_field[common] - context[common]))
        if common.any()
        else math.nan
    )
    return _finish_flags(result)


def _empty_field_metrics(prefix: str) -> dict[str, Any]:
    result = {
        f"{prefix}_valid_pixels": 0,
        f"{prefix}_valid_fraction": 0.0,
    }
    for suffix in (
        "max_wind_ms",
        "robust_peak_ms",
        "p95_wind_ms",
        "p99_wind_ms",
        "mean_wind_ms",
        "core_valid_fraction_0_100km",
        "core_mean_wind_ms",
        "ring_valid_fraction_20_60km",
        "ring_mean_wind_ms",
        "ring_minus_core_mean_ms",
        "eye_valid_fraction_0_20km",
        "eye_mean_wind_ms",
        "eye_contrast_ms",
        "radial_profile_peak_mean_ms",
        "rmw_km",
        "r64_km",
    ):
        result[f"{prefix}_{suffix}"] = math.nan
    for suffix in (
        "core_valid_pixels_0_100km",
        "ring_valid_pixels_20_60km",
        "eye_valid_pixels_0_20km",
        "radial_profile_valid_bins",
    ):
        result[f"{prefix}_{suffix}"] = 0
    return result


def _empty_difference_metrics() -> dict[str, Any]:
    return {
        "sar_minus_era5_max_ms": math.nan,
        "sar_minus_era5_robust_peak_ms": math.nan,
        "sar_minus_era5_p95_ms": math.nan,
        "sar_era5_common_valid_pixels": 0,
        "sar_era5_common_valid_fraction": 0.0,
        "sar_era5_common_mean_difference_ms": math.nan,
    }


def _difference(left: Any, right: Any) -> float:
    left_float, right_float = _finite_float(left), _finite_float(right)
    return (
        left_float - right_float
        if math.isfinite(left_float) and math.isfinite(right_float)
        else math.nan
    )


def _time_gap_hours(row: pd.Series) -> float:
    era5 = pd.to_datetime(row.get("era5_timestamp", ""), errors="coerce", utc=True)
    reference = pd.NaT
    for column in (
        "target_timestamp",
        "sar_timestamp",
        "condition_timestamp",
        "geo_timestamp",
    ):
        value = pd.to_datetime(row.get(column, ""), errors="coerce", utc=True)
        if not pd.isna(value):
            reference = value
            break
    if pd.isna(era5) or pd.isna(reference):
        return math.nan
    return float(abs((era5 - reference).total_seconds()) / 3600.0)


def _finish_flags(result: dict[str, Any]) -> dict[str, Any]:
    radial_bins = result.get("sar_radial_profile_valid_bins", 0)
    core_coverage = _finite_float(result.get("sar_core_valid_fraction_0_100km"))
    ring_coverage = _finite_float(result.get("sar_ring_valid_fraction_20_60km"))
    ring_mean = _finite_float(result.get("sar_ring_mean_wind_ms"))
    eye_coverage = _finite_float(result.get("sar_eye_valid_fraction_0_20km"))
    eye_contrast = _finite_float(result.get("sar_eye_contrast_ms"))
    robust_peak = _finite_float(result.get("sar_robust_peak_ms"))
    radial_usable = radial_bins >= 8
    quality_usable = (
        radial_usable
        and core_coverage >= MATURE_MIN_REGIONAL_COVERAGE
        and ring_coverage >= MATURE_MIN_REGIONAL_COVERAGE
    )
    result.update(
        {
            "sar_radial_profile_usable_v1": radial_usable,
            "sar_quality_usable_v1": quality_usable,
            "sar_well_developed_v1": bool(
                quality_usable
                and robust_peak >= MATURE_PEAK_THRESHOLD_MS
                and ring_mean >= MATURE_MIN_RING_MEAN_MS
                and eye_coverage >= MATURE_MIN_REGIONAL_COVERAGE
                and eye_contrast >= MATURE_MIN_EYE_CONTRAST_MS
            ),
            "sar_high_intensity_v1": bool(
                robust_peak >= HIGH_INTENSITY_PEAK_THRESHOLD_MS
            ),
            "sar_weak_v1": bool(robust_peak < WEAK_PEAK_THRESHOLD_MS),
        }
    )
    return result


def _storm_metadata(all_rows: pd.DataFrame) -> pd.DataFrame:
    frame = all_rows[["sample_id", "storm_id", "target_timestamp"]].copy()
    frame["_time"] = pd.to_datetime(
        frame["target_timestamp"], errors="coerce", utc=True
    )
    frame = frame.sort_values(["storm_id", "_time", "sample_id"], kind="stable")
    grouped = frame.groupby("storm_id", sort=False)
    frame["storm_sample_index_all"] = grouped.cumcount()
    frame["storm_sample_count_all"] = grouped["sample_id"].transform("size")
    start = grouped["_time"].transform("min")
    end = grouped["_time"].transform("max")
    frame["storm_span_hours_all"] = (end - start).dt.total_seconds() / 3600.0
    frame["storm_elapsed_hours"] = (frame["_time"] - start).dt.total_seconds() / 3600.0
    frame["storm_remaining_hours"] = (end - frame["_time"]).dt.total_seconds() / 3600.0
    return frame.set_index("sample_id")


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _discover_and_load(
    root: Path, requested: list[str] | None
) -> dict[Path, pd.DataFrame]:
    frames: dict[Path, pd.DataFrame] = {}
    for path in _manifest_paths(root, requested):
        frames[path] = pd.read_csv(path, keep_default_na=False, low_memory=False)
    if not frames:
        raise FileNotFoundError(f"No manifests found below {root}")
    return frames


def enrich_manifests(
    root: Path,
    requested: list[str] | None = None,
    *,
    dry_run: bool = False,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    frames = _discover_and_load(root, requested)
    all_rows = pd.concat(frames.values(), ignore_index=True).drop_duplicates(
        "sample_id"
    )
    storm_info = _storm_metadata(all_rows)
    summaries: dict[str, Any] = {}
    for path, frame in frames.items():
        enriched_rows = []
        for row in frame.itertuples(index=False):
            series = pd.Series(row._asdict())
            metrics = _row_metrics(root, series)
            metrics.update(storm_info.loc[str(series["sample_id"])].to_dict())
            metrics["metadata_schema_version"] = 1
            enriched_rows.append(metrics)
        metrics_frame = pd.DataFrame(enriched_rows, index=frame.index)
        output = frame.copy()
        for column in DERIVED_COLUMNS:
            if column in metrics_frame:
                output[column] = metrics_frame[column]
        if not dry_run:
            _write_atomic(output, path)
        summaries[str(path.relative_to(root))] = {
            "samples": int(len(output)),
            "storms": int(output["storm_id"].nunique()),
            "well_developed": int(output["sar_well_developed_v1"].sum()),
            "quality_usable": int(output["sar_quality_usable_v1"].sum()),
            "high_intensity": int(output["sar_high_intensity_v1"].sum()),
            "weak": int(output["sar_weak_v1"].sum()),
        }
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "dry_run": dry_run,
        "metrics": {
            "robust_peak_fraction": ROBUST_PEAK_FRACTION,
            "radial_bin_km": RADIAL_BIN_KM,
            "radial_max_radius_km": RADIAL_MAX_RADIUS_KM,
            "radial_min_pixels": RADIAL_MIN_PIXELS,
            "mature_peak_threshold_ms": MATURE_PEAK_THRESHOLD_MS,
            "high_intensity_peak_threshold_ms": HIGH_INTENSITY_PEAK_THRESHOLD_MS,
            "weak_peak_threshold_ms": WEAK_PEAK_THRESHOLD_MS,
            "mature_min_regional_coverage": MATURE_MIN_REGIONAL_COVERAGE,
            "mature_min_ring_mean_ms": MATURE_MIN_RING_MEAN_MS,
            "mature_min_eye_contrast_ms": MATURE_MIN_EYE_CONTRAST_MS,
        },
        "unavailable_upstream_fields": [
            "ibtracs_msw_ms",
            "ibtracs_category",
            "ibtracs_r64_mean_km",
            "storm_intensity_phase",
        ],
        "manifests": summaries,
    }
    if summary_path is None:
        summary_path = root / "manifest-metadata-summary.json"
    if not dry_run:
        summary_path = summary_path.expanduser()
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> None:
    args = parse_args()
    summary = enrich_manifests(
        args.root,
        args.manifests,
        dry_run=args.dry_run,
        summary_path=args.summary_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
