from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep numerical libraries polite on HPC login nodes.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import yaml
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

EPSG_4326 = "EPSG:4326"
DEFAULT_DATA_ROOT = Path("/lustre/scratch/1054/tropical_cyclone_dynamics/data")
DEFAULT_MANIFEST_FILE = (
    DEFAULT_DATA_ROOT / "index-files" / "observation_manifest_v5.csv"
)
DEFAULT_OUTPUT_ROOT = Path("data/geotiff/geo_sar")
DEFAULT_GRID_SIZE = 256
DEFAULT_GRID_RESOLUTION = 0.027
DEFAULT_CLOSEST_MATCH_HOURS = 0.5
DEFAULT_SPLITS = ("train", "val", "test")
GEO_CHANNELS = {
    "ABI": ("CMI_C08", "CMI_C09", "CMI_C13", "CMI_C14"),
    "AHI": ("B08", "B09", "B13", "B14"),
}
SAR_CHANNELS = ("wind_speed",)


@dataclass(frozen=True)
class Observation:
    observation_id: str
    storm_id: str
    split: str
    source_type: str
    source: str
    sensor: str
    path: Path
    timestamp: pd.Timestamp
    center_lat: float | None
    center_lon: float | None
    ibtracs_center_lat: float | None
    ibtracs_center_lon: float | None
    variables: tuple[str, ...]

    @property
    def center(self) -> tuple[float, float] | None:
        if self.center_lat is None or self.center_lon is None:
            return None
        return self.center_lat, self.center_lon

    @property
    def ibtracs_center(self) -> tuple[float, float] | None:
        if self.ibtracs_center_lat is None or self.ibtracs_center_lon is None:
            return None
        return self.ibtracs_center_lat, self.ibtracs_center_lon


@dataclass
class StatsAccumulator:
    sums: dict[str, float]
    sq_sums: dict[str, float]
    mins: dict[str, float]
    maxs: dict[str, float]
    counts: dict[str, int]

    @classmethod
    def create(cls) -> "StatsAccumulator":
        return cls(defaultdict(float), defaultdict(float), {}, {}, defaultdict(int))

    def update(self, source_type: str, name: str, values: np.ndarray, mask: np.ndarray) -> None:
        key = f"{source_type}:{name}"
        valid = values[np.isfinite(values) & mask]
        if valid.size == 0:
            return
        valid64 = valid.astype(np.float64)
        self.sums[key] += float(valid64.sum())
        self.sq_sums[key] += float(np.square(valid64).sum())
        self.counts[key] += int(valid.size)
        self.mins[key] = float(valid64.min()) if key not in self.mins else min(self.mins[key], float(valid64.min()))
        self.maxs[key] = float(valid64.max()) if key not in self.maxs else max(self.maxs[key], float(valid64.max()))

    def to_jsonable(self) -> dict[str, Any]:
        channels: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
        for key, count in sorted(self.counts.items()):
            if count <= 0:
                continue
            source_type, name = key.split(":", 1)
            mean = self.sums[key] / count
            variance = max(self.sq_sums[key] / count - mean * mean, 0.0)
            channels[source_type][name] = {
                "min": self.mins[key],
                "max": self.maxs[key],
                "mean": mean,
                "std": math.sqrt(variance),
                "count": count,
            }
        return {
            "normalization": "min-max",
            "source_key_format": "{source_type}:{channel}",
            "channels": {source: dict(values) for source, values in channels.items()},
        }


def main() -> None:
    args = _parse_args()
    export_geo_sar_geotiffs(
        data_root=args.data_root,
        manifest_file=args.manifest_file,
        output_root=args.output_root,
        splits=args.splits,
        grid_size=args.grid_size,
        grid_resolution=args.grid_resolution,
        closest_match_hours=args.closest_match_hours,
        center=args.center,
        shift_center=args.shift_center,
        pad=args.pad,
        limit=args.limit,
    )


def _parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    known, _ = config_parser.parse_known_args()
    config = _load_export_config(known.config)

    parser = argparse.ArgumentParser(
        description="Export SAR-anchored GEO/SAR pairs as raw-value GeoTIFFs."
    )
    parser.add_argument("--config", type=Path, default=known.config)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(config.get("data_root", DEFAULT_DATA_ROOT)),
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=Path(config.get("manifest_file", DEFAULT_MANIFEST_FILE)),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(config.get("output_root", DEFAULT_OUTPUT_ROOT)),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(config.get("splits", DEFAULT_SPLITS)),
        choices=list(DEFAULT_SPLITS),
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=int(config.get("grid_size", DEFAULT_GRID_SIZE)),
    )
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=float(config.get("grid_resolution", DEFAULT_GRID_RESOLUTION)),
    )
    parser.add_argument(
        "--closest-match-hours",
        type=float,
        default=float(
            config.get("closest_match_hours", DEFAULT_CLOSEST_MATCH_HOURS)
        ),
    )
    parser.add_argument(
        "--center",
        choices=["image_center", "ibtracs_center"],
        default=str(config.get("center", "image_center")),
    )
    parser.add_argument(
        "--shift-center",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("shift_center", True)),
        help="Shift image-centered grids just enough to include IBTrACS center.",
    )
    parser.add_argument("--pad", type=int, default=int(config.get("pad", 8)))
    parser.add_argument(
        "--limit",
        type=int,
        default=config.get("limit"),
        help="Maximum exported samples per split, useful for smoke tests.",
    )
    return parser.parse_args()


def _load_export_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return dict(config.get("export", {}))


def export_geo_sar_geotiffs(
    *,
    data_root: Path,
    manifest_file: Path,
    output_root: Path,
    splits: Sequence[str],
    grid_size: int,
    grid_resolution: float,
    closest_match_hours: float,
    center: str,
    shift_center: bool,
    pad: int,
    limit: int | None = None,
) -> None:
    data_root = data_root.expanduser().resolve()
    manifest_file = manifest_file.expanduser().resolve()
    output_root = output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    records = _read_manifest(manifest_file, data_root)
    by_split = {split: [] for split in splits}
    no_match_rows: list[dict[str, Any]] = []
    train_stats = StatsAccumulator.create()

    for split in splits:
        split_records = [record for record in records if record.split == split]
        pairs = _pair_sar_to_geo(split_records, closest_match_hours)
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        skipped = 0

        iterator = tqdm(pairs, desc=f"export {split}", unit="sample")
        for sar, geo, dt_minutes in iterator:
            if limit is not None and len(rows) >= limit:
                break
            try:
                sample = _build_sample(
                    sar=sar,
                    geo=geo,
                    grid_size=grid_size,
                    grid_resolution=grid_resolution,
                    center=center,
                    shift_center=shift_center,
                    pad=pad,
                )
            except (KeyError, OSError, ValueError) as error:
                skipped += 1
                no_match_rows.append(
                    {
                        "split": split,
                        "storm_id": sar.storm_id,
                        "sar_observation_id": sar.observation_id,
                        "geo_observation_id": geo.observation_id,
                        "reason": str(error),
                    }
                )
                continue

            sample_id = _sample_id(sar, geo)
            geo_path = split_dir / f"{sample_id}_geo.tif"
            sar_path = split_dir / f"{sample_id}_sar.tif"
            tags = _metadata_tags(
                sample_id=sample_id,
                sar=sar,
                geo=geo,
                dt_minutes=dt_minutes,
                sample=sample,
            )
            _write_geotiff(
                geo_path,
                sample["geo"],
                sample["geo_mask"],
                sample["geo_channels"],
                sample["transform"],
                tags | {"role": "condition", "source_type": "geo"},
            )
            _write_geotiff(
                sar_path,
                sample["sar"],
                sample["sar_mask"],
                sample["sar_channels"],
                sample["transform"],
                tags | {"role": "target", "source_type": "sar"},
            )
            if split == "train":
                _update_stats(train_stats, sample)

            row = {
                "sample_id": sample_id,
                "split": split,
                "storm_id": sar.storm_id,
                "condition_path": str(geo_path.relative_to(output_root)),
                "target_path": str(sar_path.relative_to(output_root)),
                "condition_source_type": "geo",
                "target_source_type": "sar",
                "condition_observation_id": geo.observation_id,
                "target_observation_id": sar.observation_id,
                "condition_timestamp": geo.timestamp.isoformat(),
                "target_timestamp": sar.timestamp.isoformat(),
                "condition_sensor": geo.sensor,
                "target_sensor": sar.sensor,
                "condition_channels": json.dumps(sample["geo_channels"]),
                "target_channels": json.dumps(sample["sar_channels"]),
                "geo_path": str(geo_path.relative_to(output_root)),
                "sar_path": str(sar_path.relative_to(output_root)),
                "geo_observation_id": geo.observation_id,
                "sar_observation_id": sar.observation_id,
                "geo_timestamp": geo.timestamp.isoformat(),
                "sar_timestamp": sar.timestamp.isoformat(),
                "dt_minutes": dt_minutes,
                "geo_sensor": geo.sensor,
                "sar_sensor": sar.sensor,
                "geo_channels": json.dumps(sample["geo_channels"]),
                "sar_channels": json.dumps(sample["sar_channels"]),
                "grid_size": grid_size,
                "grid_resolution": grid_resolution,
                "center_lat": sample["center_lat"],
                "center_lon": sample["center_lon"],
                "ibtracs_center_lat": sar.ibtracs_center_lat,
                "ibtracs_center_lon": sar.ibtracs_center_lon,
            }
            rows.append(row)
            by_split[split].append(row)

        _write_manifest(split_dir / "manifest.csv", rows)
        print(f"[{split}] wrote {len(rows)} samples to {split_dir} ({skipped} skipped)")

    _write_manifest(output_root / "manifest.csv", [row for split in splits for row in by_split[split]])
    (output_root / "stats.json").write_text(
        json.dumps(train_stats.to_jsonable(), indent=2), encoding="utf-8"
    )
    if no_match_rows:
        _write_manifest(output_root / "skipped.csv", no_match_rows)


def _read_manifest(manifest_file: Path, data_root: Path) -> list[Observation]:
    frame = pd.read_csv(manifest_file, keep_default_na=False, low_memory=False)
    required = {
        "observation_id",
        "storm_id",
        "split",
        "source_type",
        "source",
        "sensor",
        "path",
        "timestamp",
        "center_lat",
        "center_lon",
        "ibtracs_center_lat",
        "ibtracs_center_lon",
        "variables",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    frame = frame[frame["source_type"].isin({"geo", "sar"})]

    records: list[Observation] = []
    for _, row in frame.iterrows():
        timestamp = _optional_timestamp(row["timestamp"])
        if timestamp is None:
            continue
        path = Path(str(row["path"])).expanduser()
        if not path.is_absolute():
            path = data_root / path
        record = Observation(
            observation_id=str(row["observation_id"]),
            storm_id=str(row["storm_id"]).upper(),
            split=str(row["split"]),
            source_type=str(row["source_type"]),
            source=str(row["source"]),
            sensor=str(row["sensor"]),
            path=path,
            timestamp=timestamp,
            center_lat=_optional_float(row["center_lat"]),
            center_lon=_optional_float(row["center_lon"]),
            ibtracs_center_lat=_optional_float(row["ibtracs_center_lat"]),
            ibtracs_center_lon=_optional_float(row["ibtracs_center_lon"]),
            variables=tuple(_json_list(row["variables"])),
        )
        if record.source_type == "geo" and record.sensor not in GEO_CHANNELS:
            continue
        if record.source_type == "sar" and "wind_speed" not in record.variables:
            continue
        records.append(record)
    return records


def _pair_sar_to_geo(
    records: Sequence[Observation],
    closest_match_hours: float,
) -> list[tuple[Observation, Observation, float]]:
    by_storm: dict[str, list[Observation]] = defaultdict(list)
    for record in records:
        by_storm[record.storm_id].append(record)

    max_minutes = closest_match_hours * 60.0
    pairs: list[tuple[Observation, Observation, float]] = []
    for storm_records in by_storm.values():
        geo_records = sorted(
            (record for record in storm_records if record.source_type == "geo"),
            key=lambda record: record.timestamp,
        )
        sar_records = sorted(
            (record for record in storm_records if record.source_type == "sar"),
            key=lambda record: record.timestamp,
        )
        if not geo_records:
            continue
        geo_times = pd.DatetimeIndex([record.timestamp for record in geo_records])
        for sar in sar_records:
            nearest_index = _nearest_time_index(geo_times, sar.timestamp)
            if nearest_index is None:
                continue
            geo = geo_records[nearest_index]
            dt_minutes = (geo.timestamp - sar.timestamp).total_seconds() / 60.0
            if abs(dt_minutes) <= max_minutes:
                pairs.append((sar, geo, dt_minutes))
    return pairs


def _nearest_time_index(
    times: pd.DatetimeIndex,
    timestamp: pd.Timestamp,
) -> int | None:
    if len(times) == 0:
        return None
    index = int(times.searchsorted(timestamp))
    candidates = []
    if index < len(times):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return None
    return min(candidates, key=lambda idx: abs((times[idx] - timestamp).total_seconds()))


def _build_sample(
    *,
    sar: Observation,
    geo: Observation,
    grid_size: int,
    grid_resolution: float,
    center: str,
    shift_center: bool,
    pad: int,
) -> dict[str, Any]:
    center_point = _grid_center(sar, center, shift_center, grid_size, grid_resolution, pad)
    if center_point is None:
        raise ValueError("SAR observation has no usable grid center")
    grid_lat, grid_lon = _make_grid(center_point[0], center_point[1], grid_size, grid_resolution)

    geo_channels = GEO_CHANNELS[geo.sensor]
    sar_channels = SAR_CHANNELS
    geo_fields = _load_geo_channels(geo, geo_channels)
    sar_fields = _load_sar_channels(sar, sar_channels)
    _require_channels("geo", geo_channels, geo_fields)
    _require_channels("sar", sar_channels, sar_fields)

    geo_regridded = [_regrid(*geo_fields[name], grid_lat, grid_lon) for name in geo_channels]
    sar_regridded = [_regrid(*sar_fields[name], grid_lat, grid_lon) for name in sar_channels]
    geo_array = np.stack([values for values, _ in geo_regridded]).astype(np.float32)
    sar_array = np.stack([values for values, _ in sar_regridded]).astype(np.float32)
    geo_masks = np.stack([mask for _, mask in geo_regridded])
    sar_masks = np.stack([mask for _, mask in sar_regridded])
    geo_mask = np.all(geo_masks & np.isfinite(geo_array), axis=0)
    sar_mask = np.all(sar_masks & np.isfinite(sar_array), axis=0)
    if not geo_mask.any():
        raise ValueError("No valid GEO pixels after regridding")
    if not sar_mask.any():
        raise ValueError("No valid SAR pixels after regridding")

    transform = _grid_transform(center_point[0], center_point[1], grid_size, grid_resolution)
    return {
        "geo": geo_array,
        "sar": sar_array,
        "geo_mask": geo_mask,
        "sar_mask": sar_mask,
        "geo_band_masks": geo_masks,
        "sar_band_masks": sar_masks,
        "geo_channels": tuple(geo_channels),
        "sar_channels": tuple(sar_channels),
        "center_lat": center_point[0],
        "center_lon": center_point[1],
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "transform": transform,
    }


def _grid_center(
    sar: Observation,
    center: str,
    shift_center: bool,
    grid_size: int,
    resolution: float,
    pad: int,
) -> tuple[float, float] | None:
    if center == "ibtracs_center":
        return sar.ibtracs_center
    center_point = sar.center
    if center_point is None:
        center_point = sar.ibtracs_center
    if (
        center_point is not None
        and shift_center
        and sar.ibtracs_center is not None
    ):
        center_point = _shift_grid_center(
            center_point[0],
            center_point[1],
            sar.ibtracs_center[0],
            sar.ibtracs_center[1],
            grid_size,
            resolution,
            pad,
        )
    return center_point


def _make_grid(center_lat: float, center_lon: float, size: int, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    half = (size - 1) / 2
    offsets = (np.arange(size) - half) * resolution
    lat = center_lat - offsets
    lon = center_lon + offsets
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lat2d.astype(np.float64), lon2d.astype(np.float64)


def _grid_transform(center_lat: float, center_lon: float, size: int, resolution: float):
    half = (size - 1) / 2
    west = center_lon - half * resolution - resolution / 2
    north = center_lat + half * resolution + resolution / 2
    return from_origin(west, north, resolution, resolution)


def _shift_grid_center(
    center_lat: float,
    center_lon: float,
    ibtracs_lat: float,
    ibtracs_lon: float,
    size: int,
    resolution: float,
    pad: int,
) -> tuple[float, float]:
    half_offset = ((size - 1) / 2 - pad) * resolution
    if (center_lat - half_offset) > ibtracs_lat:
        center_lat = ibtracs_lat + half_offset
    elif (center_lat + half_offset) < ibtracs_lat:
        center_lat = ibtracs_lat - half_offset
    if (center_lon - half_offset) > ibtracs_lon:
        center_lon = ibtracs_lon + half_offset
    elif (center_lon + half_offset) < ibtracs_lon:
        center_lon = ibtracs_lon - half_offset
    return center_lat, center_lon


def _load_geo_channels(
    observation: Observation,
    channels: Sequence[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with xr.open_dataset(observation.path, engine="h5netcdf") as source:
        dataset = source.load()
    if "data" in dataset and "channel" in dataset["data"].dims:
        values = dataset["data"].sel(channel=list(channels)).values
    else:
        missing = [channel for channel in channels if channel not in dataset]
        if missing:
            raise KeyError(f"GEO dataset is missing channels: {missing}")
        values = np.stack([dataset[channel].values for channel in channels])
    lat = _coordinate_values(dataset, "latitude", "lat")
    lon = _coordinate_values(dataset, "longitude", "lon")
    if np.nanmedian(lat) > 90:
        lat = lat - 180.0
    lon = _fix_longitudes(lon)
    values = np.asarray(values, dtype=np.float32)
    return {
        channel: (values[index], lat, lon)
        for index, channel in enumerate(channels)
    }


def _load_sar_channels(
    observation: Observation,
    channels: Sequence[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with xr.open_dataset(observation.path, engine="h5netcdf") as source:
        dataset = source.load()
    lat = _coordinate_values(dataset, "latitude", "lat")
    lon = _fix_longitudes(_coordinate_values(dataset, "longitude", "lon"))
    result = {}
    for channel in channels:
        if channel not in dataset:
            continue
        values = dataset[channel]
        if "time" in values.dims:
            values = values.isel(time=0)
        result[channel] = (np.asarray(values.values, dtype=np.float32), lat, lon)
    return result


def _coordinate_values(dataset: xr.Dataset, primary: str, fallback: str) -> np.ndarray:
    name = primary if primary in dataset else fallback
    if name not in dataset:
        raise KeyError(f"Dataset is missing {primary!r}/{fallback!r}")
    values = np.asarray(dataset[name].values, dtype=float)
    if values.ndim == 1:
        other_name = "longitude" if name in {"latitude", "lat"} else "latitude"
        other_fallback = "lon" if other_name == "longitude" else "lat"
        paired_name = other_name if other_name in dataset else other_fallback
        if paired_name not in dataset:
            raise KeyError(f"Dataset is missing paired coordinate {paired_name!r}")
        paired = np.asarray(dataset[paired_name].values, dtype=float)
        if name in {"latitude", "lat"}:
            values, _ = np.meshgrid(values, paired, indexing="ij")
        else:
            _, values = np.meshgrid(paired, values, indexing="ij")
    return values


def _regrid(
    values: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(values) & np.isfinite(src_lat) & np.isfinite(src_lon)
    if not ok.any():
        return np.full(grid_lat.shape, np.nan, np.float32), np.zeros(grid_lat.shape, bool)
    coslat = np.cos(np.deg2rad(np.nanmean(grid_lat)))
    tree = cKDTree(np.column_stack([src_lat[ok], src_lon[ok] * coslat]))
    dist, idx = tree.query(np.column_stack([grid_lat.ravel(), grid_lon.ravel() * coslat]))
    spacing = _source_spacing(src_lat, src_lon, coslat)
    result = values[ok][idx].astype(np.float32)
    mask = dist <= 1.5 * spacing
    result[~mask] = np.nan
    return result.reshape(grid_lat.shape), mask.reshape(grid_lat.shape)


def _source_spacing(src_lat: np.ndarray, src_lon: np.ndarray, coslat: float) -> float:
    spacings = []
    if src_lat.shape[-1] > 1:
        spacings.append(np.hypot(np.diff(src_lat, axis=-1), np.diff(src_lon, axis=-1) * coslat))
    if src_lat.ndim > 1 and src_lat.shape[-2] > 1:
        spacings.append(np.hypot(np.diff(src_lat, axis=-2), np.diff(src_lon, axis=-2) * coslat))
    finite_parts = [
        spacing[np.isfinite(spacing)].ravel()
        for spacing in spacings
        if np.isfinite(spacing).any()
    ]
    if not finite_parts:
        return 0.05
    finite = np.concatenate(finite_parts)
    if finite.size == 0:
        return 0.05
    spacing = float(np.nanmedian(finite))
    return spacing if spacing > 0 else 0.05


def _write_geotiff(
    path: Path,
    array: np.ndarray,
    mask: np.ndarray,
    band_names: Sequence[str],
    transform: Any,
    tags: Mapping[str, Any],
) -> None:
    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": "float32",
        "crs": EPSG_4326,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 32,
        "blockysize": 32,
    }
    clean = np.where(mask[None, :, :], array, np.nan).astype(np.float32)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, "w", **profile) as dataset:
            dataset.write(clean)
            dataset.write_mask(mask.astype("uint8") * 255)
            for band_index, name in enumerate(band_names, start=1):
                dataset.set_band_description(band_index, name)
            dataset.update_tags(**{key: _tag_value(value) for key, value in tags.items()})


def _metadata_tags(
    *,
    sample_id: str,
    sar: Observation,
    geo: Observation,
    dt_minutes: float,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "storm_id": sar.storm_id,
        "split": sar.split,
        "sar_observation_id": sar.observation_id,
        "geo_observation_id": geo.observation_id,
        "sar_timestamp": sar.timestamp.isoformat(),
        "geo_timestamp": geo.timestamp.isoformat(),
        "dt_minutes": dt_minutes,
        "sar_sensor": sar.sensor,
        "geo_sensor": geo.sensor,
        "sar_channels": json.dumps(sample["sar_channels"]),
        "geo_channels": json.dumps(sample["geo_channels"]),
        "center_lat": sample["center_lat"],
        "center_lon": sample["center_lon"],
        "ibtracs_center_lat": sar.ibtracs_center_lat,
        "ibtracs_center_lon": sar.ibtracs_center_lon,
    }


def _update_stats(stats: StatsAccumulator, sample: Mapping[str, Any]) -> None:
    for index, channel in enumerate(sample["geo_channels"]):
        stats.update("geo", channel, sample["geo"][index], sample["geo_band_masks"][index])
        stats.update("geo", f"band_{index}", sample["geo"][index], sample["geo_band_masks"][index])
    for index, channel in enumerate(sample["sar_channels"]):
        stats.update("sar", channel, sample["sar"][index], sample["sar_band_masks"][index])
        stats.update("sar", f"band_{index}", sample["sar"][index], sample["sar_band_masks"][index])


def _write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample_id(sar: Observation, geo: Observation) -> str:
    time_tag = sar.timestamp.strftime("%Y%m%d%H%M%S")
    digest = hashlib.blake2s(
        f"{sar.observation_id}|{geo.observation_id}".encode("utf-8"),
        digest_size=4,
    ).hexdigest()
    return f"{_safe_text(sar.storm_id)}_sar_geo_{time_tag}_{digest}"


def _require_channels(
    source_type: str,
    requested: Sequence[str],
    loaded: Mapping[str, Any],
) -> None:
    missing = [channel for channel in requested if channel not in loaded]
    if missing:
        raise KeyError(f"Missing {source_type} channels: {missing}")


def _fix_longitudes(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    return np.where(lon > 180.0, lon - 360.0, lon)


def _optional_timestamp(value: Any) -> pd.Timestamp | None:
    if _missing(value):
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _optional_float(value: Any) -> float | None:
    if _missing(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _json_list(value: Any) -> list[str]:
    if _missing(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _tag_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return str(value)


def _safe_text(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    main()
