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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_env import load_local_env

load_local_env()

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
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

EPSG_4326 = "EPSG:4326"
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "TCD_DATA_ROOT",
        "/lustre/scratch/1054/tropical_cyclone_dynamics/data",
    )
)
DEFAULT_MANIFEST_FILE = (
    DEFAULT_DATA_ROOT / "index-files" / "observation_manifest_v5.csv"
)
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("GEO_SAR_OUTPUT_ROOT", "data/geotiff/geo_sar"))
DEFAULT_GRID_SIZE = 256
DEFAULT_GRID_RESOLUTION = 0.027
DEFAULT_CLOSEST_MATCH_HOURS = 0.5
DEFAULT_SPLITS = ("train", "val", "test")
GEO_CHANNEL_SETS = {
    "common4": {
        "ABI": ("CMI_C08", "CMI_C09", "CMI_C13", "CMI_C14"),
        "AHI": ("B08", "B09", "B13", "B14"),
    },
    "common10": {
        "ABI": tuple(f"CMI_C{index:02d}" for index in range(7, 17)),
        "AHI": tuple(f"B{index:02d}" for index in range(7, 17)),
    },
}
GEO_CHANNELS = GEO_CHANNEL_SETS["common4"]
GEO_CHANNEL_SET = "common4"
SAR_CHANNELS = ("wind_speed",)
ERA5_CHANNELS = (
    "precipitable_water",
    "sst",
    "pressure_msl",
    "temperature_2m",
    "dewpoint_2m",
    "u_wind_10m",
    "v_wind_10m",
)
ERA5_DERIVED_CHANNELS = ("wind_speed_10m", "relative_vorticity_10m")
EARTH_RADIUS_M = 6_371_000.0
DEFAULT_ERA5_MAX_TIME_GAP_HOURS = 3.1


@dataclass(frozen=True)
class Observation:
    observation_id: str
    storm_id: str
    split: str
    source_type: str
    source: str
    sensor: str
    path: Path
    timestamp: pd.Timestamp | None
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
    samples: dict[str, np.ndarray]
    robust_sample_size: int = 65_536
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False
    )

    @classmethod
    def create(
        cls, robust_sample_size: int = 65_536, seed: int = 0
    ) -> "StatsAccumulator":
        if robust_sample_size <= 0:
            raise ValueError("robust_sample_size must be positive")
        return cls(
            defaultdict(float),
            defaultdict(float),
            {},
            {},
            defaultdict(int),
            {},
            robust_sample_size,
            np.random.default_rng(seed),
        )

    def update(self, source_type: str, name: str, values: np.ndarray, mask: np.ndarray) -> None:
        key = f"{source_type}:{name}"
        valid = values[np.isfinite(values) & mask]
        if valid.size == 0:
            return
        valid64 = valid.astype(np.float64, copy=False).ravel()
        previous_count = self.counts[key]
        self.sums[key] += float(valid64.sum())
        self.sq_sums[key] += float(np.square(valid64).sum())
        self.counts[key] += int(valid.size)
        self.mins[key] = float(valid64.min()) if key not in self.mins else min(self.mins[key], float(valid64.min()))
        self.maxs[key] = float(valid64.max()) if key not in self.maxs else max(self.maxs[key], float(valid64.max()))
        self._update_sample(key, valid64, previous_count)

    def _update_sample(
        self, key: str, incoming: np.ndarray, previous_count: int
    ) -> None:
        """Merge a batch into a bounded uniform reservoir without pixel loops."""
        total_count = previous_count + incoming.size
        sample_size = min(self.robust_sample_size, total_count)
        current = self.samples.get(key, np.empty(0, dtype=np.float64))
        if total_count <= self.robust_sample_size:
            self.samples[key] = np.concatenate([current, incoming.copy()])
            return
        incoming_keep = int(
            self.rng.hypergeometric(
                ngood=incoming.size,
                nbad=previous_count,
                nsample=sample_size,
            )
        )
        current_keep = sample_size - incoming_keep
        current_indices = self.rng.choice(
            current.size, size=current_keep, replace=False
        )
        incoming_indices = self.rng.choice(
            incoming.size, size=incoming_keep, replace=False
        )
        self.samples[key] = np.concatenate(
            [current[current_indices], incoming[incoming_indices]]
        )

    def to_jsonable(self) -> dict[str, Any]:
        channels: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
        for key, count in sorted(self.counts.items()):
            if count <= 0:
                continue
            source_type, name = key.split(":", 1)
            mean = self.sums[key] / count
            variance = max(self.sq_sums[key] / count - mean * mean, 0.0)
            sample = self.samples[key]
            q25, median, q75 = np.quantile(sample, [0.25, 0.5, 0.75])
            std = math.sqrt(variance)
            robust_scale = float((q75 - q25) / 1.349)
            if not math.isfinite(robust_scale) or robust_scale <= 1e-12:
                robust_scale = max(std, 1e-6)
            channels[source_type][name] = {
                "min": self.mins[key],
                "max": self.maxs[key],
                "mean": mean,
                "std": std,
                "q25": float(q25),
                "median": float(median),
                "q75": float(q75),
                "robust_scale": robust_scale,
                "count": count,
            }
        return {
            "normalization": "min-max",
            "available_normalizations": ["min-max", "robust-zscore"],
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
        include_era5=args.include_era5,
        era5_channels=tuple(args.era5_channels),
        era5_max_time_gap_hours=args.era5_max_time_gap_hours,
        geo_channel_set=args.geo_channel_set,
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
    data_root = Path(os.environ.get("TCD_DATA_ROOT", config.get("data_root", DEFAULT_DATA_ROOT)))
    manifest_file = Path(
        config.get("manifest_file", data_root / "index-files" / "observation_manifest_v5.csv")
    )
    if "TCD_DATA_ROOT" in os.environ:
        manifest_file = data_root / "index-files" / "observation_manifest_v5.csv"

    parser = argparse.ArgumentParser(
        description="Export SAR-anchored GEO/SAR pairs as raw-value GeoTIFFs."
    )
    parser.add_argument("--config", type=Path, default=known.config)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=data_root,
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=manifest_file,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        # An explicit experiment config must not be silently redirected to a
        # different dataset by the generic machine-local default. CLI remains
        # the highest-precedence override through argparse.
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
        "--geo-channel-set",
        choices=sorted(GEO_CHANNEL_SETS),
        default=str(config.get("geo_channel_set", GEO_CHANNEL_SET)),
        help="Named GEO band set to export.",
    )
    parser.add_argument(
        "--include-era5",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("include_era5", False)),
        help="Append nearest-time ERA5 fields to the GEO condition tensor.",
    )
    parser.add_argument(
        "--era5-channels",
        nargs="+",
        default=list(config.get("era5_channels", ERA5_CHANNELS)),
        help="ERA5 rectilinear variables to append when --include-era5 is active.",
    )
    parser.add_argument(
        "--era5-max-time-gap-hours",
        type=float,
        default=float(
            config.get(
                "era5_max_time_gap_hours",
                DEFAULT_ERA5_MAX_TIME_GAP_HOURS,
            )
        ),
        help="Reject the nearest ERA5 analysis when it is older than this limit.",
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


def _geo_channels_by_sensor(channel_set: str) -> Mapping[str, tuple[str, ...]]:
    try:
        return GEO_CHANNEL_SETS[channel_set]
    except KeyError as error:
        raise ValueError(
            f"Unknown GEO channel set {channel_set!r}; "
            f"choose one of {sorted(GEO_CHANNEL_SETS)}"
        ) from error


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
    include_era5: bool = False,
    era5_channels: Sequence[str] = ERA5_CHANNELS,
    era5_max_time_gap_hours: float = DEFAULT_ERA5_MAX_TIME_GAP_HOURS,
    geo_channel_set: str = GEO_CHANNEL_SET,
    limit: int | None = None,
) -> None:
    data_root = data_root.expanduser().resolve()
    manifest_file = manifest_file.expanduser().resolve()
    output_root = output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    geo_channels_by_sensor = _geo_channels_by_sensor(geo_channel_set)
    records = _read_manifest(manifest_file, data_root)
    _audit_geo_channels(records, geo_channels_by_sensor)
    era5_by_storm = _records_by_storm(records, "era5") if include_era5 else {}
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
                    era5=_nearest_storm_record(era5_by_storm, geo) if include_era5 else None,
                    grid_size=grid_size,
                    grid_resolution=grid_resolution,
                    geo_channels_by_sensor=geo_channels_by_sensor,
                    era5_channels=era5_channels,
                    era5_max_time_gap_hours=era5_max_time_gap_hours,
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
            era5_path = split_dir / f"{sample_id}_era5.tif" if sample.get("era5") is not None else None
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
            if era5_path is not None:
                _write_geotiff(
                    era5_path,
                    sample["era5_array"],
                    sample["era5_mask"],
                    sample["era5_band_names"],
                    sample["transform"],
                    tags | {"role": "context", "source_type": "era5"},
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
                "context_path": str(era5_path.relative_to(output_root)) if era5_path is not None else "",
                "context_source_type": "era5" if era5_path is not None else "",
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
                "context_channels": json.dumps(sample.get("era5_band_names", ())),
                "target_channels": json.dumps(sample["sar_channels"]),
                "geo_path": str(geo_path.relative_to(output_root)),
                "era5_path": str(era5_path.relative_to(output_root)) if era5_path is not None else "",
                "sar_path": str(sar_path.relative_to(output_root)),
                "geo_observation_id": geo.observation_id,
                "sar_observation_id": sar.observation_id,
                "geo_timestamp": geo.timestamp.isoformat(),
                "sar_timestamp": sar.timestamp.isoformat(),
                "dt_minutes": dt_minutes,
                "geo_sensor": geo.sensor,
                "sar_sensor": sar.sensor,
                "geo_channels": json.dumps(sample["geo_channels"]),
                "era5_observation_id": _optional_observation_id(sample.get("era5")),
                "era5_timestamp": _optional_isoformat(sample.get("era5_timestamp")),
                "era5_channels": json.dumps(sample.get("era5_channels", ())),
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
    frame = frame[frame["source_type"].isin({"geo", "sar", "era5"})]

    records: list[Observation] = []
    for _, row in frame.iterrows():
        timestamp = _optional_timestamp(row["timestamp"])
        source_type = str(row["source_type"])
        if timestamp is None and source_type != "era5":
            continue
        path = Path(str(row["path"])).expanduser()
        if not path.is_absolute():
            path = data_root / path
        record = Observation(
            observation_id=str(row["observation_id"]),
            storm_id=str(row["storm_id"]).upper(),
            split=str(row["split"]),
            source_type=source_type,
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


def _records_by_storm(
    records: Sequence[Observation],
    source_type: str,
) -> dict[str, list[Observation]]:
    by_storm: dict[str, list[Observation]] = defaultdict(list)
    for record in records:
        if record.source_type == source_type:
            by_storm[record.storm_id].append(record)
    return by_storm


def _nearest_storm_record(
    records_by_storm: Mapping[str, Sequence[Observation]],
    reference: Observation,
) -> Observation | None:
    records = records_by_storm.get(reference.storm_id, ())
    if not records:
        return None
    if len(records) == 1 or reference.timestamp is None:
        return records[0]
    timed_records = [record for record in records if record.timestamp is not None]
    if not timed_records:
        return records[0]
    return min(
        timed_records,
        key=lambda record: abs((record.timestamp - reference.timestamp).total_seconds()),
    )


def _audit_geo_channels(
    records: Sequence[Observation],
    geo_channels_by_sensor: Mapping[str, Sequence[str]] = GEO_CHANNELS,
) -> None:
    missing: dict[tuple[str, str], list[str]] = defaultdict(list)
    checked_files: set[str] = set()
    for record in records:
        if record.source_type != "geo" or record.sensor not in geo_channels_by_sensor:
            continue
        requested = set(geo_channels_by_sensor[record.sensor])
        available = set(record.variables)
        if not requested & available:
            if record.sensor in checked_files:
                continue
            available = _geo_dataset_channels(record)
            checked_files.add(record.sensor)
        for channel in geo_channels_by_sensor[record.sensor]:
            if channel not in available:
                missing[(record.sensor, channel)].append(record.observation_id)
    if not missing:
        return

    lines = ["GEO manifest is missing required common ABI/AHI channels:"]
    for (sensor, channel), observation_ids in sorted(missing.items()):
        examples = ", ".join(observation_ids[:5])
        suffix = "..." if len(observation_ids) > 5 else ""
        lines.append(
            f"- {sensor} {channel}: {len(observation_ids)} observations "
            f"(examples: {examples}{suffix})"
        )
    raise ValueError("\n".join(lines))


def _geo_dataset_channels(record: Observation) -> set[str]:
    try:
        with xr.open_dataset(record.path, engine="h5netcdf") as dataset:
            if "data" in dataset and "channel" in dataset["data"].dims:
                return {str(value) for value in dataset["data"]["channel"].values}
            return {str(name) for name in dataset.data_vars}
    except OSError as error:
        raise ValueError(
            f"Could not inspect GEO channels for {record.observation_id} "
            f"at {record.path}: {error}"
        ) from error


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
    era5: Observation | None = None,
    grid_size: int,
    grid_resolution: float,
    center: str,
    shift_center: bool,
    pad: int,
    geo_channels_by_sensor: Mapping[str, Sequence[str]] = GEO_CHANNELS,
    era5_channels: Sequence[str] = ERA5_CHANNELS,
    era5_max_time_gap_hours: float = DEFAULT_ERA5_MAX_TIME_GAP_HOURS,
) -> dict[str, Any]:
    center_point = _grid_center(sar, center, shift_center, grid_size, grid_resolution, pad)
    if center_point is None:
        raise ValueError("SAR observation has no usable grid center")
    grid_lat, grid_lon = _make_grid(center_point[0], center_point[1], grid_size, grid_resolution)

    geo_channels = geo_channels_by_sensor[geo.sensor]
    sar_channels = SAR_CHANNELS
    geo_fields = _load_geo_channels(geo, geo_channels)
    sar_fields = _load_sar_channels(sar, sar_channels)
    _require_channels("geo", geo_channels, geo_fields)
    _require_channels("sar", sar_channels, sar_fields)

    geo_regridded = [_regrid(*geo_fields[name], grid_lat, grid_lon) for name in geo_channels]
    era5_timestamp = None
    era5_array = None
    era5_masks = None
    era5_mask = None
    era5_band_names: tuple[str, ...] = ()
    exported_era5_channels: tuple[str, ...] = ()
    if era5 is not None:
        era5_fields, era5_timestamp = _load_era5_channels(
            era5,
            geo.timestamp,
            era5_channels,
            max_time_gap_hours=era5_max_time_gap_hours,
        )
        _require_channels("era5", era5_channels, era5_fields)
        era5_fields = _append_native_era5_derived_fields(era5_fields)
        exported_era5_channels = tuple(era5_fields)
        era5_regridded = [
            _regrid_continuous(*era5_fields[name], grid_lat, grid_lon)
            for name in exported_era5_channels
        ]
        era5_array = np.stack([values for values, _ in era5_regridded]).astype(np.float32)
        era5_masks = np.stack([mask for _, mask in era5_regridded])
        era5_mask = np.all(era5_masks & np.isfinite(era5_array), axis=0)
        if not era5_mask.any():
            raise ValueError("No valid ERA5 pixels after regridding")
        era5_band_names = tuple(
            f"era5_{channel}" for channel in exported_era5_channels
        )
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
        "era5": era5,
        "era5_array": era5_array,
        "era5_mask": era5_mask,
        "era5_band_masks": era5_masks,
        "era5_band_names": era5_band_names,
        "era5_timestamp": era5_timestamp,
        "era5_channels": exported_era5_channels if era5 is not None else (),
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


def _load_era5_channels(
    observation: Observation,
    timestamp: pd.Timestamp | None,
    channels: Sequence[str],
    max_time_gap_hours: float = DEFAULT_ERA5_MAX_TIME_GAP_HOURS,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], pd.Timestamp | None]:
    if timestamp is None:
        raise ValueError("Cannot select ERA5 time without a GEO timestamp")
    with xr.open_dataset(
        observation.path,
        group="rectilinear",
        engine="h5netcdf",
        decode_times=True,
    ) as source:
        dataset = source.load()
    missing = [channel for channel in channels if channel not in dataset]
    if missing:
        raise KeyError(f"ERA5 dataset is missing channels: {missing}")
    if "time" not in dataset:
        raise KeyError("ERA5 dataset is missing time coordinate")
    times = pd.DatetimeIndex(dataset["time"].values)
    if times.tz is None:
        times = times.tz_localize("UTC")
    else:
        times = times.tz_convert("UTC")
    era5_index = _nearest_time_index(times, timestamp)
    if era5_index is None:
        raise ValueError("ERA5 dataset has no time steps")
    era5_timestamp = pd.Timestamp(times[era5_index])
    if era5_timestamp.tzinfo is None:
        era5_timestamp = era5_timestamp.tz_localize("UTC")
    else:
        era5_timestamp = era5_timestamp.tz_convert("UTC")
    time_gap_hours = abs(
        (era5_timestamp - timestamp).total_seconds()
    ) / 3600.0
    if time_gap_hours > max_time_gap_hours:
        raise ValueError(
            "Nearest ERA5 time is too stale: "
            f"{time_gap_hours:.2f} h > {max_time_gap_hours:.2f} h"
        )

    lat = _era5_coordinate_values(dataset, "latitude", "lat", era5_index)
    lon = _fix_longitudes(
        _era5_coordinate_values(dataset, "longitude", "lon", era5_index)
    )
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    result = {}
    for channel in channels:
        values = dataset[channel].isel(time=era5_index)
        if "level" in values.dims:
            raise ValueError(
                f"ERA5 channel {channel} has a level dimension; choose single-level fields"
            )
        result[channel] = (np.asarray(values.values, dtype=np.float32), lat, lon)
    return result, era5_timestamp


def _era5_coordinate_values(
    dataset: xr.Dataset,
    primary: str,
    fallback: str,
    time_index: int,
) -> np.ndarray:
    name = primary if primary in dataset else fallback
    if name not in dataset:
        raise KeyError(f"ERA5 dataset is missing {primary!r}/{fallback!r}")
    values = dataset[name]
    if "time" in values.dims:
        values = values.isel(time=time_index)
    return np.asarray(values.values, dtype=float).squeeze()


def _append_native_era5_derived_fields(
    fields: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Derive wind diagnostics before any spatial upsampling."""
    result = dict(fields)
    if "u_wind_10m" not in fields or "v_wind_10m" not in fields:
        return result
    u10, lat, lon = fields["u_wind_10m"]
    v10, v_lat, v_lon = fields["v_wind_10m"]
    if u10.shape != v10.shape or lat.shape != v_lat.shape or lon.shape != v_lon.shape:
        raise ValueError("ERA5 10 m wind components do not share a grid")
    if "wind_speed_10m" not in result:
        result["wind_speed_10m"] = (
            np.hypot(u10, v10).astype(np.float32),
            lat,
            lon,
        )
    if "relative_vorticity_10m" not in result:
        result["relative_vorticity_10m"] = (
            _native_relative_vorticity_10m(u10, v10, lat, lon),
            lat,
            lon,
        )
    return result


def _native_relative_vorticity_10m(
    u10: np.ndarray,
    v10: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
) -> np.ndarray:
    """Compute spherical relative vorticity on the native rectilinear grid."""
    values, lat_axis, lon_axis = _rectilinear_field(u10, src_lat, src_lon)
    v_values, v_lat_axis, v_lon_axis = _rectilinear_field(v10, src_lat, src_lon)
    if not np.allclose(lat_axis, v_lat_axis) or not np.allclose(lon_axis, v_lon_axis):
        raise ValueError("ERA5 wind components do not share rectilinear coordinates")
    if values.shape[0] < 2 or values.shape[1] < 2:
        return np.full(values.shape, np.nan, dtype=np.float32)
    lat_rad = np.deg2rad(lat_axis)
    lon_rad = np.unwrap(np.deg2rad(lon_axis))
    coslat = np.clip(np.cos(lat_rad)[:, None], 1e-6, None)
    d_v_d_lambda = np.gradient(v_values, lon_rad, axis=1, edge_order=1)
    d_u_coslat_d_phi = np.gradient(
        values * coslat, lat_rad, axis=0, edge_order=1
    )
    vorticity = (
        d_v_d_lambda / (EARTH_RADIUS_M * coslat)
        - d_u_coslat_d_phi / (EARTH_RADIUS_M * coslat)
    )
    return vorticity.astype(np.float32)


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


def _regrid_continuous(
    values: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly interpolate a continuous rectilinear field and validity."""
    values, lat_axis, lon_axis = _rectilinear_field(values, src_lat, src_lon)
    target_center = float(np.nanmedian(grid_lon))
    lon_axis = _longitudes_near(lon_axis, target_center)
    query_lon = _longitudes_near(grid_lon, target_center)

    lat_order = np.argsort(lat_axis)
    lon_order = np.argsort(lon_axis)
    lat_axis = lat_axis[lat_order]
    lon_axis = lon_axis[lon_order]
    values = values[np.ix_(lat_order, lon_order)]
    lon_axis, unique_lon_indices = np.unique(lon_axis, return_index=True)
    values = values[:, unique_lon_indices]
    lat_axis, unique_lat_indices = np.unique(lat_axis, return_index=True)
    values = values[unique_lat_indices]
    if lat_axis.size == 0 or lon_axis.size == 0:
        return (
            np.full(grid_lat.shape, np.nan, dtype=np.float32),
            np.zeros(grid_lat.shape, dtype=bool),
        )

    finite = np.isfinite(values)
    method = "linear" if lat_axis.size >= 2 and lon_axis.size >= 2 else "nearest"
    numerator_interpolator = RegularGridInterpolator(
        (lat_axis, lon_axis),
        np.where(finite, values, 0.0),
        method=method,
        bounds_error=False,
        fill_value=0.0,
    )
    weight_interpolator = RegularGridInterpolator(
        (lat_axis, lon_axis),
        finite.astype(np.float64),
        method=method,
        bounds_error=False,
        fill_value=0.0,
    )
    query = np.column_stack([grid_lat.ravel(), query_lon.ravel()])
    numerator = numerator_interpolator(query)
    weight = weight_interpolator(query)
    query_finite = np.isfinite(query).all(axis=1)
    mask = query_finite & (weight > 0.999)
    result = np.full(query.shape[0], np.nan, dtype=np.float32)
    result[mask] = (numerator[mask] / weight[mask]).astype(np.float32)
    return result.reshape(grid_lat.shape), mask.reshape(grid_lat.shape)


def _rectilinear_field(
    values: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64).squeeze()
    src_lat = np.asarray(src_lat, dtype=np.float64).squeeze()
    src_lon = np.asarray(src_lon, dtype=np.float64).squeeze()
    if src_lat.ndim == 1 and src_lon.ndim == 1:
        lat_axis, lon_axis = src_lat, src_lon
    elif src_lat.ndim == 2 and src_lon.ndim == 2:
        lat_axis, lon_axis = src_lat[:, 0], src_lon[0, :]
        if not np.allclose(src_lat, lat_axis[:, None], equal_nan=True) or not np.allclose(
            src_lon, lon_axis[None, :], equal_nan=True
        ):
            raise ValueError("Continuous interpolation requires a rectilinear grid")
    else:
        raise ValueError("Continuous interpolation requires 1-D or 2-D coordinates")
    if values.ndim != 2 or values.shape != (lat_axis.size, lon_axis.size):
        raise ValueError("Continuous field shape does not match its coordinates")
    if not np.isfinite(lat_axis).all() or not np.isfinite(lon_axis).all():
        raise ValueError("Rectilinear coordinates contain non-finite values")
    return values, lat_axis, lon_axis


def _longitudes_near(longitudes: np.ndarray, center: float) -> np.ndarray:
    longitudes = np.asarray(longitudes, dtype=np.float64)
    return center + (longitudes - center + 180.0) % 360.0 - 180.0


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
        "era5_observation_id": _optional_observation_id(sample.get("era5")),
        "era5_timestamp": _optional_isoformat(sample.get("era5_timestamp")),
        "era5_channels": json.dumps(sample.get("era5_channels", ())),
        "center_lat": sample["center_lat"],
        "center_lon": sample["center_lon"],
        "ibtracs_center_lat": sar.ibtracs_center_lat,
        "ibtracs_center_lon": sar.ibtracs_center_lon,
    }


def _optional_observation_id(observation: Observation | None) -> str:
    return observation.observation_id if observation is not None else ""


def _optional_isoformat(timestamp: Any) -> str:
    return timestamp.isoformat() if timestamp is not None else ""


def _update_stats(stats: StatsAccumulator, sample: Mapping[str, Any]) -> None:
    for index, channel in enumerate(sample["geo_channels"]):
        stats.update("geo", channel, sample["geo"][index], sample["geo_band_masks"][index])
        stats.update("geo", f"band_{index}", sample["geo"][index], sample["geo_band_masks"][index])
    if sample.get("era5_array") is not None:
        for index, channel in enumerate(sample.get("era5_band_names", ())):
            stats.update("era5", channel, sample["era5_array"][index], sample["era5_band_masks"][index])
            stats.update("era5", f"band_{index}", sample["era5_array"][index], sample["era5_band_masks"][index])
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
