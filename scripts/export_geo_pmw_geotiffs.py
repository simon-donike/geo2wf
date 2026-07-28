from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
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
import xarray as xr
import yaml
from tqdm.auto import tqdm

from local_env import load_local_env

load_local_env()

from export_geo_sar_geotiffs import (
    DEFAULT_CLOSEST_MATCH_HOURS,
    DEFAULT_DATA_ROOT,
    DEFAULT_GRID_RESOLUTION,
    DEFAULT_GRID_SIZE,
    DEFAULT_MANIFEST_FILE,
    DEFAULT_SPLITS,
    GEO_CHANNELS,
    GEO_CHANNEL_SET,
    GEO_CHANNEL_SETS,
    Observation,
    StatsAccumulator,
    _fix_longitudes,
    _grid_center,
    _grid_transform,
    _audit_geo_channels,
    _geo_channels_by_sensor,
    _json_list,
    _load_geo_channels,
    _make_grid,
    _nearest_time_index,
    _optional_float,
    _optional_timestamp,
    _regrid,
    _require_channels,
    _safe_text,
    _write_geotiff,
    _write_manifest,
)

DEFAULT_OUTPUT_ROOT = Path(os.environ.get("GEO_PMW_OUTPUT_ROOT", "data/geotiff/geo_pmw"))
PMW_SOURCE_CHANNELS = {
    "AMSR2_GCOMW1": "TB_A89.0V",
    "GMI_GPM": "TB_89.0V",
    "SSMIS_F16": "TB_91.665V",
    "SSMIS_F17": "TB_91.665V",
    "SSMIS_F18": "TB_91.665V",
}
PMW_CHANNELS = {sensor: (channel,) for sensor, channel in PMW_SOURCE_CHANNELS.items()}
PMW_SWATHS = {
    "AMSR2_GCOMW1": "S5",
    "GMI_GPM": "S1",
    "SSMIS_F16": "S4",
    "SSMIS_F17": "S4",
    "SSMIS_F18": "S4",
}


@dataclass(frozen=True)
class ExportConfig:
    data_root: Path
    manifest_file: Path
    output_root: Path
    splits: tuple[str, ...]
    grid_size: int
    grid_resolution: float
    closest_match_hours: float
    center: str
    shift_center: bool
    pad: int
    limit: int | None
    pmw_sensors: tuple[str, ...]
    geo_channel_set: str


def main() -> None:
    config = _parse_args()
    export_geo_pmw_geotiffs(config)


def _parse_args() -> ExportConfig:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    known, _ = config_parser.parse_known_args()
    file_config = _load_export_config(known.config)
    data_root = Path(os.environ.get("TCD_DATA_ROOT", file_config.get("data_root", DEFAULT_DATA_ROOT)))
    manifest_file = Path(
        file_config.get("manifest_file", data_root / "index-files" / "observation_manifest_v5.csv")
    )
    if "TCD_DATA_ROOT" in os.environ:
        manifest_file = data_root / "index-files" / "observation_manifest_v5.csv"

    parser = argparse.ArgumentParser(
        description="Export PMW-anchored GEO/PMW pairs as raw-value GeoTIFFs."
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
        default=Path(
            os.environ.get(
                "GEO_PMW_OUTPUT_ROOT",
                file_config.get("output_root", DEFAULT_OUTPUT_ROOT),
            )
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(file_config.get("splits", DEFAULT_SPLITS)),
        choices=list(DEFAULT_SPLITS),
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=int(file_config.get("grid_size", DEFAULT_GRID_SIZE)),
    )
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=float(file_config.get("grid_resolution", DEFAULT_GRID_RESOLUTION)),
    )
    parser.add_argument(
        "--closest-match-hours",
        type=float,
        default=float(
            file_config.get("closest_match_hours", DEFAULT_CLOSEST_MATCH_HOURS)
        ),
    )
    parser.add_argument(
        "--geo-channel-set",
        choices=sorted(GEO_CHANNEL_SETS),
        default=str(file_config.get("geo_channel_set", GEO_CHANNEL_SET)),
        help="Named GEO band set to export.",
    )
    parser.add_argument(
        "--center",
        choices=["image_center", "ibtracs_center"],
        default=str(file_config.get("center", "image_center")),
    )
    parser.add_argument(
        "--shift-center",
        action=argparse.BooleanOptionalAction,
        default=bool(file_config.get("shift_center", True)),
        help="Shift image-centered grids just enough to include IBTrACS center.",
    )
    parser.add_argument("--pad", type=int, default=int(file_config.get("pad", 8)))
    parser.add_argument(
        "--limit",
        type=int,
        default=file_config.get("limit"),
        help="Maximum exported samples per split, useful for smoke tests.",
    )
    parser.add_argument(
        "--pmw-sensors",
        nargs="+",
        default=list(file_config.get("pmw_sensors", PMW_CHANNELS)),
        choices=sorted(PMW_CHANNELS),
        help="PMW imager sensors to include.",
    )
    args = parser.parse_args()
    return ExportConfig(
        data_root=args.data_root,
        manifest_file=args.manifest_file,
        output_root=args.output_root,
        splits=tuple(args.splits),
        grid_size=args.grid_size,
        grid_resolution=args.grid_resolution,
        closest_match_hours=args.closest_match_hours,
        center=args.center,
        shift_center=args.shift_center,
        pad=args.pad,
        limit=args.limit,
        pmw_sensors=tuple(args.pmw_sensors),
        geo_channel_set=args.geo_channel_set,
    )


def _load_export_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return dict(config.get("export", {}))


def export_geo_pmw_geotiffs(config: ExportConfig) -> None:
    data_root = config.data_root.expanduser().resolve()
    manifest_file = config.manifest_file.expanduser().resolve()
    output_root = config.output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    geo_channels_by_sensor = _geo_channels_by_sensor(config.geo_channel_set)
    records = _read_manifest(manifest_file, data_root, config.pmw_sensors)
    _audit_geo_channels(records, geo_channels_by_sensor)
    by_split = {split: [] for split in config.splits}
    skipped_rows: list[dict[str, Any]] = []
    train_stats = StatsAccumulator.create()

    for split in config.splits:
        split_records = [record for record in records if record.split == split]
        pairs = _pair_pmw_to_geo(split_records, config.closest_match_hours)
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        skipped = 0

        iterator = tqdm(pairs, desc=f"export {split}", unit="sample")
        for pmw, geo, dt_minutes in iterator:
            if config.limit is not None and len(rows) >= config.limit:
                break
            try:
                sample = _build_sample(
                    pmw=pmw,
                    geo=geo,
                    grid_size=config.grid_size,
                    grid_resolution=config.grid_resolution,
                    geo_channels_by_sensor=geo_channels_by_sensor,
                    center=config.center,
                    shift_center=config.shift_center,
                    pad=config.pad,
                )
            except (KeyError, OSError, ValueError) as error:
                skipped += 1
                skipped_rows.append(
                    {
                        "split": split,
                        "storm_id": pmw.storm_id,
                        "pmw_observation_id": pmw.observation_id,
                        "geo_observation_id": geo.observation_id,
                        "reason": str(error),
                    }
                )
                continue

            sample_id = _sample_id(pmw, geo)
            geo_path = split_dir / f"{sample_id}_geo.tif"
            pmw_path = split_dir / f"{sample_id}_pmw.tif"
            tags = _metadata_tags(
                sample_id=sample_id,
                pmw=pmw,
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
                pmw_path,
                sample["pmw"],
                sample["pmw_mask"],
                sample["pmw_channels"],
                sample["transform"],
                tags | {"role": "target", "source_type": "pmw"},
            )
            if split == "train":
                _update_stats(train_stats, "geo", sample["geo_channels"], sample["geo"], sample["geo_band_masks"])
                _update_stats(train_stats, "pmw", sample["pmw_channels"], sample["pmw"], sample["pmw_band_masks"])

            row = {
                "sample_id": sample_id,
                "split": split,
                "storm_id": pmw.storm_id,
                "condition_path": str(geo_path.relative_to(output_root)),
                "target_path": str(pmw_path.relative_to(output_root)),
                "condition_source_type": "geo",
                "target_source_type": "pmw",
                "condition_observation_id": geo.observation_id,
                "target_observation_id": pmw.observation_id,
                "condition_timestamp": geo.timestamp.isoformat(),
                "target_timestamp": pmw.timestamp.isoformat(),
                "condition_sensor": geo.sensor,
                "target_sensor": pmw.sensor,
                "condition_channels": json.dumps(sample["geo_channels"]),
                "target_channels": json.dumps(sample["pmw_channels"]),
                "geo_path": str(geo_path.relative_to(output_root)),
                "pmw_path": str(pmw_path.relative_to(output_root)),
                "geo_observation_id": geo.observation_id,
                "pmw_observation_id": pmw.observation_id,
                "geo_timestamp": geo.timestamp.isoformat(),
                "pmw_timestamp": pmw.timestamp.isoformat(),
                "dt_minutes": dt_minutes,
                "geo_sensor": geo.sensor,
                "pmw_sensor": pmw.sensor,
                "geo_channels": json.dumps(sample["geo_channels"]),
                "pmw_channels": json.dumps(sample["pmw_channels"]),
                "grid_size": config.grid_size,
                "grid_resolution": config.grid_resolution,
                "center_lat": sample["center_lat"],
                "center_lon": sample["center_lon"],
                "ibtracs_center_lat": pmw.ibtracs_center_lat,
                "ibtracs_center_lon": pmw.ibtracs_center_lon,
            }
            rows.append(row)
            by_split[split].append(row)

        _write_manifest(split_dir / "manifest.csv", rows)
        print(f"[{split}] wrote {len(rows)} samples to {split_dir} ({skipped} skipped)")

    _write_manifest(
        output_root / "manifest.csv",
        [row for split in config.splits for row in by_split[split]],
    )
    (output_root / "stats.json").write_text(
        json.dumps(train_stats.to_jsonable(), indent=2), encoding="utf-8"
    )
    if skipped_rows:
        _write_manifest(output_root / "skipped.csv", skipped_rows)


def _read_manifest(
    manifest_file: Path,
    data_root: Path,
    pmw_sensors: Sequence[str],
) -> list[Observation]:
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
    frame = frame[frame["source_type"].isin({"geo", "pmw"})]
    pmw_sensors = set(pmw_sensors)

    records: list[Observation] = []
    for _, row in frame.iterrows():
        timestamp = _optional_timestamp(row["timestamp"])
        if timestamp is None:
            continue
        path = Path(str(row["path"])).expanduser()
        if not path.is_absolute():
            path = data_root / path
        variables = tuple(_json_list(row["variables"]))
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
            variables=variables,
        )
        if record.source_type == "geo" and record.sensor not in GEO_CHANNELS:
            continue
        if record.source_type == "pmw":
            if record.sensor not in pmw_sensors:
                continue
            if PMW_SOURCE_CHANNELS[record.sensor] not in variables:
                continue
        records.append(record)
    return records


def _pair_pmw_to_geo(
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
        pmw_records = sorted(
            (record for record in storm_records if record.source_type == "pmw"),
            key=lambda record: record.timestamp,
        )
        if not geo_records:
            continue
        geo_times = pd.DatetimeIndex([record.timestamp for record in geo_records])
        for pmw in pmw_records:
            nearest_index = _nearest_time_index(geo_times, pmw.timestamp)
            if nearest_index is None:
                continue
            geo = geo_records[nearest_index]
            dt_minutes = (geo.timestamp - pmw.timestamp).total_seconds() / 60.0
            if abs(dt_minutes) <= max_minutes:
                pairs.append((pmw, geo, dt_minutes))
    return pairs


def _build_sample(
    *,
    pmw: Observation,
    geo: Observation,
    grid_size: int,
    grid_resolution: float,
    center: str,
    shift_center: bool,
    pad: int,
    geo_channels_by_sensor: Mapping[str, Sequence[str]] = GEO_CHANNELS,
) -> dict[str, Any]:
    center_point = _grid_center(pmw, center, shift_center, grid_size, grid_resolution, pad)
    if center_point is None:
        raise ValueError("PMW observation has no usable grid center")
    grid_lat, grid_lon = _make_grid(center_point[0], center_point[1], grid_size, grid_resolution)

    geo_channels = geo_channels_by_sensor[geo.sensor]
    pmw_channels = PMW_CHANNELS[pmw.sensor]
    geo_fields = _load_geo_channels(geo, geo_channels)
    pmw_fields = _load_pmw_channels(pmw, pmw_channels)
    _require_channels("geo", geo_channels, geo_fields)
    _require_channels("pmw", pmw_channels, pmw_fields)

    geo_regridded = [_regrid(*geo_fields[name], grid_lat, grid_lon) for name in geo_channels]
    pmw_regridded = [_regrid(*pmw_fields[name], grid_lat, grid_lon) for name in pmw_channels]
    geo_array = np.stack([values for values, _ in geo_regridded]).astype(np.float32)
    pmw_array = np.stack([values for values, _ in pmw_regridded]).astype(np.float32)
    geo_masks = np.stack([mask for _, mask in geo_regridded])
    pmw_masks = np.stack([mask for _, mask in pmw_regridded])
    geo_mask = np.all(geo_masks & np.isfinite(geo_array), axis=0)
    pmw_mask = np.all(pmw_masks & np.isfinite(pmw_array), axis=0)
    if not geo_mask.any():
        raise ValueError("No valid GEO pixels after regridding")
    if not pmw_mask.any():
        raise ValueError("No valid PMW pixels after regridding")

    transform = _grid_transform(center_point[0], center_point[1], grid_size, grid_resolution)
    return {
        "geo": geo_array,
        "pmw": pmw_array,
        "geo_mask": geo_mask,
        "pmw_mask": pmw_mask,
        "geo_band_masks": geo_masks,
        "pmw_band_masks": pmw_masks,
        "geo_channels": tuple(geo_channels),
        "pmw_channels": tuple(pmw_channels),
        "center_lat": center_point[0],
        "center_lon": center_point[1],
        "transform": transform,
    }


def _load_pmw_channels(
    observation: Observation,
    channels: Sequence[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    group = f"passive_microwave/{PMW_SWATHS[observation.sensor]}"
    with xr.open_dataset(
        observation.path,
        group=group,
        engine="h5netcdf",
        decode_times=False,
    ) as source:
        dataset = source.load()
    source_channel = PMW_SOURCE_CHANNELS[observation.sensor]
    if source_channel not in dataset:
        raise KeyError(f"PMW dataset is missing channel in {group}: {source_channel}")
    lat = _coordinate_values(dataset, "latitude", "lat")
    lon = _fix_longitudes(_coordinate_values(dataset, "longitude", "lon"))
    result = {}
    values = np.asarray(dataset[source_channel].values, dtype=np.float32).squeeze()
    if values.ndim != 2:
        raise ValueError(
            f"PMW channel {source_channel} in {observation.path} has shape {values.shape}"
        )
    for channel in channels:
        result[channel] = (values, lat, lon)
    return result


def _coordinate_values(dataset: xr.Dataset, primary: str, fallback: str) -> np.ndarray:
    name = primary if primary in dataset else fallback
    if name not in dataset:
        raise KeyError(f"Dataset is missing {primary!r}/{fallback!r}")
    values = np.asarray(dataset[name].values, dtype=float).squeeze()
    if values.ndim == 1:
        other_name = "longitude" if name in {"latitude", "lat"} else "latitude"
        other_fallback = "lon" if other_name == "longitude" else "lat"
        paired_name = other_name if other_name in dataset else other_fallback
        if paired_name not in dataset:
            raise KeyError(f"Dataset is missing paired coordinate {paired_name!r}")
        paired = np.asarray(dataset[paired_name].values, dtype=float).squeeze()
        if name in {"latitude", "lat"}:
            values, _ = np.meshgrid(values, paired, indexing="ij")
        else:
            _, values = np.meshgrid(paired, values, indexing="ij")
    return values


def _metadata_tags(
    *,
    sample_id: str,
    pmw: Observation,
    geo: Observation,
    dt_minutes: float,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "storm_id": pmw.storm_id,
        "split": pmw.split,
        "condition_source_type": "geo",
        "target_source_type": "pmw",
        "condition_observation_id": geo.observation_id,
        "target_observation_id": pmw.observation_id,
        "condition_timestamp": geo.timestamp.isoformat(),
        "target_timestamp": pmw.timestamp.isoformat(),
        "dt_minutes": dt_minutes,
        "condition_sensor": geo.sensor,
        "target_sensor": pmw.sensor,
        "condition_channels": json.dumps(sample["geo_channels"]),
        "target_channels": json.dumps(sample["pmw_channels"]),
        "geo_observation_id": geo.observation_id,
        "pmw_observation_id": pmw.observation_id,
        "geo_timestamp": geo.timestamp.isoformat(),
        "pmw_timestamp": pmw.timestamp.isoformat(),
        "geo_sensor": geo.sensor,
        "pmw_sensor": pmw.sensor,
        "geo_channels": json.dumps(sample["geo_channels"]),
        "pmw_channels": json.dumps(sample["pmw_channels"]),
        "center_lat": sample["center_lat"],
        "center_lon": sample["center_lon"],
        "ibtracs_center_lat": pmw.ibtracs_center_lat,
        "ibtracs_center_lon": pmw.ibtracs_center_lon,
    }


def _update_stats(
    stats: StatsAccumulator,
    source_type: str,
    channels: Sequence[str],
    array: np.ndarray,
    masks: np.ndarray,
) -> None:
    for index, channel in enumerate(channels):
        stats.update(source_type, channel, array[index], masks[index])
        stats.update(source_type, f"band_{index}", array[index], masks[index])


def _sample_id(pmw: Observation, geo: Observation) -> str:
    time_tag = pmw.timestamp.strftime("%Y%m%d%H%M%S")
    digest = hashlib.blake2s(
        f"{pmw.observation_id}|{geo.observation_id}".encode("utf-8"),
        digest_size=4,
    ).hexdigest()
    return f"{_safe_text(pmw.storm_id)}_pmw_geo_{time_tag}_{digest}"


if __name__ == "__main__":
    main()
