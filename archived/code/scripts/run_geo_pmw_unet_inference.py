#!/usr/bin/env python3
"""Predict dense near-89 GHz PMW images from native GEO+ERA5 observations."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from geo2wf.data.datasets.paired_geotiff import (  # noqa: E402
    _normalization_affine_parameters,
)
from geo2wf.data.features import (  # noqa: E402
    normalized_distance_to_center,
    solar_time_features,
)
from geo2wf.training import build_model, resolve_runtime_config  # noqa: E402
from scripts.export_geo_pmw_geotiffs import PMW_CANONICAL_CHANNEL  # noqa: E402
from scripts.export_geo_sar_geotiffs import (  # noqa: E402
    GEO_CHANNEL_SETS,
    _append_native_era5_derived_fields,
    _load_geo_channels,
    _make_grid,
    _read_manifest,
    _regrid,
    _regrid_continuous,
)
from scripts.render_native_storm_gif import pmw_panel  # noqa: E402
from scripts.run_dense_pmw_unet_inference import (  # noqa: E402
    dense_manifest,
    update_storm_table,
)
from scripts.run_storm_diffusion_inference import (  # noqa: E402
    GRID_RESOLUTION_DEGREES,
    GRID_SIZE,
    ROBUST_CLIP,
    _normalize,
)
from scripts.save_deterministic_baseline_fields import _era5_fields  # noqa: E402

DATA = ROOT / "inference" / "inf_data"
OUTPUT = ROOT / "inference" / "dense_pmw_unet"
CONFIG = ROOT / "logs" / "20260804-141337_modular" / "resolved-config.yaml"
CHECKPOINT = (
    ROOT
    / "logs"
    / "20260804-141337_modular"
    / "checkpoints"
    / "epoch=056-step=5529.ckpt"
)
STATS = ROOT / "data" / "geotiff" / "geo_pmw_near89_10bands_era5" / "stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storm", default="EP182023")
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATA / "index-files" / "observation_manifest_v6.csv",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--stats", type=Path, default=STATS)
    parser.add_argument(
        "--timeline-root",
        type=Path,
        default=None,
        help=(
            "Optional densified-PMW root whose timestamps define the output timeline; "
            "by default every native GEO observation is predicted."
        ),
    )
    parser.add_argument("--max-geo-gap-minutes", type=float, default=15.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _full_bounds(center_lat: float, center_lon: float) -> torch.Tensor:
    half_extent = GRID_SIZE * GRID_RESOLUTION_DEGREES / 2.0
    return torch.tensor(
        [
            center_lon - half_extent,
            center_lon + half_extent,
            center_lat - half_extent,
            center_lat + half_extent,
        ],
        dtype=torch.float64,
    )


def prepare_sample(geo, era5_dataset: xr.Dataset, stats: dict):
    """Reproduce the direct-PMW training condition at its native 256-pixel size."""
    if geo.center is None or geo.ibtracs_center is None:
        raise ValueError(f"{geo.observation_id} has no finite center")
    grid_lat, grid_lon = _make_grid(
        geo.center[0],
        geo.center[1],
        GRID_SIZE,
        GRID_RESOLUTION_DEGREES,
    )
    try:
        geo_channels = list(GEO_CHANNEL_SETS["common10"][geo.sensor.upper()])
    except KeyError as error:
        raise ValueError(f"Unsupported GEO sensor {geo.sensor!r}") from error
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
    geo_tensor = _normalize(
        torch.from_numpy(geo_array),
        "geo",
        geo_channels,
        stats,
        normalization="robust-zscore",
        robust_clip=ROBUST_CLIP,
    )
    era5_tensor = _normalize(
        torch.from_numpy(era5_array),
        "era5",
        [f"era5_{channel}" for channel in era5_channels],
        stats,
        normalization="robust-zscore",
        robust_clip=ROBUST_CLIP,
    )
    condition = torch.nan_to_num(torch.cat([geo_tensor, era5_tensor]))
    condition = condition * valid
    bounds = _full_bounds(*geo.center)
    distance = normalized_distance_to_center(
        bounds,
        (GRID_SIZE, GRID_SIZE),
        torch.tensor(geo.ibtracs_center),
    )
    solar = solar_time_features(
        bounds,
        (GRID_SIZE, GRID_SIZE),
        geo.timestamp,
    )
    condition = torch.cat([condition, distance, solar], dim=0)
    offset, scale = _normalization_affine_parameters(
        "pmw",
        [PMW_CANONICAL_CHANNEL],
        stats,
        normalization="min-max",
    )
    sample = {
        "condition": condition.unsqueeze(0),
        "condition_mask": valid.unsqueeze(0),
        "target_norm_offset": offset.unsqueeze(0),
        "target_norm_scale": scale.unsqueeze(0),
    }
    return sample, valid.squeeze(0), grid_lat, grid_lon


def _timeline(records, storm: str, timeline_root: Path | None):
    geos = sorted(
        [
            record
            for record in records
            if record.storm_id == storm
            and record.source_type == "geo"
            and record.timestamp is not None
        ],
        key=lambda record: record.timestamp,
    )
    if not geos:
        raise ValueError(f"No GEO observations found for {storm}")
    if timeline_root is None:
        return [
            {
                "observation_id": geo.observation_id,
                "timestamp": geo.timestamp.isoformat(),
                "parsed_time": geo.timestamp,
                "geo": geo,
                "geo_gap_minutes": 0.0,
            }
            for geo in geos
        ], None

    source = dense_manifest(timeline_root, storm)
    dense = pd.read_csv(source)
    dense["parsed_time"] = pd.to_datetime(dense.timestamp, utc=True)
    dense = dense.sort_values("parsed_time").reset_index(drop=True)
    timeline = []
    for row in dense.itertuples(index=False):
        geo = min(
            geos,
            key=lambda item: abs((item.timestamp - row.parsed_time).total_seconds()),
        )
        timeline.append(
            {
                "observation_id": row.observation_id,
                "timestamp": row.timestamp,
                "parsed_time": row.parsed_time,
                "geo": geo,
                "geo_gap_minutes": abs(
                    (geo.timestamp - row.parsed_time).total_seconds()
                )
                / 60.0,
            }
        )
    return timeline, source


def main() -> None:
    args = parse_args()
    storm = args.storm.upper()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    for path in (args.manifest, args.config, args.checkpoint, args.stats):
        if not path.is_file():
            raise FileNotFoundError(path)

    records = _read_manifest(args.manifest, args.data_root)
    timeline, timeline_source = _timeline(records, storm, args.timeline_root)
    if args.limit is not None:
        timeline = timeline[: args.limit]
    era5_record = next(
        record
        for record in records
        if record.storm_id == storm and record.source_type == "era5"
    )
    with xr.open_dataset(
        era5_record.path,
        group="rectilinear",
        engine="h5netcdf",
        decode_times=True,
    ) as source:
        era5 = source.load()

    config = resolve_runtime_config(yaml.safe_load(args.config.read_text()))
    stats = json.loads(args.stats.read_text())
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(args.device)

    field_dir = args.output_root / storm / "pmw-fields"
    image_dir = args.output_root / storm / "pmw-images"
    field_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []
    with torch.inference_mode():
        progress = tqdm(range(0, len(timeline), args.batch_size), desc="GEO to PMW")
        for start in progress:
            prepared = []
            for item in timeline[start : start + args.batch_size]:
                if item["geo_gap_minutes"] > args.max_geo_gap_minutes:
                    skipped.append(
                        {
                            "storm_id": storm,
                            "observation_id": item["observation_id"],
                            "timestamp": item["timestamp"],
                            "reason": "geo_gap",
                            "geo_gap_minutes": item["geo_gap_minutes"],
                        }
                    )
                    continue
                sample, valid, grid_lat, grid_lon = prepare_sample(
                    item["geo"], era5, stats
                )
                prepared.append((item, sample, valid, grid_lat, grid_lon))
            if not prepared:
                continue
            batch = {
                key: torch.cat([entry[1][key] for entry in prepared]).to(args.device)
                for key in prepared[0][1]
            }
            predictions = (
                model.predict_physical(batch).detach().float().cpu().numpy()[:, 0]
            )
            for field, (item, _, valid, grid_lat, grid_lon) in zip(
                predictions, prepared
            ):
                safe_id = item["observation_id"].replace(":", "_")
                npz_path = field_dir / f"{safe_id}.npz"
                png_path = image_dir / f"{safe_id}.png"
                source_center = np.asarray(item["geo"].ibtracs_center, dtype=np.float32)
                np.savez_compressed(
                    npz_path,
                    brightness_temperature_k=field.astype(np.float32),
                    valid_mask=valid.numpy().astype(np.uint8),
                    grid_lat=np.asarray(grid_lat, dtype=np.float32),
                    grid_lon=np.asarray(grid_lon, dtype=np.float32),
                    source_center=source_center,
                    timeline_observation_id=np.asarray(item["observation_id"]),
                    timeline_timestamp=np.asarray(item["timestamp"]),
                    geo_observation_id=np.asarray(item["geo"].observation_id),
                    geo_timestamp=np.asarray(item["geo"].timestamp.isoformat()),
                )
                pmw_panel(field, valid.numpy()).save(png_path)
                rows.append(
                    {
                        "storm_id": storm,
                        "observation_id": item["observation_id"],
                        "timestamp": item["timestamp"],
                        "geo_observation_id": item["geo"].observation_id,
                        "geo_timestamp": item["geo"].timestamp.isoformat(),
                        "geo_gap_minutes": item["geo_gap_minutes"],
                        "npz_path": str(npz_path.relative_to(args.output_root)),
                        "png_path": str(png_path.relative_to(args.output_root)),
                        "array": "brightness_temperature_k",
                        "shape": "x".join(map(str, field.shape)),
                        "dtype": "float32",
                    }
                )

    args.output_root.mkdir(parents=True, exist_ok=True)
    update_storm_table(args.output_root / "dense-pmw-unet-manifest.csv", storm, rows)
    update_storm_table(
        args.output_root / "dense-pmw-unet-skipped.csv",
        storm,
        skipped,
        columns=[
            "storm_id",
            "observation_id",
            "timestamp",
            "reason",
            "geo_gap_minutes",
        ],
    )
    run = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fields": len(rows),
        "skipped": len(skipped),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256(args.checkpoint),
        },
        "config": str(args.config.resolve()),
        "stats": str(args.stats.resolve()),
        "timeline_source": (
            str(timeline_source.resolve())
            if timeline_source is not None
            else "native_geo"
        ),
        "model_inputs": ["geostationary", "era5"],
        "prediction": PMW_CANONICAL_CHANNEL,
        "prediction_units": "K",
        "device": args.device,
    }
    metadata_path = args.output_root / "dense-pmw-unet-run-metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    runs = metadata.get("runs", {})
    runs[storm] = run
    metadata_path.write_text(
        json.dumps({"schema_version": 1, "runs": runs}, indent=2) + "\n"
    )
    print(f"Saved {len(rows)} predicted PMW images; skipped {len(skipped)}")


if __name__ == "__main__":
    main()
