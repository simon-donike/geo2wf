#!/usr/bin/env python3
"""Cache frozen U-Net fields matched to tropical IBTrACS USA_WIND fixes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from geo2wf.data.intensity import KNOT_TO_MS  # noqa: E402
from geo2wf.models.deterministic_residual import ERA5ResidualRegressor  # noqa: E402
from scripts.export_geo_sar_geotiffs import (  # noqa: E402
    ERA5_CHANNELS,
    _nearest_time_index,
    _read_ibtracs,
    _read_manifest,
)
from scripts.run_storm_unet_inference import _prepare_sample  # noqa: E402


DEFAULT_DATA_ROOT = Path("data")
DEFAULT_MANIFEST = Path("data/index-files/observation_manifest_v6.csv")
DEFAULT_IBTRACS = Path("data/IBTrACs/ibtracs.ALL.list.v04r01.csv")
DEFAULT_CONFIG = Path("configs/config_geo_sar_10bands_era5_residual.yaml")
DEFAULT_OUTPUT_ROOT = Path("data/unet_intensity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ibtracs-file", type=Path, default=DEFAULT_IBTRACS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--closest-geo-hours", type=float, default=0.5)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.05)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--storms", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.closest_geo_hours <= 0:
        parser.error("--closest-geo-hours must be positive")
    if not 0.0 < args.minimum_valid_fraction <= 1.0:
        parser.error("--minimum-valid-fraction must be in (0, 1]")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _optional_number(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_sample_id(
    storm_id: str, target_time: pd.Timestamp, observation_id: str
) -> str:
    digest = hashlib.sha1(observation_id.encode("utf-8")).hexdigest()[:10]
    timestamp = target_time.strftime("%Y%m%dT%H%M%SZ")
    safe_storm = re.sub(r"[^A-Za-z0-9_-]+", "_", storm_id)
    return f"{safe_storm}_intensity_{timestamp}_{digest}"


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _storm_split_audit(records, selected_splits: set[str]) -> dict[str, str]:
    splits_by_storm: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.source_type == "geo" and record.split in selected_splits:
            splits_by_storm[record.storm_id].add(record.split)
    conflicts = {
        storm: splits for storm, splits in splits_by_storm.items() if len(splits) > 1
    }
    if conflicts:
        detail = ", ".join(
            f"{storm}: {sorted(splits)}" for storm, splits in sorted(conflicts.items())
        )
        raise ValueError(f"source manifest is not storm-disjoint: {detail}")
    return {storm: next(iter(splits)) for storm, splits in splits_by_storm.items()}


def _eligible_fixes(records: pd.DataFrame) -> pd.DataFrame:
    frame = records.copy()
    frame["_target_timestamp"] = pd.to_datetime(
        frame["_ibtracs_timestamp"], errors="coerce", utc=True
    )
    frame["_usa_wind"] = pd.to_numeric(frame.get("USA_WIND"), errors="coerce")
    frame["_usa_sshs"] = pd.to_numeric(frame.get("USA_SSHS"), errors="coerce")
    keep = (
        frame["_target_timestamp"].notna()
        & frame["_usa_wind"].notna()
        & frame["_usa_sshs"].between(-1, 5)
    )
    return frame.loc[keep].sort_values("_target_timestamp").reset_index(drop=True)


def _load_era5(record) -> xr.Dataset:
    with xr.open_dataset(
        record.path, group="rectilinear", engine="h5netcdf", decode_times=True
    ) as source:
        missing = set(ERA5_CHANNELS).difference(source.variables)
        if missing:
            raise ValueError(
                f"{record.path} is missing ERA5 variables: {sorted(missing)}"
            )
        return source[list(ERA5_CHANNELS)].load()


def export_intensity_cache(args: argparse.Namespace) -> dict[str, Any]:
    paths = [args.manifest, args.ibtracs_file, args.config, args.checkpoint]
    for path in paths:
        if not Path(path).expanduser().is_file():
            raise FileNotFoundError(path)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    stats_path = args.stats or Path(config["data"]["stats_file"])
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    records = _read_manifest(args.manifest, args.data_root)
    selected_splits = set(args.splits)
    storm_splits = _storm_split_audit(records, selected_splits)
    requested_storms = (
        {str(storm).strip().upper() for storm in args.storms}
        if args.storms
        else set(storm_splits)
    )
    unknown = requested_storms.difference(storm_splits)
    if unknown:
        raise ValueError(f"storms are absent from selected splits: {sorted(unknown)}")
    ibtracs_by_storm, _ = _read_ibtracs(args.ibtracs_file, storm_ids=requested_storms)
    geo_by_storm = {
        storm: sorted(
            (
                record
                for record in records
                if record.storm_id == storm
                and record.source_type == "geo"
                and record.split in selected_splits
                and record.center is not None
                and record.ibtracs_center is not None
            ),
            key=lambda record: record.timestamp,
        )
        for storm in requested_storms
    }
    era5_by_storm = {
        storm: next(
            (
                record
                for record in records
                if record.storm_id == storm and record.source_type == "era5"
            ),
            None,
        )
        for storm in requested_storms
    }

    model = (
        ERA5ResidualRegressor.load_from_checkpoint(args.checkpoint, map_location="cpu")
        .eval()
        .to(args.device)
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in args.splits
    }
    skipped: list[dict[str, Any]] = []
    written = 0
    with torch.inference_mode():
        for storm_id in sorted(requested_storms):
            fixes = ibtracs_by_storm.get(storm_id)
            geo_records = geo_by_storm.get(storm_id, [])
            era5_record = era5_by_storm.get(storm_id)
            if fixes is None or fixes.empty or not geo_records or era5_record is None:
                skipped.append(
                    {
                        "storm_id": storm_id,
                        "reason": "missing eligible IBTrACS, GEO, or ERA5 records",
                    }
                )
                continue
            eligible = _eligible_fixes(fixes)
            if eligible.empty:
                skipped.append(
                    {"storm_id": storm_id, "reason": "no tropical USA_WIND fixes"}
                )
                continue
            storm_start = pd.Timestamp(fixes["_ibtracs_timestamp"].min())
            geo_times = pd.DatetimeIndex([record.timestamp for record in geo_records])
            try:
                era5_dataset = _load_era5(era5_record)
            except (OSError, ValueError) as error:
                skipped.append({"storm_id": storm_id, "reason": str(error)})
                continue
            iterator = tqdm(
                eligible.iterrows(),
                total=len(eligible),
                desc=f"intensity {storm_id}",
                unit="fix",
            )
            for _, fix in iterator:
                if args.limit is not None and written >= args.limit:
                    break
                fix_data = fix.to_dict()
                target_time = pd.Timestamp(fix_data["_target_timestamp"])
                nearest_index = _nearest_time_index(geo_times, target_time)
                if nearest_index is None:
                    continue
                geo = geo_records[nearest_index]
                gap_minutes = (geo.timestamp - target_time).total_seconds() / 60.0
                if abs(gap_minutes) > args.closest_geo_hours * 60.0:
                    skipped.append(
                        {
                            "storm_id": storm_id,
                            "target_timestamp": target_time.isoformat(),
                            "reason": f"nearest GEO gap is {gap_minutes:.1f} minutes",
                        }
                    )
                    continue
                try:
                    batch, distance_km = _prepare_sample(geo, era5_dataset, stats)
                    if batch["condition"].shape[1] != model.condition_channels:
                        raise ValueError(
                            "prepared U-Net condition width does not match checkpoint: "
                            f"{batch['condition'].shape[1]} vs {model.condition_channels}"
                        )
                    device_batch = {
                        key: value.to(args.device) for key, value in batch.items()
                    }
                    prediction = model.predict_physical(device_batch).squeeze().cpu()
                    valid = batch["condition_mask"].squeeze().cpu().bool()
                except (KeyError, OSError, RuntimeError, ValueError) as error:
                    skipped.append(
                        {
                            "storm_id": storm_id,
                            "target_timestamp": target_time.isoformat(),
                            "observation_id": geo.observation_id,
                            "reason": str(error),
                        }
                    )
                    continue
                valid = valid & torch.isfinite(prediction) & torch.isfinite(distance_km)
                valid_fraction = float(valid.float().mean())
                if valid_fraction < args.minimum_valid_fraction:
                    skipped.append(
                        {
                            "storm_id": storm_id,
                            "target_timestamp": target_time.isoformat(),
                            "observation_id": geo.observation_id,
                            "reason": f"valid fraction {valid_fraction:.4f} is too small",
                        }
                    )
                    continue
                prediction = torch.where(
                    valid, prediction, torch.zeros_like(prediction)
                )
                finite_distance = distance_km[torch.isfinite(distance_km)]
                distance_scale = float(finite_distance.max())
                if not math.isfinite(distance_scale) or distance_scale <= 0:
                    raise ValueError(f"invalid distance grid for {geo.observation_id}")
                normalized_distance = (distance_km / distance_scale).clamp(0.0, 1.0)
                raw_max = float(prediction[valid].max())
                target_wind_kt = float(fix_data["_usa_wind"])
                target_category = int(fix_data["_usa_sshs"])
                split = storm_splits[storm_id]
                sample_id = _safe_sample_id(storm_id, target_time, geo.observation_id)
                relative_path = Path(split) / "fields" / f"{sample_id}.npz"
                _write_npz_atomic(
                    output_root / relative_path,
                    wind_speed_ms=prediction.numpy().astype(np.float32),
                    valid_mask=valid.numpy().astype(np.uint8),
                    distance_to_center=normalized_distance.numpy().astype(np.float32),
                )
                center_lat, center_lon = geo.ibtracs_center
                rows_by_split[split].append(
                    {
                        "sample_id": sample_id,
                        "storm_id": storm_id,
                        "split": split,
                        "field_path": relative_path.as_posix(),
                        "observation_id": geo.observation_id,
                        "observation_timestamp": geo.timestamp.isoformat(),
                        "target_timestamp": target_time.isoformat(),
                        "geo_dt_minutes": gap_minutes,
                        "center_lat": center_lat,
                        "center_lon": center_lon,
                        "basin": str(fix_data.get("BASIN", "")).strip().upper(),
                        "storm_elapsed_hours": (
                            geo.timestamp - storm_start
                        ).total_seconds()
                        / 3600.0,
                        "target_wind_ms": target_wind_kt * KNOT_TO_MS,
                        "target_category": target_category,
                        "raw_unet_max_wind_ms": raw_max,
                        "valid_fraction": valid_fraction,
                    }
                )
                written += 1
            if args.limit is not None and written >= args.limit:
                break

    all_rows = []
    for split, rows in rows_by_split.items():
        if rows:
            frame = pd.DataFrame(rows).sort_values(
                ["storm_id", "target_timestamp", "sample_id"]
            )
            _write_csv_atomic(frame, output_root / split / "manifest.csv")
            all_rows.extend(frame.to_dict("records"))
    if not all_rows:
        raise RuntimeError("intensity export produced no usable samples")
    _write_csv_atomic(pd.DataFrame(all_rows), output_root / "manifest.csv")
    if skipped:
        _write_csv_atomic(pd.DataFrame(skipped), output_root / "skipped.csv")

    include_test_in_train = bool(config.get("data", {}).get("include_test_in_train"))
    metadata = {
        # This exporter retains the legacy IBTrACS/max-anchor contract. Only
        # export_joint_intensity_cache writes the matched dual-reference v2.
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "single_timestep": True,
        "target": {
            "source": "IBTrACS USA_WIND",
            "units": "m s-1",
            "knot_to_ms": KNOT_TO_MS,
            "eligible_usa_sshs": list(range(-1, 6)),
        },
        "unet_checkpoint": {
            "path": str(args.checkpoint.expanduser().resolve()),
            "sha256": _sha256(args.checkpoint),
        },
        "unet_config": {
            "path": str(args.config.expanduser().resolve()),
            "sha256": _sha256(args.config),
            "include_test_in_train": include_test_in_train,
        },
        "normalization_stats": {
            "path": str(stats_path.expanduser().resolve()),
            "sha256": _sha256(stats_path),
        },
        "source_manifest": {
            "path": str(args.manifest.expanduser().resolve()),
            "sha256": _sha256(args.manifest),
        },
        "ibtracs": {
            "path": str(args.ibtracs_file.expanduser().resolve()),
            "sha256": _sha256(args.ibtracs_file),
        },
        "scientific_evaluation": (
            "development_only_upstream_test_was_trained"
            if include_test_in_train
            else "storm_disjoint_candidate"
        ),
        "samples": len(all_rows),
        "storms": len({row["storm_id"] for row in all_rows}),
        "splits": {split: len(rows) for split, rows in rows_by_split.items() if rows},
    }
    metadata_path = output_root / "cache-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    metadata = export_intensity_cache(parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
