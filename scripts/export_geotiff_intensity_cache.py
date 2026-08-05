#!/usr/bin/env python3
"""Cache frozen U-Net fields from a paired-GeoTIFF dataset and IBTrACS."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo2wf.data.collation import collate_wind_field_samples  # noqa: E402
from geo2wf.data.datasets.paired_geotiff import (  # noqa: E402
    DISTANCE_TO_IBTRACS_CENTER,
    PairedImageDataset,
)
from geo2wf.data.intensity import (  # noqa: E402
    INTENSITY_CACHE_SCHEMA_VERSION,
    KNOT_TO_MS,
)
from geo2wf.models.deterministic_residual import ERA5ResidualRegressor  # noqa: E402
from scripts.export_unet_intensity_cache import (  # noqa: E402
    _safe_sample_id,
    _sha256,
    _write_csv_atomic,
    _write_npz_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geotiff-root", type=Path, required=True)
    parser.add_argument("--ibtracs-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--closest-fix-hours", type=float, default=0.5)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and workers cannot be negative")
    if args.closest_fix_hours <= 0:
        parser.error("--closest-fix-hours must be positive")
    if not 0.0 < args.minimum_valid_fraction <= 1.0:
        parser.error("--minimum-valid-fraction must be in (0, 1]")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def _read_storm_tracks(
    path: Path, storm_ids: set[str]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Timestamp]]:
    tracks = pd.read_csv(
        path,
        skiprows=[1],
        usecols=[
            "USA_ATCF_ID",
            "ISO_TIME",
            "BASIN",
            "USA_WIND",
            "USA_SSHS",
        ],
        keep_default_na=False,
        low_memory=False,
    )
    tracks["storm_id"] = tracks["USA_ATCF_ID"].astype(str).str.strip().str.upper()
    tracks["target_timestamp"] = pd.to_datetime(
        tracks["ISO_TIME"], errors="coerce", utc=True
    )
    tracks["target_wind_kt"] = pd.to_numeric(tracks["USA_WIND"], errors="coerce")
    tracks["target_category"] = pd.to_numeric(tracks["USA_SSHS"], errors="coerce")
    tracks["basin"] = tracks["BASIN"].astype(str).str.strip().str.upper()
    tracks = tracks.loc[
        tracks["storm_id"].isin(storm_ids) & tracks["target_timestamp"].notna()
    ].copy()
    starts = tracks.groupby("storm_id")["target_timestamp"].min().to_dict()
    missing = storm_ids.difference(starts)
    if missing:
        raise ValueError(f"IBTrACS has no records for storms: {sorted(missing)}")
    eligible = tracks.loc[
        tracks["target_wind_kt"].notna() & tracks["target_category"].between(-1, 5)
    ].copy()
    by_storm = {
        storm_id: frame.sort_values("target_timestamp").reset_index(drop=True)
        for storm_id, frame in eligible.groupby("storm_id")
    }
    return by_storm, starts


def _dataset_from_config(
    root: Path,
    split: str,
    stats_path: Path,
    config: dict[str, Any],
) -> PairedImageDataset:
    data = config.get("data", {})
    return PairedImageDataset(
        root=root,
        split=split,
        stats_file=stats_path,
        target_size=tuple(data.get("target_size", [256, 256])),
        center_crop_size=(
            tuple(data["center_crop_size"])
            if data.get("center_crop_size") is not None
            else None
        ),
        augment=False,
        require_era5=True,
        include_pmw=False,
        include_ibtracs=False,
        normalization=data.get("normalization"),
        target_normalization=data.get("target_normalization"),
        robust_clip=float(data.get("robust_clip", 4.0)),
        target_robust_clip=data.get("target_robust_clip"),
        max_era5_time_gap_hours=data.get("max_era5_time_gap_hours"),
    )


def _match_dataset_to_tracks(
    dataset: PairedImageDataset,
    tracks_by_storm: dict[str, pd.DataFrame],
    closest_fix_hours: float,
) -> PairedImageDataset:
    matches: list[dict[str, Any]] = []
    maximum_gap = pd.Timedelta(hours=closest_fix_hours)
    for index, row in dataset.samples.iterrows():
        center_lat = pd.to_numeric(row.get("ibtracs_center_lat"), errors="coerce")
        center_lon = pd.to_numeric(row.get("ibtracs_center_lon"), errors="coerce")
        if not np.isfinite(center_lat) or not np.isfinite(center_lon):
            continue
        storm_id = str(row["storm_id"]).strip().upper()
        fixes = tracks_by_storm.get(storm_id)
        if fixes is None or fixes.empty:
            continue
        observation_time = pd.Timestamp(row["condition_timestamp"])
        if observation_time.tzinfo is None:
            observation_time = observation_time.tz_localize("UTC")
        else:
            observation_time = observation_time.tz_convert("UTC")
        gaps = (fixes["target_timestamp"] - observation_time).abs()
        nearest_index = gaps.idxmin()
        gap = gaps.loc[nearest_index]
        if gap > maximum_gap:
            continue
        fix = fixes.loc[nearest_index]
        matches.append(
            {
                "source_index": index,
                "intensity_target_timestamp": fix["target_timestamp"],
                "intensity_target_wind_kt": float(fix["target_wind_kt"]),
                "intensity_target_category": int(fix["target_category"]),
                "intensity_basin": str(fix["basin"]),
                "intensity_gap_minutes": (
                    observation_time - fix["target_timestamp"]
                ).total_seconds()
                / 60.0,
            }
        )
    if not matches:
        raise ValueError(
            f"{dataset.manifest_file} has no eligible IBTrACS matches within "
            f"{closest_fix_hours} hours"
        )
    matched = pd.DataFrame(matches).set_index("source_index")
    dataset.samples = dataset.samples.loc[matched.index].copy()
    for column in matched.columns:
        dataset.samples[column] = matched[column]
    dataset.samples = dataset.samples.reset_index(drop=True)
    return dataset


def export_geotiff_intensity_cache(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.geotiff_root,
        args.ibtracs_file,
        args.config,
        args.checkpoint,
    ):
        if not path.expanduser().exists():
            raise FileNotFoundError(path)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    stats_path = args.stats or args.geotiff_root / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)

    base_datasets = {
        split: _dataset_from_config(args.geotiff_root, split, stats_path, config)
        for split in args.splits
    }
    split_storms = {
        split: set(dataset.samples["storm_id"].astype(str))
        for split, dataset in base_datasets.items()
    }
    for left_index, left in enumerate(args.splits):
        for right in args.splits[left_index + 1 :]:
            overlap = split_storms[left].intersection(split_storms[right])
            if overlap:
                raise ValueError(
                    f"GeoTIFF intensity splits share storms ({left}/{right}): "
                    f"{sorted(overlap)}"
                )
    all_storms = set().union(*split_storms.values())
    tracks_by_storm, storm_starts = _read_storm_tracks(args.ibtracs_file, all_storms)
    datasets = {
        split: _match_dataset_to_tracks(
            dataset, tracks_by_storm, args.closest_fix_hours
        )
        for split, dataset in base_datasets.items()
    }

    model = (
        ERA5ResidualRegressor.load_from_checkpoint(args.checkpoint, map_location="cpu")
        .eval()
        .to(args.device)
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output_root = args.output_root.expanduser().resolve()
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in args.splits
    }
    written = 0
    with torch.inference_mode():
        for split, dataset in datasets.items():
            rows_by_id = {
                str(row.sample_id): row
                for row in dataset.samples.itertuples(index=False)
            }
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=str(args.device).startswith("cuda"),
                persistent_workers=args.num_workers > 0,
                collate_fn=collate_wind_field_samples,
            )
            iterator = tqdm(loader, desc=f"intensity cache {split}", unit="batch")
            for batch in iterator:
                device_batch = {
                    key: value.to(args.device)
                    for key, value in batch.items()
                    if torch.is_tensor(value)
                }
                prediction = model.predict_physical(device_batch).squeeze(1).cpu()
                condition_mask = batch["condition_mask"].squeeze(1).cpu().bool()
                era5_mask = batch["era5_wind_speed_mask"].squeeze(1).cpu().bool()
                distance_indices = [
                    meta["condition_channels"].index(DISTANCE_TO_IBTRACS_CENTER)
                    for meta in batch["meta"]
                ]
                if any(index != distance_indices[0] for index in distance_indices):
                    raise ValueError(
                        "distance-to-center channel position varies within a batch"
                    )
                distance = batch["condition"][:, distance_indices[0]].cpu()
                valid = (
                    condition_mask
                    & era5_mask
                    & torch.isfinite(prediction)
                    & torch.isfinite(distance)
                )
                for index, source_sample_id in enumerate(batch["sample_id"]):
                    if args.limit is not None and written >= args.limit:
                        break
                    source_row = rows_by_id[str(source_sample_id)]
                    sample_valid = valid[index]
                    valid_fraction = float(sample_valid.float().mean())
                    if valid_fraction < args.minimum_valid_fraction:
                        continue
                    field = torch.where(
                        sample_valid,
                        prediction[index],
                        torch.zeros_like(prediction[index]),
                    )
                    normalized_distance = distance[index].clamp(0.0, 1.0)
                    target_time = pd.Timestamp(source_row.intensity_target_timestamp)
                    observation_time = pd.Timestamp(source_row.condition_timestamp)
                    if observation_time.tzinfo is None:
                        observation_time = observation_time.tz_localize("UTC")
                    else:
                        observation_time = observation_time.tz_convert("UTC")
                    storm_id = str(source_row.storm_id).strip().upper()
                    elapsed_hours = (
                        observation_time - storm_starts[storm_id]
                    ).total_seconds() / 3600.0
                    if elapsed_hours < -1.0e-6:
                        raise ValueError(
                            f"observation precedes first IBTrACS record: {source_sample_id}"
                        )
                    sample_id = _safe_sample_id(
                        storm_id, target_time, str(source_sample_id)
                    )
                    relative_path = Path(split) / "fields" / f"{sample_id}.npz"
                    _write_npz_atomic(
                        output_root / relative_path,
                        wind_speed_ms=field.numpy().astype(np.float32),
                        valid_mask=sample_valid.numpy().astype(np.uint8),
                        distance_to_center=normalized_distance.numpy().astype(
                            np.float32
                        ),
                    )
                    target_wind_kt = float(source_row.intensity_target_wind_kt)
                    rows_by_split[split].append(
                        {
                            "sample_id": sample_id,
                            "storm_id": storm_id,
                            "split": split,
                            "field_path": relative_path.as_posix(),
                            "source_sample_id": str(source_sample_id),
                            "observation_timestamp": observation_time.isoformat(),
                            "target_timestamp": target_time.isoformat(),
                            "target_gap_minutes": float(
                                source_row.intensity_gap_minutes
                            ),
                            "center_lat": float(source_row.ibtracs_center_lat),
                            "center_lon": float(source_row.ibtracs_center_lon),
                            "basin": str(source_row.intensity_basin),
                            "storm_elapsed_hours": max(0.0, elapsed_hours),
                            "target_wind_ms": target_wind_kt * KNOT_TO_MS,
                            "target_category": int(
                                source_row.intensity_target_category
                            ),
                            "raw_unet_max_wind_ms": float(field[sample_valid].max()),
                            "valid_fraction": valid_fraction,
                        }
                    )
                    written += 1
                if args.limit is not None and written >= args.limit:
                    break
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
        raise RuntimeError("geotiff intensity export produced no usable samples")
    _write_csv_atomic(pd.DataFrame(all_rows), output_root / "manifest.csv")

    include_test_in_train = bool(config.get("data", {}).get("include_test_in_train"))
    metadata = {
        "schema_version": INTENSITY_CACHE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "single_timestep": True,
        "source_kind": "paired_geotiff_geo_era5_only",
        "target": {
            "source": "IBTrACS USA_WIND",
            "units": "m s-1",
            "knot_to_ms": KNOT_TO_MS,
            "eligible_usa_sshs": list(range(-1, 6)),
            "closest_fix_hours": args.closest_fix_hours,
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
        "source_manifests": {
            split: {
                "path": str(dataset.manifest_file.resolve()),
                "sha256": _sha256(dataset.manifest_file),
            }
            for split, dataset in datasets.items()
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
    metadata = export_geotiff_intensity_cache(parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
