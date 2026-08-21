#!/usr/bin/env python3
"""Cache U-Net fields for the exact jointly labelled paired-data cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo2wf.config import instantiate_datamodule, load_config_file  # noqa: E402
from geo2wf.data.collation import collate_wind_field_samples  # noqa: E402
from geo2wf.data.datasets.paired_geotiff import (  # noqa: E402
    _normalized_distance_to_center,
)
from geo2wf.data.intensity import (  # noqa: E402
    INTENSITY_CACHE_SCHEMA_VERSION,
    KNOT_TO_MS,
    tropical_category_from_wind_ms,
)
from geo2wf.data.joint_intensity import (  # noqa: E402
    JointPairedIntensityDataModule,
)
from geo2wf.models.deterministic_residual import (  # noqa: E402
    ERA5ResidualRegressor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Resolved field-only U-Net training config using the joint data module.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--minimum-valid-fraction",
        type=float,
        default=0.0,
        help=(
            "Optional stricter valid-pixel threshold. The comparison default "
            "retains every sample having at least one valid pixel."
        ),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not 0.0 <= args.minimum_valid_fraction <= 1.0:
        parser.error("--minimum-valid-fraction must be in [0, 1]")
    if len(set(args.splits)) != len(args.splits):
        parser.error("--splits must not contain duplicates")
    return args


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            frame.to_csv(stream, index=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _cohort_fingerprint(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fingerprint the scalar target contract independently of row order."""
    columns = [
        "sample_id",
        "storm_id",
        "split",
        "target_timestamp",
        "target_wind_ms",
    ]
    frame = pd.DataFrame(rows, columns=columns).sort_values("sample_id")
    serialized = frame.to_csv(
        index=False, lineterminator="\n", float_format="%.9g"
    ).encode("utf-8")
    return {
        "samples": len(frame),
        "storms": int(frame["storm_id"].nunique()),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "columns": columns,
    }


def _fallback_basin(storm_id: str) -> str:
    prefix = str(storm_id).strip().upper()[:2]
    return {
        "AL": "NA",
        "EP": "EP",
        "CP": "EP",
        "WP": "WP",
        "IO": "NI",
        "BB": "NI",
        "AS": "NI",
        "SL": "SA",
    }.get(prefix, "OTHER")


def _storm_metadata(
    ibtracs_file: Path, storm_ids: set[str]
) -> dict[str, dict[str, Any]]:
    available = set(pd.read_csv(ibtracs_file, nrows=0).columns)
    required = {"USA_ATCF_ID", "ISO_TIME"}
    missing = required.difference(available)
    if missing:
        raise ValueError(f"{ibtracs_file} is missing columns: {sorted(missing)}")
    columns = sorted(required | ({"BASIN"} if "BASIN" in available else set()))
    frame = pd.read_csv(
        ibtracs_file, usecols=columns, keep_default_na=False, low_memory=False
    )
    frame["storm_id"] = frame["USA_ATCF_ID"].astype(str).str.strip().str.upper()
    frame["timestamp"] = pd.to_datetime(
        frame["ISO_TIME"], errors="coerce", utc=True, format="mixed"
    )
    frame = frame.loc[
        frame["storm_id"].isin(storm_ids) & frame["timestamp"].notna()
    ].copy()
    result: dict[str, dict[str, Any]] = {}
    for storm_id, fixes in frame.groupby("storm_id", sort=False):
        basin = ""
        if "BASIN" in fixes:
            values = fixes["BASIN"].astype(str).str.strip().str.upper()
            values = values.loc[~values.str.casefold().isin({"", "nan", "none"})]
            if not values.empty:
                basin = str(values.iloc[0])
        result[str(storm_id)] = {
            "start": pd.Timestamp(fixes["timestamp"].min()),
            "basin": basin or _fallback_basin(str(storm_id)),
        }
    absent = sorted(storm_ids.difference(result))
    if absent:
        raise ValueError(
            "IBTrACS has no valid timestamp metadata for storms: "
            + ", ".join(absent[:10])
        )
    return result


def _split_dataset(datamodule: JointPairedIntensityDataModule, split: str):
    if split not in {
        datamodule.train_split,
        datamodule.val_split,
        datamodule.test_split,
    }:
        raise ValueError(
            f"split {split!r} is not one of the configured train/val/test splits"
        )
    return datamodule._make_dataset(split, augment=False)


def export_joint_intensity_cache(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.config, args.checkpoint):
        if not Path(path).expanduser().is_file():
            raise FileNotFoundError(path)

    config = load_config_file(args.config)
    datamodule = instantiate_datamodule(config)
    if not isinstance(datamodule, JointPairedIntensityDataModule):
        raise TypeError(
            "cache export requires a resolved config using "
            "JointPairedIntensityDataModule"
        )
    if datamodule.include_test_in_train:
        raise ValueError("comparison export requires include_test_in_train=false")

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"output root must be absent or empty to avoid mixed caches: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    model = ERA5ResidualRegressor.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    ).eval()
    model.to(args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    split_datasets = {split: _split_dataset(datamodule, split) for split in args.splits}
    for dataset in split_datasets.values():
        model.validate_data_spec(dataset.data_spec)
    storm_ids = {
        str(storm_id).strip().upper()
        for dataset in split_datasets.values()
        for storm_id in dataset.samples["storm_id"]
    }
    storm_metadata = _storm_metadata(datamodule.ibtracs_file, storm_ids)

    batch_size = args.batch_size or datamodule.batch_size
    num_workers = (
        args.num_workers if args.num_workers is not None else datamodule.num_workers
    )
    all_rows: list[dict[str, Any]] = []
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    with torch.inference_mode():
        for split, dataset in split_datasets.items():
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=datamodule.pin_memory,
                persistent_workers=datamodule.persistent_workers and num_workers > 0,
                collate_fn=collate_wind_field_samples,
            )
            split_rows: list[dict[str, Any]] = []
            progress = tqdm(loader, desc=f"joint intensity cache {split}", unit="batch")
            for batch in progress:
                device_batch = {
                    key: value.to(args.device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                prediction = model.predict_physical(device_batch).detach().cpu()
                condition_mask = batch["condition_mask"].bool()
                for index, sample_id in enumerate(batch["sample_id"]):
                    sample_id = str(sample_id)
                    if (
                        not sample_id
                        or sample_id in {".", ".."}
                        or Path(sample_id).name != sample_id
                    ):
                        raise ValueError(
                            f"sample ID is not safe for a cache filename: {sample_id!r}"
                        )
                    field = prediction[index, 0]
                    valid = condition_mask[index, 0] & torch.isfinite(field)
                    center = batch["center"][index]
                    distance = _normalized_distance_to_center(
                        batch["condition_bounds"][index], field.shape, center
                    )[0]
                    valid &= torch.isfinite(distance)
                    if not valid.any():
                        raise ValueError(
                            f"sample {sample_id} has no finite valid pixels"
                        )
                    valid_fraction = float(valid.float().mean())
                    if valid_fraction < args.minimum_valid_fraction:
                        raise ValueError(
                            f"sample {sample_id} in {split} has valid fraction "
                            f"{valid_fraction:.6f}, below the comparison minimum "
                            f"{args.minimum_valid_fraction:.6f}; refusing to change "
                            "the common cohort"
                        )
                    field = torch.where(valid, field, torch.zeros_like(field))
                    distance = torch.nan_to_num(
                        distance, nan=0.0, posinf=1.0, neginf=0.0
                    ).clamp(0.0, 1.0)

                    metadata = batch["meta"][index]
                    storm_id = str(metadata["storm_id"]).strip().upper()
                    target_time = pd.Timestamp(
                        batch["intensity_observation_timestamp"][index]
                    )
                    target_time = (
                        target_time.tz_localize("UTC")
                        if target_time.tzinfo is None
                        else target_time.tz_convert("UTC")
                    )
                    target_wind_ms = float(batch["intensity_target_ms"][index])
                    target_category = tropical_category_from_wind_ms(target_wind_ms)
                    relative = Path(split) / "fields" / f"{sample_id}.npz"
                    _atomic_npz(
                        output_root / relative,
                        wind_speed_ms=field.numpy().astype(np.float32),
                        valid_mask=valid.numpy().astype(np.uint8),
                        distance_to_center=distance.numpy().astype(np.float32),
                    )
                    start = storm_metadata[storm_id]["start"]
                    split_rows.append(
                        {
                            "sample_id": sample_id,
                            "source_sample_id": sample_id,
                            "storm_id": storm_id,
                            "split": split,
                            "field_path": relative.as_posix(),
                            "observation_timestamp": target_time.isoformat(),
                            "target_timestamp": target_time.isoformat(),
                            "center_lat": float(center[0]),
                            "center_lon": float(center[1]),
                            "basin": storm_metadata[storm_id]["basin"],
                            "storm_elapsed_hours": max(
                                0.0, (target_time - start).total_seconds() / 3600.0
                            ),
                            "target_wind_ms": target_wind_ms,
                            "target_category": target_category,
                            "raw_unet_max_wind_ms": float(field[valid].max()),
                            "valid_fraction": float(valid.float().mean()),
                        }
                    )
            if not split_rows:
                raise RuntimeError(f"cache export produced no usable {split!r} samples")
            split_rows.sort(key=lambda row: str(row["sample_id"]))
            rows_by_split[split] = split_rows
            _atomic_csv(pd.DataFrame(split_rows), output_root / split / "manifest.csv")
            all_rows.extend(split_rows)

    sample_ids = [str(row["sample_id"]) for row in all_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs are not unique across exported splits")
    all_rows.sort(key=lambda row: (str(row["split"]), str(row["sample_id"])))
    _atomic_csv(pd.DataFrame(all_rows), output_root / "manifest.csv")
    stats_file = Path(config["data"]["stats_file"]).expanduser()
    source_manifests = {
        split: {
            "path": str((datamodule.root / split / "manifest.csv").resolve()),
            "sha256": _sha256(datamodule.root / split / "manifest.csv"),
        }
        for split in args.splits
    }
    metadata = {
        "schema_version": INTENSITY_CACHE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "single_timestep": True,
        "source_kind": "joint_paired_intensity_cohort",
        "target": {
            "source": "IBTrACS USA_WIND interpolated at paired SAR target time",
            "units": "m s-1",
            "knot_to_ms": KNOT_TO_MS,
            "max_bracket_hours": datamodule.max_ibtracs_bracket_hours,
        },
        "unet_checkpoint": {
            "path": str(args.checkpoint.expanduser().resolve()),
            "sha256": _sha256(args.checkpoint),
        },
        "unet_config": {
            "path": str(args.config.expanduser().resolve()),
            "sha256": _sha256(args.config),
            "include_test_in_train": False,
        },
        "normalization_stats": {
            "path": str(stats_file.resolve()),
            "sha256": _sha256(stats_file),
        },
        "source_manifests": source_manifests,
        "ibtracs": {
            "path": str(datamodule.ibtracs_file.resolve()),
            "sha256": _sha256(datamodule.ibtracs_file),
        },
        "scientific_evaluation": "storm_disjoint_candidate",
        "samples": len(all_rows),
        "storms": len({str(row["storm_id"]) for row in all_rows}),
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
        "cohort": _cohort_fingerprint(all_rows),
        "skipped_samples": 0,
    }
    _atomic_json(metadata, output_root / "cache-metadata.json")
    return metadata


def main() -> None:
    metadata = export_joint_intensity_cache(parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
