#!/usr/bin/env python3
"""Export scalar IBTrACS and UNet+MLP 6-hour intensity forecast windows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo2wf.data.intensity import KNOT_TO_MS, UNetIntensityDataset  # noqa: E402
from geo2wf.data.intensity_forecast import (  # noqa: E402
    FORECAST_CACHE_SCHEMA_VERSION,
    FORECAST_FEATURE_NAMES,
    forecast_features,
)
from geo2wf.models.intensity_correction import UNetIntensityCorrection  # noqa: E402
from scripts.export_unet_intensity_cache import (  # noqa: E402
    _sha256,
    _write_csv_atomic,
)


RI_STORM_IDS = ("WP282025", "WP112024", "AL092024")
RI_THRESHOLD_KT = 30.0
RI_WINDOW_HOURS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibtracs-file", type=Path, required=True)
    parser.add_argument("--intensity-cache-root", type=Path, required=True)
    parser.add_argument("--intensity-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pretrain-start-year", type=int, default=2000)
    parser.add_argument("--pretrain-train-end-year", type=int, default=2018)
    parser.add_argument("--pretrain-val-end-year", type=int, default=2022)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if not (
        args.pretrain_start_year
        <= args.pretrain_train_end_year
        < args.pretrain_val_end_year
    ):
        parser.error("pretraining year boundaries must be strictly ordered")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    return args


def read_ibtracs_forecast_tracks(path: str | Path) -> pd.DataFrame:
    """Read scalar forecast fields while skipping the IBTrACS units row."""

    path = Path(path).expanduser()
    frame = pd.read_csv(
        path,
        skiprows=[1],
        usecols=["SEASON", "USA_ATCF_ID", "ISO_TIME", "USA_WIND", "USA_SSHS"],
        keep_default_na=False,
        low_memory=False,
    )
    frame["storm_id"] = frame["USA_ATCF_ID"].astype(str).str.strip().str.upper()
    frame["timestamp"] = pd.to_datetime(frame["ISO_TIME"], errors="coerce", utc=True)
    frame["wind_kt"] = pd.to_numeric(frame["USA_WIND"], errors="coerce")
    frame["category"] = pd.to_numeric(frame["USA_SSHS"], errors="coerce")
    frame["season"] = pd.to_numeric(frame["SEASON"], errors="coerce")
    frame = frame.loc[(frame["storm_id"] != "") & frame["timestamp"].notna()].copy()
    return (
        frame.sort_values(["storm_id", "timestamp"], kind="stable")
        .drop_duplicates(["storm_id", "timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _track_lookup(
    tracks: pd.DataFrame,
) -> dict[tuple[str, pd.Timestamp], tuple[float, float]]:
    return {
        (str(row.storm_id), pd.Timestamp(row.timestamp)): (
            float(row.wind_kt),
            float(row.category),
        )
        for row in tracks.itertuples(index=False)
        if np.isfinite(row.wind_kt) and np.isfinite(row.category)
    }


def _window(
    lookup: dict[tuple[str, pd.Timestamp], tuple[float, float]],
    storm_id: str,
    init_time: pd.Timestamp,
) -> dict[str, float] | None:
    values = {}
    for label, offset in (
        ("current", 0),
        ("minus_6h", -6),
        ("minus_12h", -12),
        ("plus_6h", 6),
    ):
        item = lookup.get((storm_id, init_time + pd.Timedelta(hours=offset)))
        if item is None:
            return None
        wind, category = item
        values[f"{label}_wind_kt"] = wind
        values[f"{label}_category"] = category
    if not -1 <= values["current_category"] <= 5:
        return None
    if not -1 <= values["plus_6h_category"] <= 5:
        return None
    plus_12 = lookup.get((storm_id, init_time + pd.Timedelta(hours=12)))
    values["plus_12h_wind_kt"] = float(plus_12[0]) if plus_12 else np.nan
    values["plus_12h_category"] = float(plus_12[1]) if plus_12 else np.nan
    return values


def build_historical_rows(
    tracks: pd.DataFrame,
    *,
    start_year: int,
    train_end_year: int,
    val_end_year: int,
) -> dict[str, list[dict[str, Any]]]:
    lookup = _track_lookup(tracks)
    rows = {"pretrain_train": [], "pretrain_val": []}
    for record in tracks.itertuples(index=False):
        timestamp = pd.Timestamp(record.timestamp)
        year = int(record.season) if np.isfinite(record.season) else timestamp.year
        if year < start_year or year > val_end_year:
            continue
        window = _window(lookup, str(record.storm_id), timestamp)
        if window is None:
            continue
        split = "pretrain_train" if year <= train_end_year else "pretrain_val"
        anchor = window["current_wind_kt"] * KNOT_TO_MS
        target = window["plus_6h_wind_kt"] * KNOT_TO_MS
        rows[split].append(
            {
                "sample_id": f"{record.storm_id}_{timestamp:%Y%m%dT%H%M%SZ}",
                "storm_id": str(record.storm_id),
                "split": split,
                "init_timestamp": timestamp.isoformat(),
                "anchor_wind_ms": anchor,
                "current_ibtracs_wind_ms": window["current_wind_kt"] * KNOT_TO_MS,
                "wind_minus_6h_ms": window["minus_6h_wind_kt"] * KNOT_TO_MS,
                "wind_minus_12h_ms": window["minus_12h_wind_kt"] * KNOT_TO_MS,
                "target_wind_ms": target,
                "target_delta_ms": target - anchor,
                "target_plus_12h_wind_ms": (
                    window["plus_12h_wind_kt"] * KNOT_TO_MS
                    if np.isfinite(window["plus_12h_wind_kt"])
                    else np.nan
                ),
                "source_kind": "ibtracs_history",
            }
        )
    return rows


def _current_intensity_predictions(
    cache_root: Path,
    checkpoint: Path,
    splits: Iterable[str],
    *,
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict[str, float]:
    model = (
        UNetIntensityCorrection.load_from_checkpoint(checkpoint, map_location="cpu")
        .eval()
        .to(device)
    )
    predictions: dict[str, float] = {}
    with torch.inference_mode():
        for split in splits:
            dataset = UNetIntensityDataset(cache_root, split)
            model.validate_data_spec(dataset.data_spec)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            for batch in loader:
                tensor_batch = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                output = model.predict_intensity(tensor_batch).output_msw_ms.cpu()
                predictions.update(
                    {
                        str(sample_id): float(value)
                        for sample_id, value in zip(batch["sample_id"], output)
                    }
                )
    return predictions


def _deduplicate_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.copy()
    selected["_abs_time_offset"] = pd.to_numeric(
        selected.get("target_gap_minutes"), errors="coerce"
    ).abs()
    selected["_valid_fraction"] = pd.to_numeric(
        selected.get("valid_fraction"), errors="coerce"
    ).fillna(-np.inf)
    selected["_init"] = pd.to_datetime(
        selected["target_timestamp"], errors="coerce", utc=True
    )
    selected = selected.sort_values(
        ["storm_id", "_init", "_abs_time_offset", "_valid_fraction", "sample_id"],
        ascending=[True, True, True, False, True],
        kind="stable",
    )
    return selected.drop_duplicates(["storm_id", "_init"], keep="first")


def build_matched_rows(
    cache_root: str | Path,
    tracks: pd.DataFrame,
    current_predictions: dict[str, float],
    splits: Iterable[str] = ("train", "val", "test"),
) -> dict[str, list[dict[str, Any]]]:
    cache_root = Path(cache_root)
    lookup = _track_lookup(tracks)
    result = {str(split): [] for split in splits}
    storm_owners: dict[str, str] = {}
    for split in splits:
        manifest = pd.read_csv(
            cache_root / split / "manifest.csv", keep_default_na=False
        )
        for storm_id in manifest["storm_id"].astype(str).unique():
            previous = storm_owners.setdefault(storm_id, str(split))
            if previous != split:
                raise ValueError(
                    f"matched intensity storm {storm_id} occurs in {previous} and {split}"
                )
        for _, row in _deduplicate_manifest(manifest).iterrows():
            timestamp = pd.Timestamp(row["_init"])
            window = _window(lookup, str(row["storm_id"]), timestamp)
            anchor = current_predictions.get(str(row["sample_id"]))
            if window is None or anchor is None or not np.isfinite(anchor):
                continue
            target = window["plus_6h_wind_kt"] * KNOT_TO_MS
            result[str(split)].append(
                {
                    "sample_id": str(row["sample_id"]),
                    "storm_id": str(row["storm_id"]),
                    "split": str(split),
                    "init_timestamp": timestamp.isoformat(),
                    "anchor_wind_ms": anchor,
                    "current_ibtracs_wind_ms": window["current_wind_kt"] * KNOT_TO_MS,
                    "wind_minus_6h_ms": window["minus_6h_wind_kt"] * KNOT_TO_MS,
                    "wind_minus_12h_ms": window["minus_12h_wind_kt"] * KNOT_TO_MS,
                    "target_wind_ms": target,
                    "target_delta_ms": target - anchor,
                    "target_plus_12h_wind_ms": (
                        window["plus_12h_wind_kt"] * KNOT_TO_MS
                        if np.isfinite(window["plus_12h_wind_kt"])
                        else np.nan
                    ),
                    "source_kind": "unet_mlp_anchor",
                }
            )
    return result


def earliest_ri_onset(
    tracks: pd.DataFrame, storm_id: str, threshold_kt: float = RI_THRESHOLD_KT
) -> pd.Timestamp:
    storm = tracks.loc[tracks["storm_id"] == storm_id]
    lookup = {
        pd.Timestamp(row.timestamp): float(row.wind_kt)
        for row in storm.itertuples(index=False)
        if np.isfinite(row.wind_kt)
    }
    for finish in sorted(lookup):
        onset = finish - pd.Timedelta(hours=RI_WINDOW_HOURS)
        if onset in lookup and lookup[finish] - lookup[onset] >= threshold_kt:
            return onset
    raise ValueError(f"{storm_id} has no exact >= {threshold_kt:g} kt/24 h RI window")


def select_ri_validation_cases(
    matched_val_rows: list[dict[str, Any]],
    tracks: pd.DataFrame,
    storm_ids: Iterable[str] = RI_STORM_IDS,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(matched_val_rows)
    frame["_init"] = pd.to_datetime(frame["init_timestamp"], utc=True)
    cases = []
    for storm_id in storm_ids:
        onset = earliest_ri_onset(tracks, storm_id)
        storm_rows = frame.loc[frame["storm_id"] == storm_id].sort_values("_init")
        target_12_values = pd.to_numeric(
            storm_rows["target_plus_12h_wind_ms"], errors="coerce"
        )
        storm_rows = storm_rows.loc[np.isfinite(target_12_values)]
        candidates = storm_rows.loc[storm_rows["_init"] <= onset]
        if candidates.empty:
            candidates = storm_rows.loc[storm_rows["_init"] > onset]
            if candidates.empty:
                raise ValueError(f"{storm_id} has no matched validation row")
            row = candidates.iloc[0]
            initialization_selection = "earliest_after_ri_onset"
        else:
            row = candidates.iloc[-1]
            initialization_selection = "latest_at_or_before_ri_onset"
        target_12 = pd.to_numeric(row["target_plus_12h_wind_ms"], errors="coerce")
        if not np.isfinite(target_12):
            raise ValueError(f"{storm_id} RI row has no +12 h wind target")
        cases.append(
            {
                "storm_id": storm_id,
                "sample_id": str(row["sample_id"]),
                "init_timestamp": row["_init"].isoformat(),
                "ri_onset_timestamp": onset.isoformat(),
                "initialization_selection": initialization_selection,
                "anchor_wind_ms": float(row["anchor_wind_ms"]),
                "current_ibtracs_wind_ms": float(row["current_ibtracs_wind_ms"]),
                "wind_minus_6h_ms": float(row["wind_minus_6h_ms"]),
                "wind_minus_12h_ms": float(row["wind_minus_12h_ms"]),
                "target_plus_6h_wind_ms": float(row["target_wind_ms"]),
                "target_plus_12h_wind_ms": float(target_12),
            }
        )
    return cases


def _feature_scaler(rows: list[dict[str, Any]]) -> dict[str, Any]:
    features = np.stack(
        [
            forecast_features(
                float(row["anchor_wind_ms"]),
                float(row["wind_minus_6h_ms"]),
                float(row["wind_minus_12h_ms"]),
            )
            for row in rows
        ]
    )
    std = features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {
        "source_split": "pretrain_train",
        "names": list(FORECAST_FEATURE_NAMES),
        "mean": features.mean(axis=0).tolist(),
        "std": std.tolist(),
    }


def export_forecast_cache(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.ibtracs_file,
        args.intensity_cache_root,
        args.intensity_checkpoint,
    ):
        if not Path(path).expanduser().exists():
            raise FileNotFoundError(path)
    tracks = read_ibtracs_forecast_tracks(args.ibtracs_file)
    historical = build_historical_rows(
        tracks,
        start_year=args.pretrain_start_year,
        train_end_year=args.pretrain_train_end_year,
        val_end_year=args.pretrain_val_end_year,
    )
    predictions = _current_intensity_predictions(
        args.intensity_cache_root,
        args.intensity_checkpoint,
        ("train", "val", "test"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    matched = build_matched_rows(args.intensity_cache_root, tracks, predictions)
    ri_cases = select_ri_validation_cases(matched["val"], tracks)
    all_splits = {**historical, **matched}
    if any(not rows for rows in all_splits.values()):
        empty = [split for split, rows in all_splits.items() if not rows]
        raise RuntimeError(f"forecast export produced empty splits: {empty}")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    combined = []
    for split, rows in all_splits.items():
        frame = pd.DataFrame(rows).sort_values(
            ["storm_id", "init_timestamp", "sample_id"], kind="stable"
        )
        _write_csv_atomic(frame, output_root / split / "manifest.csv")
        combined.extend(frame.to_dict("records"))
    _write_csv_atomic(pd.DataFrame(combined), output_root / "manifest.csv")

    intensity_metadata_path = args.intensity_cache_root / "cache-metadata.json"
    intensity_metadata = json.loads(intensity_metadata_path.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": FORECAST_CACHE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "scalar_maximum_wind_6h_forecast",
        "history_hours": 12,
        "forecast_hours": 6,
        "pretraining_years": {
            "start": args.pretrain_start_year,
            "train_end": args.pretrain_train_end_year,
            "validation_end": args.pretrain_val_end_year,
        },
        "feature_scaler": _feature_scaler(historical["pretrain_train"]),
        "ri_definition": {
            "threshold_kt": RI_THRESHOLD_KT,
            "window_hours": RI_WINDOW_HOURS,
            "initialization_policy": (
                "latest matched row at or before RI onset; otherwise earliest "
                "matched row after onset"
            ),
        },
        "ri_validation_cases": ri_cases,
        "ibtracs": {
            "path": str(args.ibtracs_file.expanduser().resolve()),
            "sha256": _sha256(args.ibtracs_file),
            "wind_column": "USA_WIND",
            "knot_to_ms": KNOT_TO_MS,
        },
        "intensity_cache": {
            "path": str(args.intensity_cache_root.expanduser().resolve()),
            "metadata_sha256": _sha256(intensity_metadata_path),
            "scientific_evaluation": intensity_metadata.get(
                "scientific_evaluation", "unspecified"
            ),
        },
        "intensity_checkpoint": {
            "path": str(args.intensity_checkpoint.expanduser().resolve()),
            "sha256": _sha256(args.intensity_checkpoint),
        },
        "splits": {split: len(rows) for split, rows in all_splits.items()},
    }
    (output_root / "cache-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    metadata = export_forecast_cache(parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
