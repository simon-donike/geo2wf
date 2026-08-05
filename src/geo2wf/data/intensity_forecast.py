"""Scalar 6-hour tropical-cyclone intensity forecast datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


FORECAST_CACHE_SCHEMA_VERSION = 1
FORECAST_FEATURE_NAMES = (
    "anchor_wind_ms",
    "wind_minus_6h_ms",
    "wind_minus_12h_ms",
    "anchor_change_6h_ms",
    "previous_change_6h_ms",
)
REQUIRED_COLUMNS = frozenset(
    {
        "sample_id",
        "storm_id",
        "split",
        "init_timestamp",
        "anchor_wind_ms",
        "wind_minus_6h_ms",
        "wind_minus_12h_ms",
        "target_wind_ms",
        "target_delta_ms",
        "source_kind",
    }
)


@dataclass(frozen=True)
class IntensityForecastDataSpec:
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    cache_schema_version: int = FORECAST_CACHE_SCHEMA_VERSION

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


def forecast_features(
    anchor_wind_ms: float,
    wind_minus_6h_ms: float,
    wind_minus_12h_ms: float,
) -> np.ndarray:
    values = np.asarray(
        [
            anchor_wind_ms,
            wind_minus_6h_ms,
            wind_minus_12h_ms,
            anchor_wind_ms - wind_minus_6h_ms,
            wind_minus_6h_ms - wind_minus_12h_ms,
        ],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("forecast scalar features must all be finite")
    return values


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _cache_metadata(root: Path) -> dict[str, Any]:
    path = root / "cache-metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"forecast cache metadata does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FORECAST_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported forecast cache schema: " f"{payload.get('schema_version')!r}"
        )
    scaler = payload.get("feature_scaler", {})
    if tuple(scaler.get("names", ())) != FORECAST_FEATURE_NAMES:
        raise ValueError("forecast cache feature scaler names are incompatible")
    if len(scaler.get("mean", ())) != len(FORECAST_FEATURE_NAMES) or len(
        scaler.get("std", ())
    ) != len(FORECAST_FEATURE_NAMES):
        raise ValueError("forecast cache feature scaler has the wrong width")
    return payload


class IntensityForecastDataset(Dataset):
    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root).expanduser()
        self.split = str(split)
        self.cache_metadata = _cache_metadata(self.root)
        self.manifest_path = self.root / self.split / "manifest.csv"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"forecast split manifest does not exist: {self.manifest_path}"
            )
        self.samples = pd.read_csv(self.manifest_path, keep_default_na=False)
        missing = REQUIRED_COLUMNS.difference(self.samples.columns)
        if missing:
            raise ValueError(
                f"{self.manifest_path} is missing columns: {sorted(missing)}"
            )
        if set(self.samples["split"].astype(str)).difference({self.split}):
            raise ValueError(f"{self.manifest_path} contains another declared split")
        for name in (
            "anchor_wind_ms",
            "wind_minus_6h_ms",
            "wind_minus_12h_ms",
            "target_wind_ms",
            "target_delta_ms",
        ):
            self.samples[name] = pd.to_numeric(self.samples[name], errors="coerce")
        finite = np.isfinite(
            self.samples[
                [
                    "anchor_wind_ms",
                    "wind_minus_6h_ms",
                    "wind_minus_12h_ms",
                    "target_wind_ms",
                    "target_delta_ms",
                ]
            ].to_numpy(dtype=float)
        ).all(axis=1)
        self.samples = self.samples.loc[finite].reset_index(drop=True)
        if self.samples.empty:
            raise ValueError(f"forecast split {self.split!r} has no usable samples")
        self.samples["_sample_weight"] = 1.0

    @property
    def data_spec(self) -> IntensityForecastDataSpec:
        scaler = self.cache_metadata["feature_scaler"]
        return IntensityForecastDataSpec(
            feature_names=FORECAST_FEATURE_NAMES,
            feature_mean=tuple(float(value) for value in scaler["mean"]),
            feature_std=tuple(float(value) for value in scaler["std"]),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        anchor = _finite(row["anchor_wind_ms"], "anchor_wind_ms")
        minus_6 = _finite(row["wind_minus_6h_ms"], "wind_minus_6h_ms")
        minus_12 = _finite(row["wind_minus_12h_ms"], "wind_minus_12h_ms")
        return {
            "features": torch.from_numpy(forecast_features(anchor, minus_6, minus_12)),
            "anchor_wind_ms": torch.tensor(anchor, dtype=torch.float32),
            "wind_minus_6h_ms": torch.tensor(minus_6, dtype=torch.float32),
            "wind_minus_12h_ms": torch.tensor(minus_12, dtype=torch.float32),
            "target_wind_ms": torch.tensor(
                _finite(row["target_wind_ms"], "target_wind_ms"),
                dtype=torch.float32,
            ),
            "target_delta_ms": torch.tensor(
                _finite(row["target_delta_ms"], "target_delta_ms"),
                dtype=torch.float32,
            ),
            "sample_weight": torch.tensor(
                _finite(row["_sample_weight"], "_sample_weight"),
                dtype=torch.float32,
            ),
            "sample_id": str(row["sample_id"]),
            "storm_id": str(row["storm_id"]),
            "init_timestamp": str(row["init_timestamp"]),
            "source_kind": str(row["source_kind"]),
        }


def _assert_storm_disjoint(frames: dict[str, pd.DataFrame]) -> None:
    owners: dict[str, str] = {}
    conflicts: dict[str, set[str]] = {}
    for split, frame in frames.items():
        for storm_id in (
            frame.get("storm_id", pd.Series(dtype=str)).astype(str).unique()
        ):
            previous = owners.setdefault(storm_id, split)
            if previous != split:
                conflicts.setdefault(storm_id, {previous}).add(split)
    if conflicts:
        detail = ", ".join(
            f"{storm}: {sorted(splits)}" for storm, splits in sorted(conflicts.items())
        )
        raise ValueError(f"forecast splits are not storm-disjoint: {detail}")


def _training_weights(samples: pd.DataFrame, maximum_ratio: float) -> np.ndarray:
    if maximum_ratio < 1.0:
        raise ValueError("maximum delta weight ratio must be at least one")
    storm_count = samples.groupby("storm_id")["sample_id"].transform("size")
    delta = samples["target_delta_ms"].to_numpy(dtype=float)
    bins = np.asarray([-np.inf, -5.0, -2.0, 2.0, 5.0, np.inf])
    labels = np.digitize(delta, bins[1:-1], right=False)
    counts = np.bincount(labels, minlength=len(bins) - 1)
    delta_factor = np.sqrt(counts.max() / counts[labels])
    delta_factor = np.minimum(delta_factor, maximum_ratio)
    weights = delta_factor / storm_count.to_numpy(dtype=float)
    return weights / weights.mean()


class IntensityForecastDataModule(pl.LightningDataModule):
    intensity_balanced_sampling = False

    def __init__(
        self,
        root: str | Path,
        *,
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        maximum_delta_weight_ratio: float = 4.0,
        ri_storm_ids: Sequence[str] = ("WP282025", "WP112024", "AL092024"),
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        self.train_split = str(train_split)
        self.val_split = str(val_split)
        self.test_split = str(test_split)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.maximum_delta_weight_ratio = float(maximum_delta_weight_ratio)
        self.ri_storm_ids = tuple(str(value) for value in ri_storm_ids)
        self.train_dataset: IntensityForecastDataset | None = None
        self.val_dataset: IntensityForecastDataset | None = None
        self.test_dataset: IntensityForecastDataset | None = None
        frames = {}
        for split in {self.train_split, self.val_split, self.test_split}:
            path = self.root / split / "manifest.csv"
            if path.is_file():
                frames[split] = pd.read_csv(path, keep_default_na=False)
        _assert_storm_disjoint(frames)

    def _dataset(self, split: str) -> IntensityForecastDataset:
        return IntensityForecastDataset(self.root, split)

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._dataset(self.train_split)
            self.val_dataset = self._dataset(self.val_split)
            self.train_dataset.samples["_sample_weight"] = _training_weights(
                self.train_dataset.samples, self.maximum_delta_weight_ratio
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = self._dataset(self.test_split)

    @property
    def data_spec(self) -> IntensityForecastDataSpec:
        if self.val_dataset is None:
            self.setup("fit")
        assert self.val_dataset is not None
        return self.val_dataset.data_spec

    @property
    def ri_rollout_cases(self) -> list[dict[str, Any]]:
        if self.val_dataset is None:
            self.setup("fit")
        assert self.val_dataset is not None
        configured = self.val_dataset.cache_metadata.get("ri_validation_cases", [])
        by_storm = {str(case["storm_id"]): case for case in configured}
        missing = [storm for storm in self.ri_storm_ids if storm not in by_storm]
        if missing:
            raise ValueError(
                "forecast cache is missing configured RI validation cases: "
                + ", ".join(missing)
            )
        return [dict(by_storm[storm]) for storm in self.ri_storm_ids]

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        return self._loader(self.train_dataset, True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        assert self.val_dataset is not None
        return self._loader(self.val_dataset, False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        assert self.test_dataset is not None
        return self._loader(self.test_dataset, False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


__all__ = [
    "FORECAST_CACHE_SCHEMA_VERSION",
    "FORECAST_FEATURE_NAMES",
    "IntensityForecastDataModule",
    "IntensityForecastDataSpec",
    "IntensityForecastDataset",
    "forecast_features",
]
