"""Join paired wind-field samples to continuous IBTrACS intensity labels."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from geo2wf.data.datamodule import PairedDataModule
from geo2wf.data.datasets.paired_geotiff import PairedImageDataset


IBTRACS_MAX_WIND_COMPANION = "ibtracs_max_wind"
INTENSITY_CACHE_SCHEMA_VERSION = 1
KNOT_TO_MS = 0.514444
REQUIRED_INTENSITY_COLUMNS = frozenset(
    {
        "source_sample_id",
        "storm_id",
        "split",
        "observation_timestamp",
        "target_wind_ms",
    }
)


def _load_and_validate_cache_metadata(root: Path) -> dict[str, Any]:
    path = root / "cache-metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Intensity cache metadata does not exist: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != INTENSITY_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported intensity cache schema: " f"{metadata.get('schema_version')!r}"
        )
    target = metadata.get("target", {})
    if target.get("source") != "IBTrACS USA_WIND":
        raise ValueError(
            "joint intensity labels must declare target source "
            f"'IBTrACS USA_WIND', got {target.get('source')!r}"
        )
    if target.get("units") != "m s-1":
        raise ValueError(
            "joint intensity labels must use m s-1, got " f"{target.get('units')!r}"
        )
    conversion = target.get("knot_to_ms")
    if conversion is None or not math.isclose(float(conversion), KNOT_TO_MS):
        raise ValueError(
            "joint intensity labels must use the declared knots-to-m/s "
            f"conversion {KNOT_TO_MS}, got {conversion!r}"
        )
    return metadata


def _read_intensity_manifest(root: Path, split: str) -> pd.DataFrame:
    path = root / split / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Intensity split manifest does not exist: {path}")
    labels = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED_INTENSITY_COLUMNS.difference(labels.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    declared_splits = set(labels["split"].astype(str))
    if declared_splits != {split}:
        raise ValueError(
            f"{path} must contain only split {split!r}, got {sorted(declared_splits)}"
        )
    source_ids = labels["source_sample_id"].astype(str)
    if source_ids.duplicated().any():
        duplicates = sorted(source_ids[source_ids.duplicated(keep=False)].unique())
        raise ValueError(
            "intensity source_sample_id values must be unique; duplicates: "
            + ", ".join(duplicates[:5])
        )
    target = pd.to_numeric(labels["target_wind_ms"], errors="coerce")
    valid = target.map(lambda value: math.isfinite(float(value)) and value >= 0.0)
    if not bool(valid.all()):
        raise ValueError(f"{path} contains invalid continuous IBTrACS wind targets")
    labels = labels.copy()
    labels["source_sample_id"] = source_ids
    labels["storm_id"] = labels["storm_id"].astype(str)
    labels["target_wind_ms"] = target.astype(float)
    return labels


def _validate_storm_disjoint(labels_by_split: dict[str, pd.DataFrame]) -> None:
    owners: dict[str, str] = {}
    conflicts: dict[str, set[str]] = {}
    for split, labels in labels_by_split.items():
        for storm_id in labels["storm_id"].astype(str).unique():
            previous = owners.setdefault(storm_id, split)
            if previous != split:
                conflicts.setdefault(storm_id, {previous}).add(split)
    if conflicts:
        detail = ", ".join(
            f"{storm}: {sorted(splits)}" for storm, splits in sorted(conflicts.items())
        )
        raise ValueError(f"joint intensity splits are not storm-disjoint: {detail}")


class JointPairedIntensityDataset(Dataset):
    """Filter an existing paired dataset to samples with IBTrACS wind labels."""

    def __init__(
        self,
        paired_dataset: PairedImageDataset,
        intensity_root: str | Path,
        split: str,
    ) -> None:
        self.paired_dataset = paired_dataset
        self.intensity_root = Path(intensity_root).expanduser()
        self.split = str(split)
        _load_and_validate_cache_metadata(self.intensity_root)
        labels = _read_intensity_manifest(self.intensity_root, self.split)

        paired_samples = paired_dataset.samples
        paired_ids = paired_samples["sample_id"].astype(str)
        if paired_ids.duplicated().any():
            raise ValueError("paired dataset sample_id values must be unique")
        paired_id_set = set(paired_ids)
        missing = sorted(set(labels["source_sample_id"]) - paired_id_set)
        if missing:
            raise ValueError(
                f"{len(missing)} intensity labels do not map to eligible paired "
                f"samples in split {self.split!r}: {', '.join(missing[:5])}"
            )

        labels_by_source = labels.set_index("source_sample_id", drop=False)
        self._paired_indices = [
            int(index)
            for index, sample_id in enumerate(paired_ids)
            if sample_id in labels_by_source.index
        ]
        self.samples = paired_samples.iloc[self._paired_indices].reset_index(drop=True)
        self._labels = [
            labels_by_source.loc[str(sample_id)].to_dict()
            for sample_id in self.samples["sample_id"].astype(str)
        ]
        if len(self.samples) != len(labels):
            raise RuntimeError(
                "paired/intensity join did not retain exactly one row per label"
            )
        for paired_row, label in zip(self.samples.to_dict("records"), self._labels):
            if str(paired_row["storm_id"]) != str(label["storm_id"]):
                raise ValueError(
                    f"storm mismatch for source sample {paired_row['sample_id']}: "
                    f"paired={paired_row['storm_id']!r}, label={label['storm_id']!r}"
                )

    @property
    def root(self) -> Path:
        return self.paired_dataset.root

    @property
    def data_spec(self):
        spec = self.paired_dataset.data_spec
        return replace(
            spec,
            companions=spec.companions | {IBTRACS_MAX_WIND_COMPANION},
        )

    def __len__(self) -> int:
        return len(self._paired_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.paired_dataset[self._paired_indices[index]]
        label = self._labels[index]
        sample["intensity_target_ms"] = torch.tensor(
            float(label["target_wind_ms"]), dtype=torch.float32
        )
        sample["intensity_observation_timestamp"] = str(label["observation_timestamp"])
        return sample


class JointPairedIntensityDataModule(PairedDataModule):
    """Use the canonical paired loaders on the jointly labeled subset."""

    def __init__(self, *args: Any, intensity_root: str | Path, **kwargs: Any) -> None:
        if kwargs.get("include_test_in_train", False):
            raise ValueError(
                "JointPairedIntensityDataModule requires include_test_in_train=false"
            )
        self.intensity_root = Path(intensity_root).expanduser()
        _load_and_validate_cache_metadata(self.intensity_root)
        split_names = {
            str(kwargs.get("train_split", "train")),
            str(kwargs.get("val_split", "val")),
            str(kwargs.get("test_split", "test")),
        }
        labels_by_split = {
            split: _read_intensity_manifest(self.intensity_root, split)
            for split in split_names
        }
        _validate_storm_disjoint(labels_by_split)
        super().__init__(*args, **kwargs)

    def _make_dataset(
        self, split: str, *, augment: bool = False
    ) -> JointPairedIntensityDataset:
        paired = super()._make_dataset(split, augment=augment)
        return JointPairedIntensityDataset(paired, self.intensity_root, split)


__all__ = [
    "IBTRACS_MAX_WIND_COMPANION",
    "JointPairedIntensityDataModule",
    "JointPairedIntensityDataset",
]
