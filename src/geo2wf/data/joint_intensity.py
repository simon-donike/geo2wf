"""Join paired wind-field samples to continuous IBTrACS intensity labels."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import replace
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from geo2wf.data.datamodule import PairedDataModule
from geo2wf.data.datasets.paired_geotiff import PairedImageDataset


IBTRACS_MAX_WIND_COMPANION = "ibtracs_max_wind"
KNOT_TO_MS = 0.514444
REQUIRED_IBTRACS_COLUMNS = frozenset({"USA_ATCF_ID", "ISO_TIME", "USA_WIND"})


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


def _load_ibtracs_tracks(
    path: str | Path,
    storm_ids: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load finite USA_WIND fixes without introducing category supervision."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"IBTrACS file does not exist: {path}")
    available_columns = set(pd.read_csv(path, nrows=0).columns)
    missing = REQUIRED_IBTRACS_COLUMNS.difference(available_columns)
    if missing:
        raise ValueError(f"{path} is missing IBTrACS columns: {sorted(missing)}")
    frame = pd.read_csv(
        path,
        usecols=sorted(REQUIRED_IBTRACS_COLUMNS),
        keep_default_na=False,
        low_memory=False,
    )
    frame["storm_id"] = frame["USA_ATCF_ID"].astype(str).str.strip().str.upper()
    frame["timestamp"] = pd.to_datetime(
        frame["ISO_TIME"], errors="coerce", utc=True, format="mixed"
    )
    frame["wind_kt"] = pd.to_numeric(frame["USA_WIND"], errors="coerce")
    frame = frame.loc[
        frame["storm_id"].ne("")
        & frame["timestamp"].notna()
        & frame["wind_kt"].map(
            lambda value: math.isfinite(float(value)) and float(value) >= 0.0
        )
    ].copy()
    if storm_ids is not None:
        frame = frame.loc[frame["storm_id"].isin(storm_ids)].copy()
    if frame.empty:
        raise ValueError(f"{path} contains no finite IBTrACS USA_WIND fixes")

    duplicate = frame.duplicated(["storm_id", "timestamp"], keep=False)
    if duplicate.any():
        conflicts = (
            frame.loc[duplicate].groupby(["storm_id", "timestamp"])["wind_kt"].nunique()
        )
        if (conflicts > 1).any():
            raise ValueError(
                "IBTrACS contains conflicting USA_WIND values at one storm time"
            )
    frame = frame.drop_duplicates(["storm_id", "timestamp"], keep="last")
    return {
        storm_id: fixes.sort_values("timestamp").reset_index(drop=True)
        for storm_id, fixes in frame.groupby("storm_id")
    }


def _interpolate_ibtracs_wind(
    fixes: pd.DataFrame,
    timestamp: Any,
    *,
    max_bracket_hours: float,
) -> dict[str, Any] | None:
    """Linearly interpolate USA_WIND inside a sufficiently narrow time bracket."""
    target_time = pd.Timestamp(timestamp)
    if pd.isna(target_time):
        return None
    target_time = (
        target_time.tz_localize("UTC")
        if target_time.tzinfo is None
        else target_time.tz_convert("UTC")
    )
    times = fixes["timestamp"].tolist()
    upper = bisect_left(times, target_time)
    if upper < len(times) and times[upper] == target_time:
        wind_kt = float(fixes.iloc[upper]["wind_kt"])
        lower_time = upper_time = target_time
    else:
        if upper == 0 or upper == len(times):
            return None
        lower = upper - 1
        lower_time, upper_time = times[lower], times[upper]
        bracket = upper_time - lower_time
        if bracket > pd.Timedelta(hours=max_bracket_hours):
            return None
        fraction = (target_time - lower_time) / bracket
        lower_wind = float(fixes.iloc[lower]["wind_kt"])
        upper_wind = float(fixes.iloc[upper]["wind_kt"])
        wind_kt = lower_wind + float(fraction) * (upper_wind - lower_wind)
    return {
        "target_wind_ms": wind_kt * KNOT_TO_MS,
        "observation_timestamp": target_time.isoformat(),
        "lower_fix_timestamp": lower_time.isoformat(),
        "upper_fix_timestamp": upper_time.isoformat(),
    }


class JointPairedIntensityDataset(Dataset):
    """Attach time-interpolated IBTrACS USA_WIND to paired SAR samples."""

    def __init__(
        self,
        paired_dataset: PairedImageDataset,
        ibtracs_tracks: dict[str, pd.DataFrame],
        split: str,
        *,
        max_bracket_hours: float = 3.0,
    ) -> None:
        self.paired_dataset = paired_dataset
        self.split = str(split)
        self.max_bracket_hours = float(max_bracket_hours)
        if self.max_bracket_hours <= 0:
            raise ValueError("max_bracket_hours must be positive")

        paired_samples = paired_dataset.samples
        paired_ids = paired_samples["sample_id"].astype(str)
        if paired_ids.duplicated().any():
            raise ValueError("paired dataset sample_id values must be unique")
        if "target_timestamp" not in paired_samples:
            raise ValueError(
                "paired manifest requires target_timestamp for IBTrACS interpolation"
            )

        self._paired_indices: list[int] = []
        self._labels: list[dict[str, Any]] = []
        for index, row in paired_samples.iterrows():
            storm_id = str(row["storm_id"]).strip().upper()
            fixes = ibtracs_tracks.get(storm_id)
            if fixes is None:
                continue
            label = _interpolate_ibtracs_wind(
                fixes,
                row["target_timestamp"],
                max_bracket_hours=self.max_bracket_hours,
            )
            if label is None:
                continue
            label.update(
                {
                    "source_sample_id": str(row["sample_id"]),
                    "storm_id": storm_id,
                }
            )
            self._paired_indices.append(int(index))
            self._labels.append(label)

        self.filtered_unbracketed_count = len(paired_samples) - len(
            self._paired_indices
        )
        self.samples = paired_samples.iloc[self._paired_indices].reset_index(drop=True)

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
        sample["intensity_lower_fix_timestamp"] = str(label["lower_fix_timestamp"])
        sample["intensity_upper_fix_timestamp"] = str(label["upper_fix_timestamp"])
        sample["intensity_target_ms"] = torch.tensor(
            float(label["target_wind_ms"]), dtype=torch.float32
        )
        sample["intensity_observation_timestamp"] = str(label["observation_timestamp"])
        return sample


class JointPairedIntensityDataModule(PairedDataModule):
    """Pair SAR fields with directly interpolated continuous IBTrACS labels."""

    def __init__(
        self,
        *args: Any,
        ibtracs_file: str | Path,
        max_ibtracs_bracket_hours: float = 3.0,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("include_test_in_train", False):
            raise ValueError(
                "JointPairedIntensityDataModule requires include_test_in_train=false"
            )
        self.ibtracs_file = Path(ibtracs_file).expanduser()
        self.max_ibtracs_bracket_hours = float(max_ibtracs_bracket_hours)
        if self.max_ibtracs_bracket_hours <= 0:
            raise ValueError("max_ibtracs_bracket_hours must be positive")
        super().__init__(*args, **kwargs)

        split_frames = {
            split: pd.read_csv(
                self.root / split / "manifest.csv",
                usecols=["storm_id"],
                keep_default_na=False,
            )
            for split in {self.train_split, self.val_split, self.test_split}
        }
        _validate_storm_disjoint(split_frames)
        storm_ids = set().union(
            *(
                set(frame["storm_id"].astype(str).str.upper())
                for frame in split_frames.values()
            )
        )
        self.ibtracs_tracks = _load_ibtracs_tracks(self.ibtracs_file, storm_ids)

    def _make_dataset(
        self, split: str, *, augment: bool = False
    ) -> JointPairedIntensityDataset:
        paired = super()._make_dataset(split, augment=augment)
        return JointPairedIntensityDataset(
            paired,
            self.ibtracs_tracks,
            split,
            max_bracket_hours=self.max_ibtracs_bracket_hours,
        )


__all__ = [
    "IBTRACS_MAX_WIND_COMPANION",
    "JointPairedIntensityDataModule",
    "JointPairedIntensityDataset",
]
