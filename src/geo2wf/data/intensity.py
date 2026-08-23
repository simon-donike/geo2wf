"""Single-timestep U-Net wind-field data for scalar intensity correction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


INTENSITY_CACHE_SCHEMA_VERSION = 2
SUPPORTED_INTENSITY_CACHE_SCHEMA_VERSIONS = frozenset({1, 2})
KNOT_TO_MS = 0.514444
TROPICAL_CATEGORY_MIN = -1
TROPICAL_CATEGORY_MAX = 5
BASINS = ("NA", "EP", "WP", "NI", "SI", "SP", "SA", "OTHER")
INTENSITY_METADATA_NAMES = (
    "latitude_scaled",
    "longitude_sin",
    "longitude_cos",
    *(f"basin_{basin.lower()}" for basin in BASINS),
    "utc_time_sin",
    "utc_time_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "local_solar_time_sin",
    "local_solar_time_cos",
    "storm_elapsed_log_scaled",
    "valid_fraction",
)

REQUIRED_MANIFEST_COLUMNS = frozenset(
    {
        "sample_id",
        "storm_id",
        "split",
        "field_path",
        "observation_timestamp",
        "target_timestamp",
        "center_lat",
        "center_lon",
        "basin",
        "storm_elapsed_hours",
        "target_wind_ms",
        "target_category",
        "raw_unet_max_wind_ms",
        "valid_fraction",
    }
)
V2_MANIFEST_COLUMNS = frozenset(
    {
        "intensity_target_source",
        "anchor_statistic",
        "ibtracs_target_ms",
        "sar_robust_peak_target_ms",
        "sar_max_wind_ms",
        "raw_unet_robust_peak_ms",
        "is_rapid_intensification",
        "ri_24h_change_ms",
        "cohort_retained_count",
        "filtered_unbracketed_count",
        "filtered_invalid_sar_center_count",
        "filtered_unusable_sar_count",
    }
)


@dataclass(frozen=True)
class IntensityDataSpec:
    """Shape and feature contract for one cached U-Net field."""

    spatial_shape: tuple[int, int]
    metadata_names: tuple[str, ...] = INTENSITY_METADATA_NAMES
    cache_schema_version: int = INTENSITY_CACHE_SCHEMA_VERSION

    @property
    def metadata_feature_count(self) -> int:
        return len(self.metadata_names)


def tropical_category_from_wind_ms_tensor(wind_ms: torch.Tensor) -> torch.Tensor:
    """Map continuous one-minute wind to TD/TS/C1--C5 without rounding."""

    wind_kt = wind_ms / KNOT_TO_MS
    category = torch.full_like(wind_kt, -1, dtype=torch.long)
    for threshold_kt, value in (
        (34.0, 0),
        (64.0, 1),
        (83.0, 2),
        (96.0, 3),
        (113.0, 4),
        (137.0, 5),
    ):
        category = torch.where(
            wind_kt >= threshold_kt,
            torch.as_tensor(value, device=wind_ms.device),
            category,
        )
    return category


def tropical_category_from_wind_ms(wind_ms: float) -> int:
    value = torch.tensor(float(wind_ms), dtype=torch.float64)
    return int(tropical_category_from_wind_ms_tensor(value).item())


def category_macro_f1_tensor(
    prediction_category: torch.Tensor, target_category: torch.Tensor
) -> torch.Tensor:
    """Macro F1 over categories represented in the reference cohort."""

    prediction_category = prediction_category.reshape(-1)
    target_category = target_category.reshape(-1)
    if (
        prediction_category.shape != target_category.shape
        or not target_category.numel()
    ):
        raise ValueError("category tensors must have one matching non-empty dimension")
    values = []
    for category in range(TROPICAL_CATEGORY_MIN, TROPICAL_CATEGORY_MAX + 1):
        observed = target_category == category
        if not bool(observed.any()):
            continue
        predicted = prediction_category == category
        true_positive = (observed & predicted).sum().to(torch.float64)
        false_positive = ((~observed) & predicted).sum().to(torch.float64)
        false_negative = (observed & (~predicted)).sum().to(torch.float64)
        values.append(
            2.0
            * true_positive
            / (2.0 * true_positive + false_positive + false_negative)
        )
    return torch.stack(values).mean()


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def encode_intensity_metadata(row: pd.Series | dict[str, Any]) -> torch.Tensor:
    """Encode only scientifically available current-timestep metadata."""

    timestamp = pd.Timestamp(row["observation_timestamp"])
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    latitude = _finite_float(row["center_lat"], "center_lat")
    longitude = _finite_float(row["center_lon"], "center_lon")
    elapsed_hours = max(
        0.0, _finite_float(row["storm_elapsed_hours"], "storm_elapsed_hours")
    )
    valid_fraction = float(
        np.clip(_finite_float(row["valid_fraction"], "valid_fraction"), 0.0, 1.0)
    )

    longitude_radians = math.radians(longitude)
    utc_hours = (
        timestamp.hour
        + timestamp.minute / 60.0
        + timestamp.second / 3600.0
        + timestamp.microsecond / 3.6e9
    )
    utc_angle = 2.0 * math.pi * utc_hours / 24.0
    year_fraction = (timestamp.dayofyear - 1 + utc_hours / 24.0) / 365.2425
    year_angle = 2.0 * math.pi * year_fraction
    local_hours = (utc_hours + longitude / 15.0) % 24.0
    local_angle = 2.0 * math.pi * local_hours / 24.0
    basin = str(row.get("basin", "OTHER")).strip().upper()
    basin = basin if basin in BASINS[:-1] else "OTHER"
    basin_encoding = [float(basin == candidate) for candidate in BASINS]

    values = [
        float(np.clip(latitude / 90.0, -1.0, 1.0)),
        math.sin(longitude_radians),
        math.cos(longitude_radians),
        *basin_encoding,
        math.sin(utc_angle),
        math.cos(utc_angle),
        math.sin(year_angle),
        math.cos(year_angle),
        math.sin(local_angle),
        math.cos(local_angle),
        math.log1p(elapsed_hours) / math.log1p(30.0 * 24.0),
        valid_fraction,
    ]
    if len(values) != len(INTENSITY_METADATA_NAMES):
        raise RuntimeError("intensity metadata schema and encoder disagree")
    return torch.tensor(values, dtype=torch.float32)


def _read_cache_metadata(root: Path) -> dict[str, Any]:
    path = root / "cache-metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Intensity cache metadata does not exist: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    version = metadata.get("schema_version")
    if version not in SUPPORTED_INTENSITY_CACHE_SCHEMA_VERSIONS:
        raise ValueError(
            "unsupported intensity cache schema: "
            f"expected one of {sorted(SUPPORTED_INTENSITY_CACHE_SCHEMA_VERSIONS)}, "
            f"got {version!r}"
        )
    return metadata


class UNetIntensityDataset(Dataset):
    """Load one frozen U-Net prediction and matched scalar wind references."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        expected_unet_checkpoint_sha256: str | None = None,
        verify_cache: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        self.split = str(split)
        self.cache_metadata = _read_cache_metadata(self.root)
        actual_sha = str(
            self.cache_metadata.get("unet_checkpoint", {}).get("sha256", "")
        )
        if (
            expected_unet_checkpoint_sha256 is not None
            and actual_sha != expected_unet_checkpoint_sha256
        ):
            raise ValueError(
                "intensity cache U-Net checkpoint mismatch: expected "
                f"{expected_unet_checkpoint_sha256}, got {actual_sha or 'missing'}"
            )
        self.manifest_path = self.root / self.split / "manifest.csv"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Intensity split manifest does not exist: {self.manifest_path}"
            )
        self.samples = pd.read_csv(self.manifest_path, keep_default_na=False)
        missing = REQUIRED_MANIFEST_COLUMNS.difference(self.samples.columns)
        if self.cache_metadata.get("schema_version") == 2:
            missing |= V2_MANIFEST_COLUMNS.difference(self.samples.columns)
        if missing:
            raise ValueError(
                f"{self.manifest_path} is missing columns: {sorted(missing)}"
            )
        declared_splits = set(self.samples["split"].astype(str))
        if declared_splits.difference({self.split}):
            raise ValueError(
                f"{self.manifest_path} contains rows from splits {declared_splits}"
            )
        categories = pd.to_numeric(self.samples["target_category"], errors="coerce")
        winds = pd.to_numeric(self.samples["target_wind_ms"], errors="coerce")
        keep = (
            categories.between(TROPICAL_CATEGORY_MIN, TROPICAL_CATEGORY_MAX)
            & winds.notna()
        )
        self.samples = self.samples.loc[keep].reset_index(drop=True)
        if self.samples.empty:
            raise ValueError(f"Intensity split {self.split!r} has no usable samples")
        if "_sample_weight" not in self.samples:
            self.samples["_sample_weight"] = 1.0
        self.verify_cache = bool(verify_cache)
        self._data_spec: IntensityDataSpec | None = None

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def data_spec(self) -> IntensityDataSpec:
        if self._data_spec is None:
            sample = self[0]
            self._data_spec = IntensityDataSpec(
                spatial_shape=tuple(int(value) for value in sample["wind_field"].shape),
                cache_schema_version=int(self.cache_metadata["schema_version"]),
            )
        return self._data_spec

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        path = self.root / str(row["field_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Cached U-Net field does not exist: {path}")
        with np.load(path, allow_pickle=False) as payload:
            required = {"wind_speed_ms", "valid_mask", "distance_to_center"}
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            wind = np.asarray(payload["wind_speed_ms"], dtype=np.float32).squeeze()
            mask = np.asarray(payload["valid_mask"], dtype=bool).squeeze()
            distance = np.asarray(
                payload["distance_to_center"], dtype=np.float32
            ).squeeze()
        if wind.ndim != 2 or mask.shape != wind.shape or distance.shape != wind.shape:
            raise ValueError(
                f"{path} arrays must share one [H,W] shape, got "
                f"{wind.shape}, {mask.shape}, {distance.shape}"
            )
        finite_valid = mask & np.isfinite(wind) & np.isfinite(distance)
        if not finite_valid.any():
            raise ValueError(f"{path} has no finite valid pixels")
        raw_max = float(np.max(wind[finite_valid]))
        valid_values = wind[finite_valid]
        robust_fraction = float(
            self.cache_metadata.get("target", {}).get("sar_robust_peak_fraction", 0.005)
        )
        if not 0.0 < robust_fraction <= 1.0:
            raise ValueError("cached SAR robust-peak fraction must be in (0, 1]")
        robust_count = max(1, int(math.ceil(len(valid_values) * robust_fraction)))
        raw_robust_peak = float(
            np.mean(np.partition(valid_values, -robust_count)[-robust_count:])
        )
        if self.verify_cache and not math.isclose(
            raw_max,
            _finite_float(row["raw_unet_max_wind_ms"], "raw_unet_max_wind_ms"),
            rel_tol=1e-5,
            abs_tol=1e-4,
        ):
            raise ValueError(f"{path} raw maximum disagrees with its manifest row")
        if (
            self.cache_metadata.get("schema_version") == 2
            and self.verify_cache
            and not math.isclose(
                raw_robust_peak,
                _finite_float(
                    row["raw_unet_robust_peak_ms"], "raw_unet_robust_peak_ms"
                ),
                rel_tol=1e-5,
                abs_tol=1e-4,
            )
        ):
            raise ValueError(f"{path} robust peak disagrees with its manifest row")
        wind = np.where(finite_valid, wind, 0.0).astype(np.float32, copy=False)
        distance = np.nan_to_num(distance, nan=0.0, posinf=1.0, neginf=0.0)
        distance = np.clip(distance, 0.0, 1.0).astype(np.float32, copy=False)
        version = int(self.cache_metadata.get("schema_version", 1))
        target_wind_ms = _finite_float(row["target_wind_ms"], "target_wind_ms")
        target_source = (
            str(row["intensity_target_source"]).strip().lower()
            if version == 2
            else "ibtracs"
        )
        if target_source not in {"ibtracs", "sar_robust_peak"}:
            raise ValueError(
                f"unknown cached intensity target source {target_source!r}"
            )
        ibtracs_target_ms = (
            _finite_float(row["ibtracs_target_ms"], "ibtracs_target_ms")
            if version == 2
            else target_wind_ms
        )
        sar_target_ms = (
            _finite_float(row["sar_robust_peak_target_ms"], "sar_robust_peak_target_ms")
            if version == 2
            else math.nan
        )
        anchor_statistic = (
            str(row["anchor_statistic"]).strip().lower() if version == 2 else "max"
        )
        expected_anchor = "max" if target_source == "ibtracs" else "robust_peak"
        if anchor_statistic != expected_anchor:
            raise ValueError(
                f"cached target {target_source!r} requires {expected_anchor!r} "
                f"anchor, got {anchor_statistic!r}"
            )
        expected_target = (
            ibtracs_target_ms if target_source == "ibtracs" else sar_target_ms
        )
        if not math.isclose(
            target_wind_ms, expected_target, rel_tol=1.0e-6, abs_tol=1.0e-4
        ):
            raise ValueError(
                "cached primary intensity target disagrees with its selected reference"
            )
        filtering_counts = self.cache_metadata.get("filtering_counts", {}).get(
            self.split, {}
        )
        return {
            "wind_field": torch.from_numpy(wind),
            "valid_mask": torch.from_numpy(finite_valid),
            "distance_to_center": torch.from_numpy(distance),
            "metadata": encode_intensity_metadata(row),
            "target_wind_ms": torch.tensor(
                target_wind_ms,
                dtype=torch.float32,
            ),
            "target_category": torch.tensor(
                int(row["target_category"]), dtype=torch.long
            ),
            "raw_unet_max_wind_ms": torch.tensor(raw_max, dtype=torch.float32),
            "raw_unet_robust_peak_ms": torch.tensor(
                raw_robust_peak, dtype=torch.float32
            ),
            "ibtracs_target_ms": torch.tensor(ibtracs_target_ms, dtype=torch.float32),
            "sar_robust_peak_target_ms": torch.tensor(
                sar_target_ms, dtype=torch.float32
            ),
            "sar_max_wind_ms": torch.tensor(
                (
                    _finite_float(row["sar_max_wind_ms"], "sar_max_wind_ms")
                    if version == 2
                    else math.nan
                ),
                dtype=torch.float32,
            ),
            "is_rapid_intensification": torch.tensor(
                (
                    bool(row["is_rapid_intensification"])
                    if version == 2
                    and isinstance(row["is_rapid_intensification"], (bool, np.bool_))
                    else (
                        str(row["is_rapid_intensification"]).strip().lower()
                        in {"1", "true", "yes"}
                        if version == 2
                        else False
                    )
                ),
                dtype=torch.bool,
            ),
            "ri_24h_change_ms": torch.tensor(
                (
                    pd.to_numeric(row["ri_24h_change_ms"], errors="coerce")
                    if version == 2
                    else math.nan
                ),
                dtype=torch.float32,
            ),
            "intensity_target_source": target_source,
            "anchor_statistic": anchor_statistic,
            "intensity_filtering_counts": {
                name: torch.tensor(
                    int(filtering_counts.get(name, default)), dtype=torch.long
                )
                for name, default in (
                    ("retained", len(self)),
                    ("filtered_unbracketed", 0),
                    ("filtered_invalid_sar_center", 0),
                    ("filtered_unusable_sar", 0),
                )
            },
            "sample_weight": torch.tensor(
                _finite_float(row["_sample_weight"], "_sample_weight"),
                dtype=torch.float32,
            ),
            "sample_id": str(row["sample_id"]),
            "storm_id": str(row["storm_id"]),
            "observation_timestamp": str(row["observation_timestamp"]),
        }


def _assert_storm_disjoint(manifests: dict[str, pd.DataFrame]) -> None:
    owner: dict[str, str] = {}
    conflicts: dict[str, set[str]] = {}
    for split, frame in manifests.items():
        if "storm_id" not in frame:
            continue
        for storm_id in frame["storm_id"].astype(str).unique():
            previous = owner.setdefault(storm_id, split)
            if previous != split:
                conflicts.setdefault(storm_id, {previous}).add(split)
    if conflicts:
        detail = ", ".join(
            f"{storm}: {sorted(splits)}" for storm, splits in sorted(conflicts.items())
        )
        raise ValueError(f"intensity splits are not storm-disjoint: {detail}")


def _storm_and_category_weights(
    samples: pd.DataFrame, max_category_weight_ratio: float
) -> np.ndarray:
    if max_category_weight_ratio < 1.0:
        raise ValueError("max_category_weight_ratio must be at least one")
    category_counts = samples.groupby("target_category")["sample_id"].transform("size")
    maximum_count = float(category_counts.max())
    category_factor = np.sqrt(maximum_count / category_counts.to_numpy(dtype=float))
    category_factor = np.minimum(category_factor, max_category_weight_ratio)
    temporary = samples[["storm_id"]].copy()
    temporary["factor"] = category_factor
    storm_totals = temporary.groupby("storm_id")["factor"].transform("sum")
    weights = category_factor / storm_totals.to_numpy(dtype=float)
    return weights / weights.mean()


class UNetIntensityDataModule(pl.LightningDataModule):
    """Lightning data module for cached, single-timestep intensity fields."""

    intensity_balanced_sampling = False

    def __init__(
        self,
        root: str | Path,
        *,
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        expected_unet_checkpoint_sha256: str | None = None,
        verify_cache: bool = True,
        max_category_weight_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.expected_unet_checkpoint_sha256 = expected_unet_checkpoint_sha256
        self.verify_cache = bool(verify_cache)
        self.max_category_weight_ratio = float(max_category_weight_ratio)
        self.train_dataset: UNetIntensityDataset | None = None
        self.val_dataset: UNetIntensityDataset | None = None
        self.test_dataset: UNetIntensityDataset | None = None
        manifests = {}
        for split in {self.train_split, self.val_split, self.test_split}:
            path = self.root / split / "manifest.csv"
            if path.is_file():
                manifests[split] = pd.read_csv(path, keep_default_na=False)
        _assert_storm_disjoint(manifests)

    def _dataset(self, split: str) -> UNetIntensityDataset:
        return UNetIntensityDataset(
            self.root,
            split,
            expected_unet_checkpoint_sha256=self.expected_unet_checkpoint_sha256,
            verify_cache=self.verify_cache,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._dataset(self.train_split)
            self.val_dataset = self._dataset(self.val_split)
            self.train_dataset.samples["_sample_weight"] = _storm_and_category_weights(
                self.train_dataset.samples,
                self.max_category_weight_ratio,
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = self._dataset(self.test_split)

    @property
    def data_spec(self) -> IntensityDataSpec:
        if self.val_dataset is None:
            self.setup("fit")
        assert self.val_dataset is not None
        return self.val_dataset.data_spec

    def _loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
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
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        assert self.val_dataset is not None
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        assert self.test_dataset is not None
        return self._loader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


__all__ = [
    "BASINS",
    "INTENSITY_CACHE_SCHEMA_VERSION",
    "SUPPORTED_INTENSITY_CACHE_SCHEMA_VERSIONS",
    "INTENSITY_METADATA_NAMES",
    "IntensityDataSpec",
    "KNOT_TO_MS",
    "UNetIntensityDataModule",
    "UNetIntensityDataset",
    "category_macro_f1_tensor",
    "encode_intensity_metadata",
    "tropical_category_from_wind_ms",
    "tropical_category_from_wind_ms_tensor",
]
