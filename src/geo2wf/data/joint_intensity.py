"""Join paired wind-field samples to continuous IBTrACS intensity labels."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import replace
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from geo2wf.data.datamodule import PairedDataModule
from geo2wf.data.datasets.paired_geotiff import PairedImageDataset


INTENSITY_TARGET_COMPANION = "intensity_target"
# Legacy companion retained so older data specs/checkpoint wrappers still pass.
IBTRACS_MAX_WIND_COMPANION = "ibtracs_max_wind"
IBTRACS_STRUCTURE_COMPANION = "ibtracs_structure"
KNOT_TO_MS = 0.514444
NAUTICAL_MILE_TO_KM = 1.852
REQUIRED_IBTRACS_COLUMNS = frozenset({"USA_ATCF_ID", "ISO_TIME", "USA_WIND"})
IBTRACS_STRUCTURE_COLUMNS = frozenset(
    {
        "USA_EYE",
        "USA_RMW",
        *(
            f"USA_R{threshold}_{quadrant}"
            for threshold in (34, 50, 64)
            for quadrant in ("NE", "SE", "SW", "NW")
        ),
    }
)
IBTRACS_COMBINED_RADIUS_NAMES = ("r34", "r50", "r64")
IBTRACS_STRUCTURE_TARGET_NAMES = (
    "eye_size",
    "rmw",
    "r34_equivalent",
    "r50_equivalent",
    "r64_equivalent",
)
IBTRACS_STRUCTURE_BATCH_KEYS = (
    "ibtracs_eye_size_km",
    "ibtracs_rmw_km",
    "ibtracs_r34_equivalent_km",
    "ibtracs_r50_equivalent_km",
    "ibtracs_r64_equivalent_km",
)
INTENSITY_TARGET_SOURCES = frozenset({"ibtracs", "sar_robust_peak"})


def ibtracs_structure_targets(
    batch: Mapping[str, Any], reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``[B,5]`` structure targets and validity masks when available."""
    values = []
    masks = []
    for key in IBTRACS_STRUCTURE_BATCH_KEYS:
        valid_key = f"{key}_valid"
        if key not in batch or valid_key not in batch:
            return None
        value = torch.as_tensor(batch[key], device=reference.device).reshape(-1)
        valid = torch.as_tensor(
            batch[valid_key], device=reference.device, dtype=torch.bool
        ).reshape(-1)
        if value.shape != (reference.shape[0],) or valid.shape != value.shape:
            raise ValueError(f"{key} must have one value per sample")
        values.append(value.to(reference))
        masks.append(valid & torch.isfinite(value) & (value >= 0.0))
    return torch.stack(values, dim=1), torch.stack(masks, dim=1)


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
    selected_columns = REQUIRED_IBTRACS_COLUMNS | (
        IBTRACS_STRUCTURE_COLUMNS & available_columns
    )
    frame = pd.read_csv(
        path,
        usecols=sorted(selected_columns),
        keep_default_na=False,
        low_memory=False,
    )
    for column in IBTRACS_STRUCTURE_COLUMNS.difference(frame.columns):
        frame[column] = np.nan
    frame["storm_id"] = frame["USA_ATCF_ID"].astype(str).str.strip().str.upper()
    frame["timestamp"] = pd.to_datetime(
        frame["ISO_TIME"], errors="coerce", utc=True, format="mixed"
    )
    frame["wind_kt"] = pd.to_numeric(frame["USA_WIND"], errors="coerce")
    for column in IBTRACS_STRUCTURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
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
        lower = upper
        lower_time = upper_time = target_time
    else:
        if upper == 0 or upper == len(times):
            return None
        lower = upper - 1
        lower_time, upper_time = times[lower], times[upper]
        bracket = upper_time - lower_time
        if bracket > pd.Timedelta(hours=max_bracket_hours):
            return None
    fraction = (
        0.0
        if lower == upper
        else float((target_time - lower_time) / (upper_time - lower_time))
    )

    def interpolated(column: str) -> float:
        lower_value = float(fixes.iloc[lower][column])
        upper_value = float(fixes.iloc[upper][column])
        if not math.isfinite(lower_value) or not math.isfinite(upper_value):
            return math.nan
        value = lower_value + fraction * (upper_value - lower_value)
        return value if value >= 0.0 else math.nan

    wind_kt = interpolated("wind_kt")
    if not math.isfinite(wind_kt):
        return None
    structure_nm = {
        "eye_size": interpolated("USA_EYE"),
        "rmw": interpolated("USA_RMW"),
    }
    for radius in IBTRACS_COMBINED_RADIUS_NAMES:
        quadrant_values = [
            interpolated(f"USA_{radius.upper()}_{quadrant}")
            for quadrant in ("NE", "SE", "SW", "NW")
        ]
        # The equivalent-circle radius preserves the area of a complete
        # four-quadrant wind envelope. Partial reports remain invalid.
        structure_nm[f"{radius}_equivalent"] = (
            float(np.sqrt(np.mean(np.square(quadrant_values))))
            if all(math.isfinite(value) for value in quadrant_values)
            else math.nan
        )
    return {
        "target_wind_ms": wind_kt * KNOT_TO_MS,
        "observation_timestamp": target_time.isoformat(),
        "lower_fix_timestamp": lower_time.isoformat(),
        "upper_fix_timestamp": upper_time.isoformat(),
        **{
            f"ibtracs_{name}_km": (
                value * NAUTICAL_MILE_TO_KM if math.isfinite(value) else math.nan
            )
            for name, value in structure_nm.items()
        },
    }


def _target_tensor_and_mask(
    path: Path,
    *,
    target_size: tuple[int, int],
    center_crop_size: tuple[int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read the physical target using the paired dataset's resize/crop contract."""

    with rasterio.open(path) as source:
        masked = source.read(1, masked=True).astype("float32")
        bounds = source.bounds
    values = np.asarray(masked.filled(np.nan), dtype=np.float32)
    valid = np.isfinite(values) & ~np.ma.getmaskarray(masked)
    target = torch.from_numpy(np.where(valid, values, 0.0)).unsqueeze(0)
    mask = torch.from_numpy(valid).unsqueeze(0)
    if target.shape[-2:] != target_size:
        target = F.interpolate(
            target.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        ).squeeze(0)
        mask = (
            F.interpolate(mask.unsqueeze(0).float(), size=target_size, mode="nearest")
            .squeeze(0)
            .bool()
        )
        target = target * mask.to(target)
    tensor_bounds = torch.tensor(
        [bounds.left, bounds.right, bounds.bottom, bounds.top], dtype=torch.float64
    )
    if center_crop_size is not None:
        crop_height, crop_width = center_crop_size
        height, width = target.shape[-2:]
        if crop_height > height or crop_width > width:
            raise ValueError(
                f"center crop {center_crop_size} exceeds target shape {(height, width)}"
            )
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        target = target[:, top : top + crop_height, left : left + crop_width]
        mask = mask[:, top : top + crop_height, left : left + crop_width]
        bounds_left, bounds_right, bounds_bottom, bounds_top = tensor_bounds
        x_resolution = (bounds_right - bounds_left) / width
        y_resolution = (bounds_top - bounds_bottom) / height
        tensor_bounds = torch.stack(
            [
                bounds_left + left * x_resolution,
                bounds_left + (left + crop_width) * x_resolution,
                bounds_top - (top + crop_height) * y_resolution,
                bounds_top - top * y_resolution,
            ]
        )
    return target, mask, tensor_bounds


def _center_mask_value(
    mask: torch.Tensor,
    bounds: torch.Tensor,
    center_lat: float,
    center_lon: float,
) -> bool:
    """Return validity of the nearest raster cell containing the storm center."""

    if not math.isfinite(center_lat) or not math.isfinite(center_lon):
        return False
    height, width = mask.shape[-2:]
    left, right, bottom, top = (float(value) for value in bounds)
    if not (bottom < top and left < right):
        return False
    row = math.floor((top - center_lat) * height / (top - bottom))
    column = math.floor((center_lon - left) * width / (right - left))
    if not (0 <= row < height and 0 <= column < width):
        return False
    return bool(mask[..., row, column].all())


def _sar_intensity_diagnostics(
    paired_dataset: PairedImageDataset,
    row: pd.Series,
    *,
    robust_peak_fraction: float,
) -> dict[str, Any]:
    """Calculate center validity and scalar winds from the effective SAR crop."""

    relative_path = str(row.get("target_path") or row.get("sar_path") or "")
    if not relative_path:
        return {
            "has_valid_center": False,
            "max_wind_ms": math.nan,
            "robust_peak_ms": math.nan,
            "valid_pixels": 0,
        }
    target, mask, bounds = _target_tensor_and_mask(
        paired_dataset.root / relative_path,
        target_size=tuple(paired_dataset.target_size),
        center_crop_size=(
            tuple(paired_dataset.center_crop_size)
            if paired_dataset.center_crop_size is not None
            else None
        ),
    )
    center_lat = pd.to_numeric(row.get("ibtracs_center_lat"), errors="coerce")
    center_lon = pd.to_numeric(row.get("ibtracs_center_lon"), errors="coerce")
    center_valid = _center_mask_value(
        mask,
        bounds,
        float(center_lat),
        float(center_lon),
    )
    values = target[mask & torch.isfinite(target)]
    if not values.numel():
        return {
            "has_valid_center": center_valid,
            "max_wind_ms": math.nan,
            "robust_peak_ms": math.nan,
            "valid_pixels": 0,
        }
    count = max(1, int(math.ceil(values.numel() * robust_peak_fraction)))
    return {
        "has_valid_center": center_valid,
        "max_wind_ms": float(values.max()),
        "robust_peak_ms": float(torch.topk(values, count, sorted=False).values.mean()),
        "valid_pixels": int(values.numel()),
    }


def _ri_diagnostics(
    fixes: pd.DataFrame,
    timestamp: Any,
    *,
    current_wind_ms: float,
    max_bracket_hours: float,
    threshold_kt: float,
    window_hours: float,
) -> tuple[float, bool]:
    target_time = pd.Timestamp(timestamp)
    prior = _interpolate_ibtracs_wind(
        fixes,
        target_time - pd.Timedelta(hours=window_hours),
        max_bracket_hours=max_bracket_hours,
    )
    if prior is None:
        return math.nan, False
    change_ms = current_wind_ms - float(prior["target_wind_ms"])
    threshold_ms = threshold_kt * KNOT_TO_MS
    is_ri = change_ms > threshold_ms or math.isclose(
        change_ms, threshold_ms, rel_tol=1.0e-12, abs_tol=1.0e-9
    )
    return change_ms, is_ri


class JointPairedIntensityDataset(Dataset):
    """Attach matched IBTrACS and SAR scalar intensity references."""

    def __init__(
        self,
        paired_dataset: PairedImageDataset,
        ibtracs_tracks: dict[str, pd.DataFrame],
        split: str,
        *,
        max_bracket_hours: float = 3.0,
        intensity_target_source: str = "ibtracs",
        require_sar_valid_center: bool = False,
        sar_robust_peak_fraction: float = 0.005,
        ri_threshold_kt: float = 30.0,
        ri_window_hours: float = 24.0,
    ) -> None:
        self.paired_dataset = paired_dataset
        self.split = str(split)
        self.max_bracket_hours = float(max_bracket_hours)
        if self.max_bracket_hours <= 0:
            raise ValueError("max_bracket_hours must be positive")
        self.intensity_target_source = str(intensity_target_source).strip().lower()
        if self.intensity_target_source not in INTENSITY_TARGET_SOURCES:
            raise ValueError(
                "intensity_target_source must be one of "
                f"{sorted(INTENSITY_TARGET_SOURCES)}, got {intensity_target_source!r}"
            )
        self.require_sar_valid_center = bool(require_sar_valid_center)
        self.sar_robust_peak_fraction = float(sar_robust_peak_fraction)
        self.ri_threshold_kt = float(ri_threshold_kt)
        self.ri_window_hours = float(ri_window_hours)
        if not 0.0 < self.sar_robust_peak_fraction <= 1.0:
            raise ValueError("sar_robust_peak_fraction must be in (0, 1]")
        if self.ri_threshold_kt <= 0.0 or self.ri_window_hours <= 0.0:
            raise ValueError("RI threshold and window must be positive")

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
        self.filtered_unbracketed_count = 0
        self.filtered_invalid_sar_center_count = 0
        self.filtered_unusable_sar_count = 0
        for index, row in paired_samples.iterrows():
            storm_id = str(row["storm_id"]).strip().upper()
            fixes = ibtracs_tracks.get(storm_id)
            if fixes is None:
                self.filtered_unbracketed_count += 1
                continue
            label = _interpolate_ibtracs_wind(
                fixes,
                row["target_timestamp"],
                max_bracket_hours=self.max_bracket_hours,
            )
            if label is None:
                self.filtered_unbracketed_count += 1
                continue
            sar = _sar_intensity_diagnostics(
                paired_dataset,
                row,
                robust_peak_fraction=self.sar_robust_peak_fraction,
            )
            if not math.isfinite(float(sar["robust_peak_ms"])):
                self.filtered_unusable_sar_count += 1
                continue
            if self.require_sar_valid_center and not sar["has_valid_center"]:
                self.filtered_invalid_sar_center_count += 1
                continue
            ibtracs_wind_ms = float(label["target_wind_ms"])
            ri_change_ms, is_ri = _ri_diagnostics(
                fixes,
                label["observation_timestamp"],
                current_wind_ms=ibtracs_wind_ms,
                max_bracket_hours=self.max_bracket_hours,
                threshold_kt=self.ri_threshold_kt,
                window_hours=self.ri_window_hours,
            )
            target_wind_ms = (
                ibtracs_wind_ms
                if self.intensity_target_source == "ibtracs"
                else float(sar["robust_peak_ms"])
            )
            label.update(
                {
                    "source_sample_id": str(row["sample_id"]),
                    "storm_id": storm_id,
                    "target_wind_ms": target_wind_ms,
                    "ibtracs_wind_ms": ibtracs_wind_ms,
                    "sar_max_wind_ms": float(sar["max_wind_ms"]),
                    "sar_robust_peak_ms": float(sar["robust_peak_ms"]),
                    "sar_valid_pixels": int(sar["valid_pixels"]),
                    "sar_has_valid_center": bool(sar["has_valid_center"]),
                    "intensity_target_source": self.intensity_target_source,
                    "ri_24h_change_ms": ri_change_ms,
                    "is_rapid_intensification": is_ri,
                }
            )
            self._paired_indices.append(int(index))
            self._labels.append(label)

        self.samples = paired_samples.iloc[self._paired_indices].reset_index(drop=True)

    @property
    def root(self) -> Path:
        return self.paired_dataset.root

    @property
    def data_spec(self):
        spec = self.paired_dataset.data_spec
        return replace(
            spec,
            companions=spec.companions
            | {
                INTENSITY_TARGET_COMPANION,
                IBTRACS_MAX_WIND_COMPANION,
                IBTRACS_STRUCTURE_COMPANION,
            },
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
        sample["ibtracs_target_ms"] = torch.tensor(
            float(label["ibtracs_wind_ms"]), dtype=torch.float32
        )
        for name in (
            "eye_size",
            "rmw",
            "r34_equivalent",
            "r50_equivalent",
            "r64_equivalent",
        ):
            key = f"ibtracs_{name}_km"
            value = float(label.get(key, math.nan))
            sample[key] = torch.tensor(value, dtype=torch.float32)
            sample[f"{key}_valid"] = torch.tensor(
                math.isfinite(value), dtype=torch.bool
            )
        sample["sar_robust_peak_target_ms"] = torch.tensor(
            float(label["sar_robust_peak_ms"]), dtype=torch.float32
        )
        sample["sar_max_wind_ms"] = torch.tensor(
            float(label["sar_max_wind_ms"]), dtype=torch.float32
        )
        sample["sar_valid_pixels"] = torch.tensor(
            int(label["sar_valid_pixels"]), dtype=torch.long
        )
        sample["sar_has_valid_center"] = torch.tensor(
            bool(label["sar_has_valid_center"]), dtype=torch.bool
        )
        sample["is_rapid_intensification"] = torch.tensor(
            bool(label["is_rapid_intensification"]), dtype=torch.bool
        )
        sample["ri_24h_change_ms"] = torch.tensor(
            float(label["ri_24h_change_ms"]), dtype=torch.float32
        )
        sample["intensity_target_source"] = str(label["intensity_target_source"])
        sample["intensity_filtering_counts"] = {
            "retained": torch.tensor(len(self), dtype=torch.long),
            "filtered_unbracketed": torch.tensor(
                self.filtered_unbracketed_count, dtype=torch.long
            ),
            "filtered_invalid_sar_center": torch.tensor(
                self.filtered_invalid_sar_center_count, dtype=torch.long
            ),
            "filtered_unusable_sar": torch.tensor(
                self.filtered_unusable_sar_count, dtype=torch.long
            ),
        }
        sample["intensity_observation_timestamp"] = str(label["observation_timestamp"])
        return sample


class JointPairedIntensityDataModule(PairedDataModule):
    """Pair SAR fields with matched scalar intensity references."""

    def __init__(
        self,
        *args: Any,
        ibtracs_file: str | Path,
        max_ibtracs_bracket_hours: float = 3.0,
        intensity_target_source: str = "ibtracs",
        require_sar_valid_center: bool = False,
        sar_robust_peak_fraction: float = 0.005,
        ri_threshold_kt: float = 30.0,
        ri_window_hours: float = 24.0,
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
        self.intensity_target_source = str(intensity_target_source).strip().lower()
        self.require_sar_valid_center = bool(require_sar_valid_center)
        self.sar_robust_peak_fraction = float(sar_robust_peak_fraction)
        self.ri_threshold_kt = float(ri_threshold_kt)
        self.ri_window_hours = float(ri_window_hours)
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
            intensity_target_source=self.intensity_target_source,
            require_sar_valid_center=self.require_sar_valid_center,
            sar_robust_peak_fraction=self.sar_robust_peak_fraction,
            ri_threshold_kt=self.ri_threshold_kt,
            ri_window_hours=self.ri_window_hours,
        )


__all__ = [
    "IBTRACS_MAX_WIND_COMPANION",
    "INTENSITY_TARGET_COMPANION",
    "INTENSITY_TARGET_SOURCES",
    "JointPairedIntensityDataModule",
    "JointPairedIntensityDataset",
    "_center_mask_value",
    "_ri_diagnostics",
    "_sar_intensity_diagnostics",
]
