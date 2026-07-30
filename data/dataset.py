from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IBTRACS_CENTER_COLUMNS = ("ibtracs_center_lat", "ibtracs_center_lon")
DISTANCE_TO_IBTRACS_CENTER = "distance_to_ibtracs_center"
LOCAL_SOLAR_TIME_SIN = "local_solar_time_sin"
LOCAL_SOLAR_TIME_COS = "local_solar_time_cos"
SOLAR_ZENITH_ANGLE = "solar_zenith_angle"
SOLAR_TIME_CHANNELS = (
    LOCAL_SOLAR_TIME_SIN,
    LOCAL_SOLAR_TIME_COS,
    SOLAR_ZENITH_ANGLE,
)

EARTH_RADIUS_M = 6_371_000.0
ERA5_U10_CHANNELS = {"era5_u_wind_10m", "u_wind_10m", "era5_u10", "u10"}
ERA5_V10_CHANNELS = {"era5_v_wind_10m", "v_wind_10m", "era5_v10", "v10"}
ERA5_WIND_SPEED_10M = "era5_wind_speed_10m"
ERA5_RELATIVE_VORTICITY_10M = "era5_relative_vorticity_10m"
NORMALIZATION_MIN_MAX = "min-max"
NORMALIZATION_ROBUST_ZSCORE = "robust-zscore"
DEFAULT_ROBUST_CLIP = 4.0
DEFAULT_CHANNEL_STATS = {
    ("era5", ERA5_WIND_SPEED_10M): {
        "min": 0.0,
        "max": 85.0,
        "median": 10.0,
        "robust_scale": 5.0,
    },
    ("era5", ERA5_RELATIVE_VORTICITY_10M): {
        "min": -0.005,
        "max": 0.005,
        "median": 0.0,
        "robust_scale": 1.0e-4,
    },
}


class PairedImageDataset(Dataset):
    """Read exported paired GeoTIFF samples from a split manifest."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        stats_file: str | Path | None = None,
        target_size: tuple[int, int] = (256, 256),
        center_crop_size: tuple[int, int] | None = None,
        augment: bool = False,
        require_era5: bool = False,
        normalization: str | None = None,
        target_normalization: str | None = None,
        robust_clip: float = DEFAULT_ROBUST_CLIP,
        target_robust_clip: float | None = None,
        max_era5_time_gap_hours: float | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.split = split
        self.manifest_file = self.root / split / "manifest.csv"
        if not self.manifest_file.exists():
            raise FileNotFoundError(
                f"GeoTIFF manifest does not exist: {self.manifest_file}"
            )
        self.samples = pd.read_csv(self.manifest_file, keep_default_na=False)
        self.manifest_sample_count = len(self.samples)
        self.require_era5 = require_era5
        if self.require_era5:
            self.samples = self.samples.loc[
                _manifest_has_era5(self.samples)
            ].reset_index(drop=True)
        self.filtered_missing_era5_count = (
            self.manifest_sample_count - len(self.samples)
        )
        self.max_era5_time_gap_hours = max_era5_time_gap_hours
        self.filtered_stale_era5_count = 0
        if max_era5_time_gap_hours is not None:
            if max_era5_time_gap_hours <= 0:
                raise ValueError("max_era5_time_gap_hours must be positive")
            gap_hours = _manifest_era5_time_gap_hours(self.samples)
            keep = gap_hours.notna() & (
                gap_hours <= float(max_era5_time_gap_hours)
            )
            self.filtered_stale_era5_count = int((~keep).sum())
            self.samples = self.samples.loc[keep].reset_index(drop=True)
        self.stats_file = (
            Path(stats_file).expanduser()
            if stats_file is not None
            else self.root / "stats.json"
        )
        if not self.stats_file.exists():
            raise FileNotFoundError(f"Stats file does not exist: {self.stats_file}")
        self.stats = json.loads(self.stats_file.read_text(encoding="utf-8"))
        self.target_size = target_size
        self.center_crop_size = center_crop_size
        self.augment = augment
        self.normalization = _normalization_method(self.stats, normalization)
        self.target_normalization = _normalization_method(
            self.stats,
            (
                self.normalization
                if target_normalization is None
                else target_normalization
            ),
        )
        if robust_clip <= 0:
            raise ValueError("robust_clip must be positive")
        self.robust_clip = float(robust_clip)
        self.target_robust_clip = (
            self.robust_clip
            if target_robust_clip is None
            else float(target_robust_clip)
        )
        if self.target_robust_clip <= 0:
            raise ValueError("target_robust_clip must be positive")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples.iloc[idx]
        storm_center = _manifest_ibtracs_center(row)
        condition_source_type = _row_value(row, "condition_source_type", "geo")
        target_source_type = _row_value(row, "target_source_type", "sar")
        condition_channels = _json_list(
            _row_value(row, "condition_channels", row.get("geo_channels"))
        )
        target_channels = _json_list(
            _row_value(row, "target_channels", row.get("sar_channels"))
        )
        condition_path = _row_value(row, "condition_path", row.get("geo_path"))
        target_path = _row_value(row, "target_path", row.get("sar_path"))
        context_path = _row_value(row, "context_path", row.get("era5_path", ""))
        context_source_type = _row_value(
            row, "context_source_type", "era5" if context_path else ""
        )
        context_channels = _json_list(
            _row_value(row, "context_channels", row.get("era5_channels", "[]"))
        )

        condition, condition_mask, condition_bounds = _read_geotiff(
            self.root / condition_path,
            reject_all_zero_fill=True,
        )
        condition_channel_mask = condition_mask.expand(
            len(condition_channels), -1, -1
        )
        context = None
        context_mask = None
        era5_wind_speed_physical = None
        era5_wind_speed_mask = None
        if context_path:
            context, context_mask, context_bounds = _read_geotiff(
                self.root / context_path
            )
            if context_source_type == "era5":
                context, context_channels = _append_era5_derived_channels(
                    context, context_channels, context_bounds
                )
                context_mask = context_mask & context.isfinite().all(
                    dim=0, keepdim=True
                )
                speed_index = _find_channel_index(
                    context_channels, {ERA5_WIND_SPEED_10M}
                )
                if speed_index is not None:
                    era5_wind_speed_physical = context[
                        speed_index : speed_index + 1
                    ].clone()
                    era5_wind_speed_mask = (
                        context_mask
                        & era5_wind_speed_physical.isfinite().all(
                            dim=0, keepdim=True
                        )
                    )
        target, target_mask, target_bounds = _read_geotiff(self.root / target_path)
        target_physical = torch.nan_to_num(
            target, nan=0.0, posinf=0.0, neginf=0.0
        ) * target_mask.to(target.dtype)
        target_norm_offset, target_norm_scale = _normalization_affine_parameters(
            target_source_type,
            target_channels,
            self.stats,
            normalization=self.target_normalization,
            robust_clip=self.target_robust_clip,
        )
        condition_zero_values = _normalized_physical_zero(
            condition_source_type,
            condition_channels,
            self.stats,
            normalization=self.normalization,
            robust_clip=self.robust_clip,
        )
        condition = _normalize(
            condition,
            condition_source_type,
            condition_channels,
            self.stats,
            normalization=self.normalization,
            robust_clip=self.robust_clip,
        )
        if context is not None:
            context_zero_values = _normalized_physical_zero(
                context_source_type,
                context_channels,
                self.stats,
                normalization=self.normalization,
                robust_clip=self.robust_clip,
            )
            context = _normalize(
                context,
                context_source_type,
                context_channels,
                self.stats,
                normalization=self.normalization,
                robust_clip=self.robust_clip,
            )
        target = _normalize(
            target,
            target_source_type,
            target_channels,
            self.stats,
            normalization=self.target_normalization,
            robust_clip=self.target_robust_clip,
        )
        condition = torch.nan_to_num(condition, nan=0.0)
        if context is not None:
            context = torch.nan_to_num(context, nan=0.0)
        target = torch.nan_to_num(target, nan=0.0)
        condition = condition * condition_mask.to(condition.dtype)
        if context is not None and context_mask is not None:
            context = context * context_mask.to(context.dtype)
            condition = torch.cat([condition, context], dim=0)
            condition_channel_mask = torch.cat(
                [
                    condition_channel_mask,
                    context_mask.expand(len(context_channels), -1, -1),
                ],
                dim=0,
            )
            condition_mask = condition_mask & context_mask
            condition_channels = condition_channels + context_channels
            condition_zero_values = torch.cat(
                [condition_zero_values, context_zero_values]
            )
        target = target * target_mask.to(target.dtype)
        original_target_mask = target_mask
        target, target_mask = _resize_target(
            target, original_target_mask, self.target_size
        )
        target_physical, _ = _resize_target(
            target_physical, original_target_mask, self.target_size
        )
        era5_wind_speed = None
        if (
            era5_wind_speed_physical is not None
            and era5_wind_speed_mask is not None
            and target_source_type == "sar"
            and "wind_speed" in target_channels
        ):
            era5_wind_speed_physical = torch.nan_to_num(
                era5_wind_speed_physical,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ) * era5_wind_speed_mask.to(era5_wind_speed_physical.dtype)
            era5_wind_speed = _normalize(
                era5_wind_speed_physical,
                "sar",
                ["wind_speed"],
                self.stats,
                normalization=self.target_normalization,
                robust_clip=self.target_robust_clip,
            ) * era5_wind_speed_mask.to(era5_wind_speed_physical.dtype)
            original_era5_mask = era5_wind_speed_mask
            era5_wind_speed, era5_wind_speed_mask = _resize_target(
                era5_wind_speed, original_era5_mask, self.target_size
            )
            era5_wind_speed_physical, _ = _resize_target(
                era5_wind_speed_physical,
                original_era5_mask,
                self.target_size,
            )
        if self.center_crop_size is not None:
            condition_shape = condition.shape[-2:]
            target_shape = target.shape[-2:]
            condition = _center_crop(condition, self.center_crop_size)
            condition_mask = _center_crop(condition_mask, self.center_crop_size)
            condition_channel_mask = _center_crop(
                condition_channel_mask, self.center_crop_size
            )
            target = _center_crop(target, self.center_crop_size)
            target_mask = _center_crop(target_mask, self.center_crop_size)
            target_physical = _center_crop(target_physical, self.center_crop_size)
            condition_bounds = _center_crop_bounds(
                condition_bounds, condition_shape, self.center_crop_size
            )
            target_bounds = _center_crop_bounds(
                target_bounds, target_shape, self.center_crop_size
            )
            if era5_wind_speed is not None:
                era5_wind_speed = _center_crop(
                    era5_wind_speed, self.center_crop_size
                )
                era5_wind_speed_physical = _center_crop(
                    era5_wind_speed_physical, self.center_crop_size
                )
                era5_wind_speed_mask = _center_crop(
                    era5_wind_speed_mask, self.center_crop_size
                )
        distance_to_center = _normalized_distance_to_center(
            condition_bounds,
            condition.shape[-2:],
            storm_center,
        )
        solar_time = _solar_time_features(
            condition_bounds,
            condition.shape[-2:],
            _condition_timestamp(row),
        )
        condition = torch.cat(
            [condition, distance_to_center, solar_time], dim=0
        )
        condition_channels = condition_channels + [
            DISTANCE_TO_IBTRACS_CENTER,
            *SOLAR_TIME_CHANNELS,
        ]
        condition_zero_values = torch.cat(
            [condition_zero_values, condition_zero_values.new_zeros(4)]
        )
        condition_channel_mask = torch.cat(
            [
                condition_channel_mask,
                torch.ones_like(distance_to_center, dtype=torch.bool),
                torch.ones_like(solar_time, dtype=torch.bool),
            ],
            dim=0,
        )
        if self.augment:
            flip_state = _random_flip_state()
            condition, target, condition_mask, target_mask = _paired_random_flips(
                condition,
                target,
                condition_mask,
                target_mask,
                condition_channels=condition_channels,
                condition_zero_values=condition_zero_values,
                condition_channel_mask=condition_channel_mask,
                flip_state=flip_state,
            )
            flip_dims = _flip_dims(flip_state)
            if flip_dims:
                target_physical = torch.flip(target_physical, dims=flip_dims)
                if era5_wind_speed is not None:
                    era5_wind_speed = torch.flip(era5_wind_speed, dims=flip_dims)
                    era5_wind_speed_physical = torch.flip(
                        era5_wind_speed_physical, dims=flip_dims
                    )
                    era5_wind_speed_mask = torch.flip(
                        era5_wind_speed_mask, dims=flip_dims
                    )
        sample = {
            "condition": condition,
            "target": target,
            "target_physical": target_physical,
            "target_norm_offset": target_norm_offset,
            "target_norm_scale": target_norm_scale,
            "condition_mask": condition_mask,
            "target_mask": target_mask,
            "condition_bounds": condition_bounds,
            "target_bounds": target_bounds,
            "center": storm_center,
            "sample_id": str(row["sample_id"]),
            "meta": {
                "storm_id": str(row["storm_id"]),
                "condition_source_type": condition_source_type,
                "target_source_type": target_source_type,
                "condition_sensor": _row_value(
                    row, "condition_sensor", row.get("geo_sensor", "")
                ),
                "target_sensor": _row_value(
                    row, "target_sensor", row.get("sar_sensor", "")
                ),
                "dt_minutes": float(row["dt_minutes"]),
                "condition_channels": condition_channels,
                "context_source_type": context_source_type,
                "context_channels": context_channels,
                "target_channels": target_channels,
            },
        }
        if era5_wind_speed is not None:
            sample.update(
                {
                    "era5_wind_speed": era5_wind_speed,
                    "era5_wind_speed_physical": era5_wind_speed_physical,
                    "era5_wind_speed_mask": era5_wind_speed_mask,
                }
            )
        return sample


def _resize_target(
    target: torch.Tensor,
    target_mask: torch.Tensor,
    size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize a target and its validity mask to a fixed spatial size."""
    if target.shape[-2:] == size:
        return target, target_mask
    target = F.interpolate(
        target.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    ).squeeze(0)
    target_mask = F.interpolate(
        target_mask.unsqueeze(0).float(), size=size, mode="nearest"
    ).squeeze(0).bool()
    return target * target_mask.to(target.dtype), target_mask

def _center_crop(
    tensor: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """Crop the same centered spatial window from a CHW tensor."""
    crop_height, crop_width = size
    height, width = tensor.shape[-2:]
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError("center crop dimensions must be positive")
    if crop_height > height or crop_width > width:
        raise ValueError(
            f"center crop {size} exceeds tensor spatial shape {(height, width)}"
        )
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return tensor[..., top : top + crop_height, left : left + crop_width]


def _center_crop_bounds(
    bounds: torch.Tensor,
    original_size: tuple[int, int],
    crop_size: tuple[int, int],
) -> torch.Tensor:
    """Update [left, right, bottom, top] bounds for a centered pixel crop."""
    height, width = original_size
    crop_height, crop_width = crop_size
    if crop_height > height or crop_width > width:
        raise ValueError(
            f"center crop {crop_size} exceeds spatial shape {original_size}"
        )
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    bounds_left, bounds_right, bounds_bottom, bounds_top = bounds.unbind()
    x_resolution = (bounds_right - bounds_left) / width
    y_resolution = (bounds_top - bounds_bottom) / height
    return torch.stack(
        [
            bounds_left + left * x_resolution,
            bounds_left + (left + crop_width) * x_resolution,
            bounds_top - (top + crop_height) * y_resolution,
            bounds_top - top * y_resolution,
        ]
    )


def _normalized_distance_to_center(
    bounds: torch.Tensor,
    size: tuple[int, int],
    center: torch.Tensor,
) -> torch.Tensor:
    """Return great-circle distance to an IBTrACS center, scaled to [0, 1]."""
    height, width = size
    if height <= 0 or width <= 0:
        raise ValueError("distance raster dimensions must be positive")
    if center.numel() != 2 or not torch.isfinite(center).all():
        raise ValueError(
            "a finite IBTrACS latitude/longitude center is required to build "
            "the distance-to-center input"
        )

    left, right, bottom, top = bounds.to(dtype=torch.float64).unbind()
    center_latitude, center_longitude = center.to(dtype=torch.float64).unbind()
    row = torch.arange(height, dtype=torch.float64, device=bounds.device) + 0.5
    column = torch.arange(width, dtype=torch.float64, device=bounds.device) + 0.5
    latitude = top - row[:, None] * (top - bottom) / height
    longitude = left + column[None, :] * (right - left) / width

    latitude_radians = torch.deg2rad(latitude)
    center_latitude_radians = torch.deg2rad(center_latitude)
    delta_latitude = latitude_radians - center_latitude_radians
    delta_longitude = torch.deg2rad(
        torch.remainder(longitude - center_longitude + 180.0, 360.0) - 180.0
    )
    haversine = (
        torch.sin(delta_latitude / 2.0).square()
        + torch.cos(latitude_radians)
        * torch.cos(center_latitude_radians)
        * torch.sin(delta_longitude / 2.0).square()
    ).clamp(0.0, 1.0)
    angular_distance = 2.0 * torch.atan2(
        torch.sqrt(haversine),
        torch.sqrt((1.0 - haversine).clamp_min(0.0)),
    )
    max_distance = angular_distance.max()
    if max_distance > torch.finfo(angular_distance.dtype).eps:
        angular_distance = angular_distance / max_distance
    else:
        angular_distance = torch.zeros_like(angular_distance)
    return angular_distance.to(dtype=torch.float32).unsqueeze(0)


def _condition_timestamp(row: pd.Series) -> pd.Timestamp:
    """Return the UTC GEO acquisition time used to construct the condition."""
    for column in ("condition_timestamp", "geo_timestamp"):
        value = _row_value(row, column, "")
        if value:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                continue
            if timestamp.tzinfo is None:
                return timestamp.tz_localize("UTC")
            return timestamp.tz_convert("UTC")
    raise ValueError(
        "manifest row is missing a valid condition_timestamp/geo_timestamp; "
        "solar-time condition channels cannot be constructed"
    )


def _solar_time_features(
    bounds: torch.Tensor,
    shape: tuple[int, int],
    timestamp: pd.Timestamp,
) -> torch.Tensor:
    """Return local-solar-time sine/cosine and normalized solar zenith.

    Local solar time includes the equation-of-time correction and varies with
    pixel longitude. Solar zenith is divided by pi, giving a [0, 1] channel.
    """
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("solar feature raster dimensions must be positive")
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    left, right, bottom, top = (float(value) for value in bounds)
    longitudes = torch.from_numpy(
        _cell_centers(left, right, width)
    ).to(torch.float64)
    latitudes = torch.from_numpy(
        _cell_centers(top, bottom, height)
    ).to(torch.float64)
    latitude_grid, longitude_grid = torch.meshgrid(
        latitudes, longitudes, indexing="ij"
    )

    utc_minutes = (
        timestamp.hour * 60.0
        + timestamp.minute
        + timestamp.second / 60.0
        + timestamp.microsecond / 60_000_000.0
    )
    days_in_year = 366.0 if timestamp.is_leap_year else 365.0
    fractional_year = (
        2.0
        * np.pi
        / days_in_year
        * (timestamp.dayofyear - 1.0 + (utc_minutes / 60.0 - 12.0) / 24.0)
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(fractional_year)
        - 0.032077 * np.sin(fractional_year)
        - 0.014615 * np.cos(2.0 * fractional_year)
        - 0.040849 * np.sin(2.0 * fractional_year)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(fractional_year)
        + 0.070257 * np.sin(fractional_year)
        - 0.006758 * np.cos(2.0 * fractional_year)
        + 0.000907 * np.sin(2.0 * fractional_year)
        - 0.002697 * np.cos(3.0 * fractional_year)
        + 0.00148 * np.sin(3.0 * fractional_year)
    )

    solar_minutes = torch.remainder(
        utc_minutes + equation_of_time + 4.0 * longitude_grid,
        1440.0,
    )
    solar_phase = solar_minutes * (2.0 * np.pi / 1440.0)
    hour_angle = solar_minutes * (np.pi / 720.0) - np.pi
    latitude_radians = torch.deg2rad(latitude_grid)
    cosine_zenith = (
        torch.sin(latitude_radians) * np.sin(declination)
        + torch.cos(latitude_radians)
        * np.cos(declination)
        * torch.cos(hour_angle)
    ).clamp(-1.0, 1.0)
    normalized_zenith = torch.acos(cosine_zenith) / np.pi

    return torch.stack(
        [
            torch.sin(solar_phase),
            torch.cos(solar_phase),
            normalized_zenith,
        ]
    ).to(torch.float32)


def _paired_random_flips(
    condition: torch.Tensor,
    target: torch.Tensor,
    condition_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    condition_channels: list[str] | None = None,
    condition_zero_values: torch.Tensor | None = None,
    condition_channel_mask: torch.Tensor | None = None,
    flip_state: tuple[bool, bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flip a pair, transforming ERA5 vector and pseudoscalar channels."""
    if flip_state is None:
        flip_state = _random_flip_state()
    flip_dims = _flip_dims(flip_state)
    if flip_dims:
        condition = torch.flip(condition, dims=flip_dims)
        target = torch.flip(target, dims=flip_dims)
        condition_mask = torch.flip(condition_mask, dims=flip_dims)
        target_mask = torch.flip(target_mask, dims=flip_dims)
        if condition_channel_mask is not None:
            condition_channel_mask = torch.flip(
                condition_channel_mask, dims=flip_dims
            )
        if condition_channels is not None:
            if condition_zero_values is None:
                raise ValueError(
                    "condition_zero_values are required for physics-aware flips"
                )
            condition = _transform_flipped_era5_channels(
                condition,
                condition_channels,
                condition_zero_values,
                flip_state,
                valid_mask=(
                    condition_channel_mask
                    if condition_channel_mask is not None
                    else condition_mask
                ),
            )
    return condition, target, condition_mask, target_mask


def _random_flip_state() -> tuple[bool, bool]:
    """Return ``(horizontal, vertical)`` random flip decisions."""
    return bool(torch.rand(()) < 0.5), bool(torch.rand(()) < 0.5)


def _flip_dims(flip_state: tuple[bool, bool]) -> list[int]:
    horizontal, vertical = flip_state
    dims = []
    if horizontal:
        dims.append(-1)
    if vertical:
        dims.append(-2)
    return dims


def _transform_flipped_era5_channels(
    condition: torch.Tensor,
    channels: list[str],
    normalized_zero: torch.Tensor,
    flip_state: tuple[bool, bool],
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply vector/reflection parity after the spatial tensor flip."""
    if len(channels) != condition.shape[0] or normalized_zero.numel() != len(channels):
        raise ValueError("condition channel metadata does not match the tensor")
    horizontal, vertical = flip_state
    reflected_once = horizontal != vertical
    vorticity_names = {
        ERA5_RELATIVE_VORTICITY_10M,
        "relative_vorticity_10m",
    }
    for index, channel in enumerate(channels):
        negate = (
            (horizontal and channel in ERA5_U10_CHANNELS)
            or (vertical and channel in ERA5_V10_CHANNELS)
            or (reflected_once and channel in vorticity_names)
        )
        if negate:
            zero = normalized_zero[index].to(
                device=condition.device, dtype=condition.dtype
            )
            reflected = (2.0 * zero - condition[index]).clamp(0.0, 1.0)
            if valid_mask is not None:
                if valid_mask.shape[0] == condition.shape[0]:
                    pixel_mask = valid_mask[index].bool()
                else:
                    pixel_mask = valid_mask.all(dim=0)
                reflected = torch.where(pixel_mask, reflected, condition[index])
            condition[index] = reflected
    return condition


def _read_geotiff(
    path: Path,
    *,
    reject_all_zero_fill: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with rasterio.open(path) as dataset:
        array = dataset.read().astype("float32")
        mask = dataset.dataset_mask() > 0
        bounds = torch.tensor([dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top], dtype=torch.float64)
    finite_mask = torch.from_numpy(array).isfinite().all(dim=0)
    tensor = torch.from_numpy(array)
    mask_tensor = torch.from_numpy(mask).bool() & finite_mask
    if reject_all_zero_fill:
        mask_tensor = mask_tensor & ~torch.isclose(
            tensor,
            torch.zeros((), dtype=tensor.dtype),
        ).all(dim=0)
    mask_tensor = mask_tensor.unsqueeze(0)
    return tensor, mask_tensor, bounds


def _normalize(
    tensor: torch.Tensor,
    source_type: str,
    channels: list[str],
    stats: dict[str, Any],
    *,
    normalization: str | None = None,
    robust_clip: float = DEFAULT_ROBUST_CLIP,
) -> torch.Tensor:
    offset, scale = _normalization_affine_parameters(
        source_type,
        channels,
        stats,
        normalization=normalization,
        robust_clip=robust_clip,
    )
    shape = (-1,) + (1,) * (tensor.ndim - 1)
    offset = offset.to(device=tensor.device, dtype=torch.float32).view(shape)
    scale = scale.to(device=tensor.device, dtype=torch.float32).view(shape)
    return ((tensor.float() - offset) / scale).clamp(0.0, 1.0)


def _denormalize(
    tensor: torch.Tensor,
    source_type: str,
    channels: list[str],
    stats: dict[str, Any],
    *,
    normalization: str | None = None,
    robust_clip: float = DEFAULT_ROBUST_CLIP,
) -> torch.Tensor:
    """Invert the affine normalization (for values before clipping)."""
    offset, scale = _normalization_affine_parameters(
        source_type,
        channels,
        stats,
        normalization=normalization,
        robust_clip=robust_clip,
    )
    shape = (-1,) + (1,) * (tensor.ndim - 1)
    offset = offset.to(device=tensor.device, dtype=tensor.dtype).view(shape)
    scale = scale.to(device=tensor.device, dtype=tensor.dtype).view(shape)
    return tensor * scale + offset


def _normalization_affine_parameters(
    source_type: str,
    channels: list[str],
    stats: dict[str, Any],
    *,
    normalization: str | None = None,
    robust_clip: float = DEFAULT_ROBUST_CLIP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-channel ``offset`` and ``scale`` for a train-stat affine map."""
    method = _normalization_method(stats, normalization)
    if robust_clip <= 0:
        raise ValueError("robust_clip must be positive")
    offsets = []
    scales = []
    for index, channel in enumerate(channels):
        channel_stats = _channel_stats(
            source_type, channel, index, stats, normalization=method
        )
        if method == NORMALIZATION_MIN_MAX:
            offset = float(channel_stats["min"])
            scale = float(channel_stats["max"]) - offset
        else:
            center_value = channel_stats.get("median", channel_stats.get("mean"))
            spread_value = channel_stats.get(
                "robust_scale", channel_stats.get("std")
            )
            if center_value is None or spread_value is None:
                raise KeyError(
                    f"Missing robust train stats for {source_type}:{channel}"
                )
            center = float(center_value)
            spread = float(spread_value)
            offset = center - robust_clip * spread
            scale = 2.0 * robust_clip * spread
        offsets.append(offset)
        scales.append(max(scale, 1e-6))
    return torch.tensor(offsets, dtype=torch.float32), torch.tensor(
        scales, dtype=torch.float32
    )


def _normalized_physical_zero(
    source_type: str,
    channels: list[str],
    stats: dict[str, Any],
    *,
    normalization: str | None = None,
    robust_clip: float = DEFAULT_ROBUST_CLIP,
) -> torch.Tensor:
    offset, scale = _normalization_affine_parameters(
        source_type,
        channels,
        stats,
        normalization=normalization,
        robust_clip=robust_clip,
    )
    return -offset / scale


def _normalization_method(
    stats: dict[str, Any], normalization: str | None = None
) -> str:
    requested = str(normalization or stats.get("normalization", NORMALIZATION_MIN_MAX))
    requested = requested.strip().lower().replace("_", "-")
    aliases = {
        "minmax": NORMALIZATION_MIN_MAX,
        "min-max": NORMALIZATION_MIN_MAX,
        "robust": NORMALIZATION_ROBUST_ZSCORE,
        "robust-zscore": NORMALIZATION_ROBUST_ZSCORE,
        "clipped-zscore": NORMALIZATION_ROBUST_ZSCORE,
    }
    try:
        return aliases[requested]
    except KeyError as error:
        raise ValueError(
            f"Unsupported normalization {requested!r}; choose min-max or robust-zscore"
        ) from error


def _channel_stats(
    source_type: str,
    channel: str,
    index: int,
    stats: dict[str, Any],
    *,
    normalization: str,
) -> dict[str, Any]:
    channels_by_source = stats.get("channels", {})
    if (
        normalization == NORMALIZATION_ROBUST_ZSCORE
        and source_type == "era5"
        and channel == ERA5_WIND_SPEED_10M
    ):
        target_stats = channels_by_source.get("sar", {}).get("wind_speed")
        if target_stats is not None:
            return target_stats
    stats_by_channel = channels_by_source.get(source_type, {})
    channel_stats = stats_by_channel.get(channel)
    if channel_stats is None:
        channel_stats = stats_by_channel.get(f"band_{index}")
    if channel_stats is None:
        channel_stats = DEFAULT_CHANNEL_STATS.get((source_type, channel))
    if channel_stats is None:
        raise KeyError(f"Missing stats for {source_type}:{channel}")
    return channel_stats


def _append_era5_derived_channels(
    tensor: torch.Tensor,
    channels: list[str],
    bounds: torch.Tensor,
) -> tuple[torch.Tensor, list[str]]:
    u_index = _find_channel_index(channels, ERA5_U10_CHANNELS)
    v_index = _find_channel_index(channels, ERA5_V10_CHANNELS)
    if u_index is None or v_index is None:
        return tensor, channels

    derived = []
    derived_channels = []
    u10 = tensor[u_index]
    v10 = tensor[v_index]
    if ERA5_WIND_SPEED_10M not in channels:
        derived.append(torch.sqrt(u10.pow(2) + v10.pow(2)))
        derived_channels.append(ERA5_WIND_SPEED_10M)
    if ERA5_RELATIVE_VORTICITY_10M not in channels:
        # Old exports contain nearest-neighbour-upsampled wind. Differentiating
        # those blocks creates false grid-edge vorticity. Keep the historical
        # 20-channel schema with a physically neutral placeholder; new exports
        # provide native-grid vorticity explicitly and take the no-op path here.
        _ = bounds
        derived.append(torch.zeros_like(u10))
        derived_channels.append(ERA5_RELATIVE_VORTICITY_10M)
    if not derived:
        return tensor, channels
    return (
        torch.cat([tensor, torch.stack(derived)], dim=0),
        channels + derived_channels,
    )


def _find_channel_index(channels: list[str], names: set[str]) -> int | None:
    for index, channel in enumerate(channels):
        if channel in names:
            return index
    return None


def _relative_vorticity_10m(
    u10: torch.Tensor,
    v10: torch.Tensor,
    bounds: torch.Tensor,
) -> torch.Tensor:
    height, width = u10.shape[-2:]
    if height < 2 or width < 2:
        return torch.full_like(u10, torch.nan)

    left, right, bottom, top = [float(value) for value in bounds.tolist()]
    lon = _cell_centers(left, right, width)
    lat = _cell_centers(top, bottom, height)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    coslat = np.cos(lat_rad)[:, None]
    coslat = np.clip(coslat, 1e-6, None)

    u = u10.detach().cpu().numpy().astype(np.float64, copy=False)
    v = v10.detach().cpu().numpy().astype(np.float64, copy=False)
    d_v_d_lambda = np.gradient(v, lon_rad, axis=1, edge_order=1)
    d_u_coslat_d_phi = np.gradient(u * coslat, lat_rad, axis=0, edge_order=1)
    vorticity = (
        d_v_d_lambda / (EARTH_RADIUS_M * coslat)
        - d_u_coslat_d_phi / (EARTH_RADIUS_M * coslat)
    )
    return torch.from_numpy(vorticity.astype(np.float32)).to(
        device=u10.device, dtype=u10.dtype
    )


def _cell_centers(start: float, stop: float, count: int) -> np.ndarray:
    spacing = (stop - start) / count
    return start + (np.arange(count, dtype=np.float64) + 0.5) * spacing


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in json.loads(str(value))]


def _row_value(row: pd.Series, primary: str, fallback: Any = "") -> str:
    if primary in row.index and str(row[primary]).strip():
        return str(row[primary])
    return str(fallback)


def _manifest_ibtracs_center(row: pd.Series) -> torch.Tensor:
    """Read the IBTrACS storm center from an exported manifest row."""
    return torch.tensor(
        [_row_float(row, column) for column in IBTRACS_CENTER_COLUMNS],
        dtype=torch.float64,
    )


def _row_float(row: pd.Series, column: str) -> float:
    if column not in row.index or str(row[column]).strip() in {"", "nan", "None"}:
        return np.nan
    value = float(row[column])
    return value if np.isfinite(value) else np.nan


def _manifest_has_era5(samples: pd.DataFrame) -> pd.Series:
    """Return rows with an ERA5 path, including legacy manifest layouts."""
    has_context = pd.Series(False, index=samples.index)
    if "context_path" in samples.columns:
        has_context = samples["context_path"].astype(str).str.strip().ne("")
        if "context_source_type" in samples.columns:
            has_context &= (
                samples["context_source_type"]
                .astype(str)
                .str.strip()
                .str.casefold()
                .eq("era5")
            )
    if "era5_path" in samples.columns:
        has_context |= samples["era5_path"].astype(str).str.strip().ne("")
    return has_context


def _manifest_era5_time_gap_hours(samples: pd.DataFrame) -> pd.Series:
    """Return absolute ERA5-to-observation gaps, or NaN when unverifiable."""
    if "era5_timestamp" not in samples:
        return pd.Series(np.nan, index=samples.index, dtype=float)
    era5_time = pd.to_datetime(
        samples["era5_timestamp"], errors="coerce", utc=True
    )
    reference_time = pd.Series(pd.NaT, index=samples.index, dtype="datetime64[ns, UTC]")
    for column in (
        "condition_timestamp",
        "geo_timestamp",
        "target_timestamp",
        "sar_timestamp",
    ):
        if column in samples:
            parsed = pd.to_datetime(samples[column], errors="coerce", utc=True)
            reference_time = reference_time.fillna(parsed)
    return (era5_time - reference_time).abs().dt.total_seconds() / 3600.0
