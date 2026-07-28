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


EARTH_RADIUS_M = 6_371_000.0
ERA5_U10_CHANNELS = {"era5_u_wind_10m", "u_wind_10m", "era5_u10", "u10"}
ERA5_V10_CHANNELS = {"era5_v_wind_10m", "v_wind_10m", "era5_v10", "v10"}
ERA5_WIND_SPEED_10M = "era5_wind_speed_10m"
ERA5_RELATIVE_VORTICITY_10M = "era5_relative_vorticity_10m"
DEFAULT_CHANNEL_STATS = {
    ("era5", ERA5_WIND_SPEED_10M): {"min": 0.0, "max": 85.0},
    ("era5", ERA5_RELATIVE_VORTICITY_10M): {"min": -0.005, "max": 0.005},
}


class PairedImageDataset(Dataset):
    """Read exported paired GeoTIFF samples from a split manifest."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        stats_file: str | Path | None = None,
        target_size: tuple[int, int] = (256, 256),
        augment: bool = False,
    ) -> None:
        self.root = Path(root).expanduser()
        self.split = split
        self.manifest_file = self.root / split / "manifest.csv"
        if not self.manifest_file.exists():
            raise FileNotFoundError(
                f"GeoTIFF manifest does not exist: {self.manifest_file}"
            )
        self.samples = pd.read_csv(self.manifest_file, keep_default_na=False)
        self.stats_file = (
            Path(stats_file).expanduser()
            if stats_file is not None
            else self.root / "stats.json"
        )
        if not self.stats_file.exists():
            raise FileNotFoundError(f"Stats file does not exist: {self.stats_file}")
        self.stats = json.loads(self.stats_file.read_text(encoding="utf-8"))
        self.target_size = target_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples.iloc[idx]
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
        context = None
        context_mask = None
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
        target, target_mask, target_bounds = _read_geotiff(self.root / target_path)
        condition = _normalize(
            condition, condition_source_type, condition_channels, self.stats
        )
        if context is not None:
            context = _normalize(
                context, context_source_type, context_channels, self.stats
            )
        target = _normalize(target, target_source_type, target_channels, self.stats)
        condition = torch.nan_to_num(condition, nan=0.0)
        if context is not None:
            context = torch.nan_to_num(context, nan=0.0)
        target = torch.nan_to_num(target, nan=0.0)
        condition = condition * condition_mask.to(condition.dtype)
        if context is not None and context_mask is not None:
            context = context * context_mask.to(context.dtype)
            condition = torch.cat([condition, context], dim=0)
            condition_mask = condition_mask & context_mask
            condition_channels = condition_channels + context_channels
        target = target * target_mask.to(target.dtype)
        target, target_mask = _resize_target(target, target_mask, self.target_size)
        if self.augment:
            condition, target, condition_mask, target_mask = _paired_random_flips(
                condition, target, condition_mask, target_mask
            )
        return {
            "condition": condition,
            "target": target,
            "condition_mask": condition_mask,
            "target_mask": target_mask,
            "condition_bounds": condition_bounds,
            "target_bounds": target_bounds,
            "center": torch.tensor([_row_float(row, "center_lat"), _row_float(row, "center_lon")], dtype=torch.float64),
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


def _paired_random_flips(
    condition: torch.Tensor,
    target: torch.Tensor,
    condition_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the same random horizontal and vertical flips to a paired sample."""
    flip_dims = []
    if torch.rand(()) < 0.5:
        flip_dims.append(-1)
    if torch.rand(()) < 0.5:
        flip_dims.append(-2)
    if flip_dims:
        condition = torch.flip(condition, dims=flip_dims)
        target = torch.flip(target, dims=flip_dims)
        condition_mask = torch.flip(condition_mask, dims=flip_dims)
        target_mask = torch.flip(target_mask, dims=flip_dims)
    return condition, target, condition_mask, target_mask


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
) -> torch.Tensor:
    normalized = tensor.clone().float()
    stats_by_channel = stats.get("channels", {}).get(source_type, {})
    for index, channel in enumerate(channels):
        channel_stats = stats_by_channel.get(channel)
        if channel_stats is None:
            channel_stats = stats_by_channel.get(f"band_{index}")
        if channel_stats is None:
            channel_stats = DEFAULT_CHANNEL_STATS.get((source_type, channel))
        if channel_stats is None:
            raise KeyError(f"Missing stats for {source_type}:{channel}")
        min_value = float(channel_stats["min"])
        max_value = float(channel_stats["max"])
        denom = max(max_value - min_value, 1e-6)
        normalized[index] = (normalized[index] - min_value) / denom
    return normalized.clamp(0.0, 1.0)


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
        derived.append(_relative_vorticity_10m(u10, v10, bounds))
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


def _row_float(row: pd.Series, column: str) -> float:
    if column not in row.index or str(row[column]).strip() in {"", "nan", "None"}:
        return np.nan
    value = float(row[column])
    return value if np.isfinite(value) else np.nan
