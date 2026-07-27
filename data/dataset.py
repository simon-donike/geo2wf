from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


class PairedImageDataset(Dataset):
    """Read exported GEO/SAR GeoTIFF pairs from a split manifest."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        stats_file: str | Path | None = None,
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples.iloc[idx]
        geo_channels = _json_list(row["geo_channels"])
        sar_channels = _json_list(row["sar_channels"])
        condition, condition_mask = _read_geotiff(self.root / row["geo_path"])
        target, target_mask = _read_geotiff(self.root / row["sar_path"])
        condition = _normalize(condition, "geo", geo_channels, self.stats)
        target = _normalize(target, "sar", sar_channels, self.stats)
        condition = torch.nan_to_num(condition, nan=0.0)
        target = torch.nan_to_num(target, nan=0.0)
        condition = condition * condition_mask.to(condition.dtype)
        target = target * target_mask.to(target.dtype)
        return {
            "condition": condition,
            "target": target,
            "condition_mask": condition_mask,
            "target_mask": target_mask,
            "sample_id": str(row["sample_id"]),
            "meta": {
                "storm_id": str(row["storm_id"]),
                "geo_sensor": str(row["geo_sensor"]),
                "sar_sensor": str(row["sar_sensor"]),
                "dt_minutes": float(row["dt_minutes"]),
                "geo_channels": geo_channels,
                "sar_channels": sar_channels,
            },
        }


def _read_geotiff(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with rasterio.open(path) as dataset:
        array = dataset.read().astype("float32")
        mask = dataset.dataset_mask() > 0
    tensor = torch.from_numpy(array)
    mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0)
    return tensor, mask_tensor


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
            raise KeyError(f"Missing stats for {source_type}:{channel}")
        min_value = float(channel_stats["min"])
        max_value = float(channel_stats["max"])
        denom = max(max_value - min_value, 1e-6)
        normalized[index] = (normalized[index] - min_value) / denom
    return normalized.clamp(0.0, 1.0)


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in json.loads(str(value))]
