"""Stable tensor and metadata contracts shared by datasets and models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from typing_extensions import NotRequired, TypedDict

import torch


class SampleMetadata(TypedDict, total=False):
    """Non-tensor metadata retained per sample by the canonical collator."""

    storm_id: str
    condition_source_type: str
    target_source_type: str
    condition_sensor: str
    target_sensor: str
    dt_minutes: float
    condition_channels: list[str]
    context_source_type: str
    context_channels: list[str]
    target_channels: list[str]
    pmw_channels: list[str]
    pmw_sensor: str
    pmw_dt_minutes: float


class WindFieldBatch(TypedDict):
    """Canonical collated batch with explicit optional companion fields."""

    condition: torch.Tensor
    condition_mask: torch.Tensor
    target: torch.Tensor
    target_physical: torch.Tensor
    target_mask: torch.Tensor
    target_norm_offset: torch.Tensor
    target_norm_scale: torch.Tensor
    condition_bounds: torch.Tensor
    target_bounds: torch.Tensor
    center: torch.Tensor
    sample_id: list[str]
    meta: list[SampleMetadata]
    era5_wind_speed: NotRequired[torch.Tensor]
    era5_wind_speed_physical: NotRequired[torch.Tensor]
    era5_wind_speed_mask: NotRequired[torch.Tensor]
    pmw: NotRequired[torch.Tensor]
    pmw_physical: NotRequired[torch.Tensor]
    pmw_mask: NotRequired[torch.Tensor]
    pmw_bounds: NotRequired[torch.Tensor]
    ibtracs: NotRequired[list[Mapping[str, Any]]]


@dataclass(frozen=True)
class DataSpec:
    """Dataset capabilities used to reject incompatible models early."""

    condition_channels: tuple[str, ...]
    target_channels: tuple[str, ...]
    spatial_shape: tuple[int, int]
    target_units: str = "m s-1"
    companions: frozenset[str] = frozenset()

    @property
    def condition_channel_count(self) -> int:
        return len(self.condition_channels)

    @property
    def target_channel_count(self) -> int:
        return len(self.target_channels)


REQUIRED_BATCH_FIELDS = frozenset(
    {
        "condition",
        "condition_mask",
        "target",
        "target_physical",
        "target_mask",
        "target_norm_offset",
        "target_norm_scale",
        "condition_bounds",
        "target_bounds",
        "center",
        "sample_id",
        "meta",
    }
)


def validate_batch(batch: Mapping[str, Any]) -> None:
    """Validate the inexpensive structural portion of the batch contract."""

    missing = sorted(REQUIRED_BATCH_FIELDS.difference(batch))
    if missing:
        raise KeyError("wind-field batch is missing: " + ", ".join(missing))
    for name in REQUIRED_BATCH_FIELDS.difference({"sample_id", "meta"}):
        if not torch.is_tensor(batch[name]):
            raise TypeError(f"batch field {name!r} must be a tensor")
