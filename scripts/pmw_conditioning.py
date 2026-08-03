"""Shared PMW matching and condition preparation for raw-storm inference."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import numpy as np
import torch

from geo2wf.data.normalization import normalize as _normalize
from scripts.export_geo_sar_geotiffs import (
    PMW_SOURCE_CHANNELS,
    Observation,
    _build_pmw_companion,
)


def pmw_condition_settings(config: Mapping[str, Any]) -> tuple[bool, float, bool]:
    """Return enabled, maximum gap in hours, and offset-channel activation."""
    data = config.get("data", {})
    enabled = bool(data.get("pmw_as_condition", False))
    maximum_gap = float(data.get("max_pmw_time_gap_hours", 1.0))
    include_offset = bool(data.get("pmw_include_time_offset", False))
    if enabled and not data.get("include_pmw", False):
        raise ValueError("pmw_as_condition requires data.include_pmw")
    if enabled and maximum_gap <= 0:
        raise ValueError("data.max_pmw_time_gap_hours must be positive")
    if include_offset and not enabled:
        raise ValueError("pmw_include_time_offset requires pmw_as_condition")
    return enabled, maximum_gap, include_offset


def supported_pmw_by_storm(
    records: Iterable[Observation],
) -> dict[str, list[Observation]]:
    """Group supported, timestamped PMW observations by storm."""
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for record in records:
        if (
            record.source_type == "pmw"
            and record.sensor in PMW_SOURCE_CHANNELS
            and record.timestamp is not None
        ):
            grouped[record.storm_id].append(record)
    return {
        storm: sorted(items, key=lambda item: item.timestamp)
        for storm, items in grouped.items()
    }


def nearest_supported_pmw(
    reference: Observation,
    records_by_storm: Mapping[str, Sequence[Observation]],
    *,
    max_time_gap_hours: float,
) -> tuple[Observation | None, float | None, str]:
    """Select the closest supported PMW overpass within the configured window."""
    if reference.timestamp is None:
        return None, None, "reference_missing_timestamp"
    candidates = records_by_storm.get(reference.storm_id, ())
    if not candidates:
        return None, None, "no_supported_pmw_for_storm"
    selected = min(
        candidates,
        key=lambda item: abs((item.timestamp - reference.timestamp).total_seconds()),
    )
    gap_minutes = (selected.timestamp - reference.timestamp).total_seconds() / 60.0
    if abs(gap_minutes) > max_time_gap_hours * 60.0:
        return selected, gap_minutes, "outside_time_window"
    return selected, gap_minutes, "matched"


def prepare_pmw_condition_features(
    reference: Observation,
    pmw: Observation,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    stats: dict[str, Any],
    *,
    max_time_gap_hours: float,
    include_time_offset: bool,
    crop_size: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Regrid and normalize PMW into value, mask, and optional offset fields."""
    if reference.timestamp is None or pmw.timestamp is None:
        raise ValueError("PMW conditioning requires finite reference and PMW times")
    gap_minutes = (pmw.timestamp - reference.timestamp).total_seconds() / 60.0
    if abs(gap_minutes) > max_time_gap_hours * 60.0:
        raise ValueError(
            f"PMW gap {gap_minutes:.2f} min exceeds {max_time_gap_hours:.2f} h"
        )
    companion = _build_pmw_companion(pmw, grid_lat, grid_lon)
    values = torch.from_numpy(companion["pmw"])
    mask = torch.from_numpy(companion["pmw_mask"]).bool().unsqueeze(0)
    values = _normalize(
        values,
        "pmw",
        list(companion["pmw_channels"]),
        stats,
        normalization="robust-zscore",
        robust_clip=4.0,
    )
    values = torch.nan_to_num(values) * mask.to(values.dtype)
    features = [values, mask.to(values.dtype)]
    if include_time_offset:
        maximum_minutes = max_time_gap_hours * 60.0
        normalized_gap = float(
            np.clip(0.5 + gap_minutes / (2.0 * maximum_minutes), 0.0, 1.0)
        )
        features.append(values.new_full((1, *values.shape[-2:]), normalized_gap))
    feature_tensor = _center_crop(torch.cat(features, dim=0), crop_size)
    return feature_tensor, _center_crop(mask, crop_size), gap_minutes


def pmw_audit_row(
    reference: Observation,
    pmw: Observation | None,
    gap_minutes: float | None,
    status: str,
    *,
    reason: str = "",
) -> dict[str, Any]:
    """Build one stable audit record for matched or skipped inference frames."""
    return {
        "observation_id": reference.observation_id,
        "reference_timestamp": (
            reference.timestamp.isoformat() if reference.timestamp is not None else ""
        ),
        "pmw_observation_id": pmw.observation_id if pmw is not None else "",
        "pmw_sensor": pmw.sensor if pmw is not None else "",
        "pmw_timestamp": (
            pmw.timestamp.isoformat()
            if pmw is not None and pmw.timestamp is not None
            else ""
        ),
        "pmw_dt_minutes": gap_minutes if gap_minutes is not None else np.nan,
        "status": status,
        "reason": reason or status,
    }


def _center_crop(tensor: torch.Tensor, size: int) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    if size > height or size > width:
        raise ValueError(f"crop size {size} exceeds PMW shape {(height, width)}")
    top = (height - size) // 2
    left = (width - size) // 2
    return tensor[..., top : top + size, left : left + size]
