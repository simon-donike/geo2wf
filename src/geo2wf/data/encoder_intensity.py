"""Condition-only GEO/ERA5 samples for encoder-only IBTrACS regression."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from geo2wf.data.contracts import DataSpec
from geo2wf.data.datamodule import PairedDataModule
from geo2wf.data.datasets.paired_geotiff import (
    DISTANCE_TO_IBTRACS_CENTER,
    SOLAR_TIME_CHANNELS,
    PairedImageDataset,
    _append_era5_derived_channels,
    _center_crop,
    _center_crop_bounds,
    _condition_timestamp,
    _json_list,
    _manifest_ibtracs_center,
    _normalize,
    _normalized_distance_to_center,
    _normalized_physical_zero,
    _paired_random_flips,
    _read_geotiff,
    _row_value,
    _solar_time_features,
)
from geo2wf.data.joint_intensity import (
    IBTRACS_STRUCTURE_BATCH_KEYS,
    IBTRACS_STRUCTURE_COMPANION,
    INTENSITY_TARGET_COMPANION,
    JointPairedIntensityDataModule,
    JointPairedIntensityDataset,
    _interpolate_ibtracs_wind,
)


ELIGIBILITY_CACHE_SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligibility_fingerprint(
    paired: PairedImageDataset,
    ibtracs_file: Path,
    *,
    max_bracket_hours: float,
    require_sar_valid_center: bool,
    sar_robust_peak_fraction: float,
) -> tuple[str, dict[str, Any]]:
    paired_ids = "\n".join(paired.samples["sample_id"].astype(str)).encode()
    payload = {
        "schema_version": ELIGIBILITY_CACHE_SCHEMA_VERSION,
        "manifest_sha256": _file_sha256(paired.manifest_file),
        "paired_sample_ids_sha256": hashlib.sha256(paired_ids).hexdigest(),
        "ibtracs_sha256": _file_sha256(ibtracs_file),
        "target_size": list(paired.target_size),
        "center_crop_size": (
            list(paired.center_crop_size)
            if paired.center_crop_size is not None
            else None
        ),
        "max_bracket_hours": float(max_bracket_hours),
        "require_sar_valid_center": bool(require_sar_valid_center),
        "sar_robust_peak_fraction": float(sar_robust_peak_fraction),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def cached_joint_eligible_sample_ids(
    paired: PairedImageDataset,
    ibtracs_tracks: dict[str, pd.DataFrame],
    ibtracs_file: Path,
    *,
    cache_dir: Path,
    max_bracket_hours: float,
    require_sar_valid_center: bool,
    sar_robust_peak_fraction: float,
    ri_threshold_kt: float,
    ri_window_hours: float,
) -> list[str]:
    """Return the joint cohort, scanning SAR only when its sidecar is absent."""

    fingerprint, inputs = _eligibility_fingerprint(
        paired,
        ibtracs_file,
        max_bracket_hours=max_bracket_hours,
        require_sar_valid_center=require_sar_valid_center,
        sar_robust_peak_fraction=sar_robust_peak_fraction,
    )
    cache_file = cache_dir / f"{paired.split}-{fingerprint}.json"
    if cache_file.is_file():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            raise ValueError(f"invalid eligibility cache fingerprint: {cache_file}")
        return [str(value) for value in payload["sample_ids"]]

    joint = JointPairedIntensityDataset(
        paired,
        ibtracs_tracks,
        paired.split,
        max_bracket_hours=max_bracket_hours,
        intensity_target_source="ibtracs",
        require_sar_valid_center=require_sar_valid_center,
        sar_robust_peak_fraction=sar_robust_peak_fraction,
        ri_threshold_kt=ri_threshold_kt,
        ri_window_hours=ri_window_hours,
    )
    sample_ids = joint.samples["sample_id"].astype(str).tolist()
    payload = {
        "schema_version": ELIGIBILITY_CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "inputs": inputs,
        "sample_ids": sample_ids,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_name(f".{cache_file.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, cache_file)
    return sample_ids


class EncoderIBTrACSDataset(Dataset):
    """Read only model conditions and continuous IBTrACS labels."""

    def __init__(
        self,
        paired: PairedImageDataset,
        ibtracs_tracks: dict[str, pd.DataFrame],
        eligible_sample_ids: list[str],
        *,
        max_bracket_hours: float,
    ) -> None:
        self.paired = paired
        self.root = paired.root
        self.split = paired.split
        eligible = set(eligible_sample_ids)
        self.samples = paired.samples.loc[
            paired.samples["sample_id"].astype(str).isin(eligible)
        ].reset_index(drop=True)
        if len(self.samples) != len(eligible):
            missing = eligible.difference(self.samples["sample_id"].astype(str))
            raise ValueError(
                f"eligibility cache references missing samples: {sorted(missing)}"
            )
        self.labels: list[dict[str, Any]] = []
        for _, row in self.samples.iterrows():
            storm_id = str(row["storm_id"]).strip().upper()
            label = _interpolate_ibtracs_wind(
                ibtracs_tracks[storm_id],
                row["target_timestamp"],
                max_bracket_hours=max_bracket_hours,
            )
            if label is None:
                raise ValueError(
                    f"cached sample {row['sample_id']} no longer has an IBTrACS label"
                )
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def data_spec(self) -> DataSpec:
        if not len(self):
            raise ValueError(f"{self.split} encoder intensity dataset is empty")
        sample = self[0]
        return DataSpec(
            condition_channels=tuple(sample["meta"]["condition_channels"]),
            target_channels=(),
            spatial_shape=tuple(sample["condition"].shape[-2:]),
            target_units="m s-1",
            companions=frozenset(
                {INTENSITY_TARGET_COMPANION, IBTRACS_STRUCTURE_COMPANION}
            ),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        condition_source_type = _row_value(row, "condition_source_type", "geo")
        condition_channels = _json_list(
            _row_value(row, "condition_channels", row.get("geo_channels"))
        )
        condition_path = _row_value(row, "condition_path", row.get("geo_path"))
        context_path = _row_value(row, "context_path", row.get("era5_path", ""))
        context_source_type = _row_value(
            row, "context_source_type", "era5" if context_path else ""
        )
        context_channels = _json_list(
            _row_value(row, "context_channels", row.get("era5_channels", "[]"))
        )
        if not self.paired.use_era5 and context_source_type == "era5":
            context_path = ""
            context_source_type = ""
            context_channels = []

        condition, condition_mask, condition_bounds = _read_geotiff(
            self.root / condition_path, reject_all_zero_fill=True
        )
        channel_mask = condition_mask.expand(len(condition_channels), -1, -1)
        zero_values = _normalized_physical_zero(
            condition_source_type,
            condition_channels,
            self.paired.stats,
            normalization=self.paired.normalization,
            robust_clip=self.paired.robust_clip,
        )
        condition = _normalize(
            condition,
            condition_source_type,
            condition_channels,
            self.paired.stats,
            normalization=self.paired.normalization,
            robust_clip=self.paired.robust_clip,
        )
        condition = torch.nan_to_num(condition, nan=0.0) * condition_mask.to(
            condition.dtype
        )

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
            context_zero = _normalized_physical_zero(
                context_source_type,
                context_channels,
                self.paired.stats,
                normalization=self.paired.normalization,
                robust_clip=self.paired.robust_clip,
            )
            context = _normalize(
                context,
                context_source_type,
                context_channels,
                self.paired.stats,
                normalization=self.paired.normalization,
                robust_clip=self.paired.robust_clip,
            )
            context = torch.nan_to_num(context, nan=0.0) * context_mask.to(
                context.dtype
            )
            condition = torch.cat([condition, context], dim=0)
            channel_mask = torch.cat(
                [channel_mask, context_mask.expand(len(context_channels), -1, -1)]
            )
            condition_mask = condition_mask & context_mask
            condition_channels += context_channels
            zero_values = torch.cat([zero_values, context_zero])

        if self.paired.center_crop_size is not None:
            original_shape = condition.shape[-2:]
            condition = _center_crop(condition, self.paired.center_crop_size)
            condition_mask = _center_crop(condition_mask, self.paired.center_crop_size)
            channel_mask = _center_crop(channel_mask, self.paired.center_crop_size)
            condition_bounds = _center_crop_bounds(
                condition_bounds, original_shape, self.paired.center_crop_size
            )

        center = _manifest_ibtracs_center(row)
        distance = _normalized_distance_to_center(
            condition_bounds, condition.shape[-2:], center
        )
        solar = _solar_time_features(
            condition_bounds, condition.shape[-2:], _condition_timestamp(row)
        )
        condition = torch.cat([condition, distance, solar], dim=0)
        condition_channels += [DISTANCE_TO_IBTRACS_CENTER, *SOLAR_TIME_CHANNELS]
        zero_values = torch.cat([zero_values, zero_values.new_zeros(4)])
        channel_mask = torch.cat(
            [
                channel_mask,
                torch.ones_like(distance, dtype=torch.bool),
                torch.ones_like(solar, dtype=torch.bool),
            ]
        )
        if self.paired.augment:
            dummy = condition.new_zeros((1, *condition.shape[-2:]))
            dummy_mask = torch.ones_like(dummy, dtype=torch.bool)
            condition, _, condition_mask, _ = _paired_random_flips(
                condition,
                dummy,
                condition_mask,
                dummy_mask,
                condition_channels=condition_channels,
                condition_zero_values=zero_values,
                condition_channel_mask=channel_mask,
            )

        label = self.labels[index]
        sample = {
            "condition": condition,
            "condition_mask": condition_mask,
            "condition_bounds": condition_bounds,
            "center": center,
            "sample_id": str(row["sample_id"]),
            "intensity_target_ms": torch.tensor(
                float(label["target_wind_ms"]), dtype=torch.float32
            ),
            "meta": {
                "storm_id": str(row["storm_id"]),
                "condition_source_type": condition_source_type,
                "condition_sensor": _row_value(
                    row, "condition_sensor", row.get("geo_sensor", "")
                ),
                "condition_channels": condition_channels,
                "context_source_type": context_source_type,
                "context_channels": context_channels,
            },
        }
        for key in IBTRACS_STRUCTURE_BATCH_KEYS:
            value = float(label.get(key, float("nan")))
            sample[key] = torch.tensor(value, dtype=torch.float32)
            sample[f"{key}_valid"] = torch.tensor(
                math.isfinite(value), dtype=torch.bool
            )
        return sample


class EncoderIBTrACSDataModule(JointPairedIntensityDataModule):
    """Preserve the joint cohort while yielding no SAR target tensors."""

    def __init__(
        self,
        *args: Any,
        eligibility_cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eligibility_cache_dir = (
            Path(eligibility_cache_dir).expanduser()
            if eligibility_cache_dir is not None
            else self.root / ".geo2wf" / "encoder-ibtracs-cohorts"
        )

    def _make_dataset(
        self, split: str, *, augment: bool = False
    ) -> EncoderIBTrACSDataset:
        paired = PairedDataModule._make_dataset(self, split, augment=augment)
        eligible_ids = cached_joint_eligible_sample_ids(
            paired,
            self.ibtracs_tracks,
            self.ibtracs_file,
            cache_dir=self.eligibility_cache_dir,
            max_bracket_hours=self.max_ibtracs_bracket_hours,
            require_sar_valid_center=self.require_sar_valid_center,
            sar_robust_peak_fraction=self.sar_robust_peak_fraction,
            ri_threshold_kt=self.ri_threshold_kt,
            ri_window_hours=self.ri_window_hours,
        )
        return EncoderIBTrACSDataset(
            paired,
            self.ibtracs_tracks,
            eligible_ids,
            max_bracket_hours=self.max_ibtracs_bracket_hours,
        )


__all__ = [
    "ELIGIBILITY_CACHE_SCHEMA_VERSION",
    "EncoderIBTrACSDataModule",
    "EncoderIBTrACSDataset",
    "cached_joint_eligible_sample_ids",
]
