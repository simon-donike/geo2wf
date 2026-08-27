"""Learned single-field correction from U-Net wind maps to scalar intensity."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from geo2wf.data.intensity import (
    IBTRACS_STRUCTURE_MANIFEST_COLUMNS,
    IBTRACS_STRUCTURE_TARGET_NAMES,
    INTENSITY_METADATA_NAMES,
    IntensityDataSpec,
    UNET_IMAGE_STRUCTURE_MANIFEST_COLUMNS,
    UNET_IMAGE_STRUCTURE_TARGET_NAMES,
    category_macro_f1_tensor,
    tropical_category_from_wind_ms_tensor,
)
from geo2wf.tracking.intensity_media import log_wandb_intensity_evaluation


CATEGORY_NAMES = {
    -1: "td",
    0: "ts",
    1: "c1",
    2: "c2",
    3: "c3",
    4: "c4",
    5: "c5",
}


@dataclass(frozen=True)
class IntensityPredictionBatch:
    raw_unet_anchor_ms: torch.Tensor
    raw_unet_max_wind_ms: torch.Tensor
    raw_unet_robust_peak_ms: torch.Tensor
    correction_ms: torch.Tensor
    output_msw_ms: torch.Tensor
    output_category: torch.Tensor
    structure_prediction_km: torch.Tensor | None = None

    @property
    def structure_outputs_km(self) -> dict[str, torch.Tensor]:
        if self.structure_prediction_km is None:
            return {}
        return {
            name: self.structure_prediction_km[:, index]
            for index, name in enumerate(IBTRACS_STRUCTURE_TARGET_NAMES)
        }


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, max(channels // 2, 1)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ScalarResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs + self.block(inputs))


class WindFieldEncoder(nn.Module):
    """Compact CNN retaining both broad structure and peak-sensitive features."""

    def __init__(
        self,
        base_channels: int = 16,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        if base_channels <= 0 or not channel_mults:
            raise ValueError("field encoder widths must be positive and non-empty")
        if any(int(multiplier) <= 0 for multiplier in channel_mults):
            raise ValueError("channel_mults must contain positive integers")
        widths = [base_channels * int(value) for value in channel_mults]
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(_group_count(widths[0]), widths[0]),
            nn.SiLU(),
        )
        stages = []
        for index, width in enumerate(widths):
            layers: list[nn.Module] = []
            if index:
                layers.extend(
                    [
                        nn.Conv2d(
                            widths[index - 1], width, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(_group_count(width), width),
                        nn.SiLU(),
                    ]
                )
            layers.extend([ScalarResidualBlock(width), ScalarResidualBlock(width)])
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.output_features = 2 * widths[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stages(self.stem(inputs))
        # Reductions over the flattened spatial axes are exactly equivalent to
        # adaptive pooling to a 1x1 output. Unlike adaptive_max_pool2d CUDA
        # backward, amax supports deterministic training.
        spatial_features = features.flatten(2)
        average = spatial_features.mean(dim=2)
        maximum = spatial_features.amax(dim=2)
        return torch.cat([average, maximum], dim=1)


def _as_rows(
    sample_ids: Sequence[str],
    storm_ids: Sequence[str],
    observation_timestamps: Sequence[str],
    target_sources: Sequence[str],
    prediction: torch.Tensor,
    target: torch.Tensor,
    raw_anchor: torch.Tensor,
    raw_max: torch.Tensor,
    raw_robust_peak: torch.Tensor,
    ibtracs_target: torch.Tensor,
    sar_robust_peak_target: torch.Tensor,
    is_rapid_intensification: torch.Tensor,
    correction: torch.Tensor,
    prediction_category: torch.Tensor,
    target_category: torch.Tensor,
) -> list[dict[str, Any]]:
    arrays = [
        value.detach().cpu().reshape(-1).tolist()
        for value in (
            prediction,
            target,
            raw_anchor,
            raw_max,
            raw_robust_peak,
            ibtracs_target,
            sar_robust_peak_target,
            is_rapid_intensification,
            correction,
            prediction_category,
            target_category,
        )
    ]
    return [
        {
            "sample_id": str(sample_id),
            "storm_id": str(storm_id),
            "observation_timestamp": str(observation_timestamp),
            "intensity_target_source": str(target_source),
            "prediction_ms": float(predicted),
            "target_ms": float(observed),
            "raw_unet_ms": float(baseline_anchor),
            "raw_unet_anchor_ms": float(baseline_anchor),
            "raw_unet_max_ms": float(baseline_max),
            "raw_unet_robust_peak_ms": float(baseline_robust_peak),
            "ibtracs_target_ms": float(ibtracs_observed),
            "sar_robust_peak_target_ms": float(sar_observed),
            "is_rapid_intensification": bool(is_ri),
            "correction_ms": float(correction_ms),
            "prediction_category": int(category),
            "target_category": int(observed_category),
        }
        for (
            sample_id,
            storm_id,
            observation_timestamp,
            target_source,
            predicted,
            observed,
            baseline_anchor,
            baseline_max,
            baseline_robust_peak,
            ibtracs_observed,
            sar_observed,
            is_ri,
            correction_ms,
            category,
            observed_category,
        ) in zip(
            sample_ids,
            storm_ids,
            observation_timestamps,
            target_sources,
            *arrays,
        )
    ]


def rows_for_intensity_reference(
    rows: Sequence[Mapping[str, Any]], reference: str
) -> list[dict[str, Any]]:
    """Project dual-reference rows onto one scalar evaluation target."""

    reference = str(reference).strip().lower()
    if reference == "ibtracs":
        target_key = "ibtracs_target_ms"
        baseline_key = "raw_unet_max_ms"
    elif reference == "sar_robust_peak":
        target_key = "sar_robust_peak_target_ms"
        baseline_key = "raw_unet_robust_peak_ms"
    else:
        raise ValueError(f"unknown intensity reference {reference!r}")
    projected = []
    for row in rows:
        target = float(row.get(target_key, math.nan))
        baseline = float(row.get(baseline_key, math.nan))
        if not math.isfinite(target) or not math.isfinite(baseline):
            continue
        projected.append(
            {
                **dict(row),
                "target_ms": target,
                "target_category": int(
                    tropical_category_from_wind_ms_tensor(
                        torch.tensor(target, dtype=torch.float64)
                    ).item()
                ),
                "raw_unet_ms": baseline,
            }
        )
    return projected


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "mae_ms": float(np.mean(np.abs(error))),
        "rmse_ms": float(np.sqrt(np.mean(np.square(error)))),
        "bias_ms": float(np.mean(error)),
    }


def summarize_intensity_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return JSON-safe regression, category, per-storm, and baseline metrics."""

    if not rows:
        raise ValueError("cannot summarize an empty intensity evaluation")
    prediction = np.asarray([row["prediction_ms"] for row in rows], dtype=float)
    target = np.asarray([row["target_ms"] for row in rows], dtype=float)
    raw = np.asarray([row["raw_unet_ms"] for row in rows], dtype=float)
    correction = np.asarray(
        [
            row.get(
                "correction_ms", float(row["prediction_ms"]) - float(row["raw_unet_ms"])
            )
            for row in rows
        ],
        dtype=float,
    )
    predicted_category = np.asarray(
        [row["prediction_category"] for row in rows], dtype=int
    )
    target_category = np.asarray([row["target_category"] for row in rows], dtype=int)
    if not (
        np.isfinite(prediction).all()
        and np.isfinite(target).all()
        and np.isfinite(raw).all()
        and np.isfinite(correction).all()
    ):
        raise ValueError("intensity evaluation rows must be finite")

    per_storm: dict[str, dict[str, float]] = {}
    for storm_id in sorted({str(row["storm_id"]) for row in rows}):
        selected = np.asarray(
            [str(row["storm_id"]) == storm_id for row in rows], dtype=bool
        )
        per_storm[storm_id] = {
            "samples": int(selected.sum()),
            **_regression_metrics(prediction[selected], target[selected]),
            "raw_unet_mae_ms": _regression_metrics(raw[selected], target[selected])[
                "mae_ms"
            ],
        }
    storm_macro_mae = float(
        np.mean([metrics["mae_ms"] for metrics in per_storm.values()])
    )

    classes = list(range(-1, 6))
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for observed, predicted in zip(target_category, predicted_category):
        if observed in classes and predicted in classes:
            confusion[observed + 1, predicted + 1] += 1
    f1_values = []
    per_category = {}
    for category in classes:
        observed = target_category == category
        predicted = predicted_category == category
        true_positive = int(np.count_nonzero(observed & predicted))
        false_positive = int(np.count_nonzero(~observed & predicted))
        false_negative = int(np.count_nonzero(observed & ~predicted))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else math.nan
        if observed.any():
            f1_values.append(f1)
            per_category[CATEGORY_NAMES[category]] = {
                "samples": int(observed.sum()),
                "mae_ms": float(
                    np.mean(np.abs(prediction[observed] - target[observed]))
                ),
                "accuracy": float(np.mean(predicted_category[observed] == category)),
                "f1": float(f1),
            }

    return {
        "samples": len(rows),
        "storms": len(per_storm),
        "regression": _regression_metrics(prediction, target),
        "raw_unet_baseline": _regression_metrics(raw, target),
        "correction": {
            "mean_ms": float(np.mean(correction)),
            "mean_abs_ms": float(np.mean(np.abs(correction))),
        },
        "storm_macro_mae_ms": storm_macro_mae,
        "category": {
            "accuracy": float(np.mean(predicted_category == target_category)),
            "macro_f1": float(np.mean(f1_values)),
            "within_one_accuracy": float(
                np.mean(np.abs(predicted_category - target_category) <= 1)
            ),
            "labels": [CATEGORY_NAMES[value] for value in classes],
            "confusion_matrix": confusion.tolist(),
            "per_category": per_category,
        },
        "per_storm": per_storm,
    }


class UNetIntensityCorrection(pl.LightningModule):
    """Correct one frozen U-Net wind field to a supervised intensity scalar."""

    checkpoint_monitor = "val/storm_macro_mae_ms"
    checkpoint_mode = "min"

    def __init__(
        self,
        metadata_features: int = len(INTENSITY_METADATA_NAMES),
        field_base_channels: int = 16,
        field_channel_mults: Sequence[int] = (1, 2, 4, 8),
        metadata_hidden_features: int = 32,
        fusion_hidden_features: int = 128,
        dropout: float = 0.1,
        wind_soft_scale_ms: float = 20.0,
        anchor_statistic: str = "max",
        robust_peak_fraction: float = 0.005,
        huber_delta_ms: float = 5.0,
        structure_head_enabled: bool = False,
        structure_loss_weight: float = 0.0,
        structure_huber_delta_km: float = 20.0,
        use_field: bool = True,
        use_metadata: bool = True,
        lr: float = 3.0e-4,
        weight_decay: float = 1.0e-4,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 10,
        lr_scheduler_min_lr: float = 1.0e-6,
        log_wandb_validation_media: bool = True,
        validation_plot_storm_count: int = 3,
        validation_plot_storm_ids: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if metadata_features <= 0 or metadata_hidden_features <= 0:
            raise ValueError("metadata dimensions must be positive")
        if fusion_hidden_features <= 0 or wind_soft_scale_ms <= 0:
            raise ValueError("fusion width and wind scale must be positive")
        if huber_delta_ms <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError(
                "Huber delta must be positive and dropout must be in [0,1)"
            )
        if not use_field and not use_metadata:
            raise ValueError("at least one of use_field or use_metadata must be true")
        if validation_plot_storm_count < 1:
            raise ValueError("validation_plot_storm_count must be at least one")
        anchor_statistic = str(anchor_statistic).strip().lower()
        if anchor_statistic not in {"max", "robust_peak"}:
            raise ValueError("anchor_statistic must be 'max' or 'robust_peak'")
        if not 0.0 < robust_peak_fraction <= 1.0:
            raise ValueError("robust_peak_fraction must be in (0, 1]")
        if structure_loss_weight < 0.0 or structure_huber_delta_km <= 0.0:
            raise ValueError(
                "structure loss weight/delta must be non-negative/positive"
            )
        if structure_loss_weight > 0.0 and not structure_head_enabled:
            raise ValueError("positive structure loss requires structure_head_enabled")
        self.save_hyperparameters()
        self.metadata_features = int(metadata_features)
        self.wind_soft_scale_ms = float(wind_soft_scale_ms)
        self.anchor_statistic = anchor_statistic
        self.robust_peak_fraction = float(robust_peak_fraction)
        self.huber_delta_ms = float(huber_delta_ms)
        self.structure_head_enabled = bool(structure_head_enabled)
        self.structure_loss_weight = float(structure_loss_weight)
        self.structure_huber_delta_km = float(structure_huber_delta_km)
        self.use_field = bool(use_field)
        self.use_metadata = bool(use_metadata)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.lr_scheduler_factor = float(lr_scheduler_factor)
        self.lr_scheduler_patience = int(lr_scheduler_patience)
        self.lr_scheduler_min_lr = float(lr_scheduler_min_lr)
        self.log_wandb_validation_media = bool(log_wandb_validation_media)
        self.validation_plot_storm_count = int(validation_plot_storm_count)
        self.validation_plot_storm_ids = tuple(validation_plot_storm_ids or ())

        feature_width = 0
        if self.use_field:
            self.field_encoder = WindFieldEncoder(
                field_base_channels, tuple(field_channel_mults)
            )
            feature_width += self.field_encoder.output_features
        else:
            self.field_encoder = None
        if self.use_metadata:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(self.metadata_features, metadata_hidden_features),
                nn.LayerNorm(metadata_hidden_features),
                nn.SiLU(),
                nn.Linear(metadata_hidden_features, metadata_hidden_features),
                nn.SiLU(),
            )
            feature_width += metadata_hidden_features
        else:
            self.metadata_encoder = None
        self.fusion = nn.Sequential(
            nn.Linear(feature_width, fusion_hidden_features),
            nn.LayerNorm(fusion_hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_features, fusion_hidden_features // 2),
            nn.SiLU(),
        )
        self.correction_head = nn.Linear(fusion_hidden_features // 2, 1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        self.structure_head = (
            nn.Linear(fusion_hidden_features // 2, len(IBTRACS_STRUCTURE_TARGET_NAMES))
            if self.structure_head_enabled
            else None
        )
        if self.structure_head is not None:
            nn.init.normal_(self.structure_head.weight, std=1.0e-3)
            nn.init.constant_(self.structure_head.bias, 4.0)
        self._evaluation_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def validate_data_spec(self, spec: IntensityDataSpec) -> None:
        if not isinstance(spec, IntensityDataSpec):
            raise TypeError(
                "UNetIntensityCorrection requires an IntensityDataSpec, got "
                f"{type(spec).__name__}"
            )
        if spec.metadata_feature_count != self.metadata_features:
            raise ValueError(
                f"model expects {self.metadata_features} metadata features, "
                f"data provides {spec.metadata_feature_count}: {spec.metadata_names}"
            )
        if self.structure_loss_weight > 0.0 and spec.cache_schema_version < 3:
            raise ValueError(
                "structure loss requires an intensity cache with schema version 3"
            )

    @staticmethod
    def _batch_tensors(batch: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
        required = (
            "wind_field",
            "valid_mask",
            "distance_to_center",
            "metadata",
        )
        missing = [name for name in required if name not in batch]
        if missing:
            raise KeyError("intensity batch is missing: " + ", ".join(missing))
        wind = batch["wind_field"]
        mask = batch["valid_mask"]
        distance = batch["distance_to_center"]
        metadata = batch["metadata"]
        if not all(
            torch.is_tensor(value) for value in (wind, mask, distance, metadata)
        ):
            raise TypeError("intensity image and metadata fields must be tensors")
        if wind.ndim != 3 or mask.shape != wind.shape or distance.shape != wind.shape:
            raise ValueError("intensity spatial tensors must share shape [B,H,W]")
        if metadata.ndim != 2 or metadata.shape[0] != wind.shape[0]:
            raise ValueError("intensity metadata must have shape [B,F]")
        return wind, mask, distance, metadata

    def predict_intensity(self, batch: Mapping[str, Any]) -> IntensityPredictionBatch:
        wind, mask, distance, metadata = self._batch_tensors(batch)
        finite_mask = mask.bool() & torch.isfinite(wind) & torch.isfinite(distance)
        if not finite_mask.flatten(1).any(dim=1).all():
            raise ValueError(
                "every intensity sample must contain a valid current field"
            )
        clean_wind = torch.where(finite_mask, wind, torch.zeros_like(wind))
        raw_max = torch.where(finite_mask, wind, -torch.inf).flatten(1).amax(dim=1)
        robust_peaks = []
        for sample_wind, sample_mask in zip(wind, finite_mask):
            values = sample_wind[sample_mask]
            count = max(1, int(math.ceil(values.numel() * self.robust_peak_fraction)))
            robust_peaks.append(torch.topk(values, count, sorted=False).values.mean())
        raw_robust_peak = torch.stack(robust_peaks)
        raw_anchor = raw_max if self.anchor_statistic == "max" else raw_robust_peak
        features = []
        if self.field_encoder is not None:
            encoded_wind = torch.asinh(
                clean_wind.clamp_min(0.0) / self.wind_soft_scale_ms
            )
            field_input = torch.stack(
                [encoded_wind, finite_mask.to(wind), distance.clamp(0.0, 1.0)],
                dim=1,
            )
            features.append(self.field_encoder(field_input))
        if self.metadata_encoder is not None:
            if metadata.shape[1] != self.metadata_features:
                raise ValueError(
                    f"metadata must have {self.metadata_features} features, "
                    f"got {metadata.shape[1]}"
                )
            features.append(self.metadata_encoder(metadata.to(wind)))
        fused = self.fusion(torch.cat(features, dim=1))
        correction = self.correction_head(fused).squeeze(1)
        output = (raw_anchor + correction).clamp_min(0.0)
        structure = (
            F.softplus(self.structure_head(fused))
            if self.structure_head is not None
            else None
        )
        return IntensityPredictionBatch(
            raw_unet_anchor_ms=raw_anchor,
            raw_unet_max_wind_ms=raw_max,
            raw_unet_robust_peak_ms=raw_robust_peak,
            correction_ms=correction,
            output_msw_ms=output,
            output_category=tropical_category_from_wind_ms_tensor(output),
            structure_prediction_km=structure,
        )

    def forward(
        self,
        wind_field: torch.Tensor,
        valid_mask: torch.Tensor,
        distance_to_center: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_intensity(
            {
                "wind_field": wind_field,
                "valid_mask": valid_mask,
                "distance_to_center": distance_to_center,
                "metadata": metadata,
            }
        ).output_msw_ms

    def _loss(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, IntensityPredictionBatch]:
        prediction = self.predict_intensity(batch)
        target = batch["target_wind_ms"].to(prediction.output_msw_ms).reshape(-1)
        weights = batch.get("sample_weight", torch.ones_like(target)).to(target)
        element_loss = F.smooth_l1_loss(
            prediction.output_msw_ms,
            target,
            reduction="none",
            beta=self.huber_delta_ms,
        )
        intensity_loss = (element_loss * weights).sum() / weights.sum().clamp_min(1e-12)
        structure_loss, _ = self._structure_loss_and_metrics(prediction, batch)
        loss = intensity_loss + self.structure_loss_weight * structure_loss
        return loss, prediction

    @staticmethod
    def _structure_targets(
        batch: Mapping[str, Any], reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        values = []
        masks = []
        for key in IBTRACS_STRUCTURE_MANIFEST_COLUMNS:
            valid_key = f"{key}_valid"
            if key not in batch or valid_key not in batch:
                return None
            value = torch.as_tensor(batch[key], device=reference.device).reshape(-1)
            valid = torch.as_tensor(
                batch[valid_key], device=reference.device, dtype=torch.bool
            ).reshape(-1)
            values.append(value.to(reference))
            masks.append(valid & torch.isfinite(value) & (value >= 0.0))
        return torch.stack(values, dim=1), torch.stack(masks, dim=1)

    def _structure_loss_and_metrics(
        self,
        prediction: IntensityPredictionBatch,
        batch: Mapping[str, Any],
        sample_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = prediction.structure_prediction_km
        if output is None:
            return prediction.output_msw_ms.sum() * 0.0, {}
        targets = self._structure_targets(batch, output)
        if targets is None:
            if self.structure_loss_weight > 0.0:
                raise KeyError("structure-supervised MLP requires IBTrACS targets")
            return output.sum() * 0.0, {}
        target, valid = targets
        if sample_mask is not None:
            valid = valid & sample_mask.to(valid.device, dtype=torch.bool).reshape(
                -1, 1
            )
        safe_target = torch.where(valid, target, output.detach())
        element_loss = F.smooth_l1_loss(
            output,
            safe_target,
            reduction="none",
            beta=self.structure_huber_delta_km,
        )
        weight = valid.to(output)
        loss = (element_loss * weight).sum() / weight.sum().clamp_min(1.0)
        metrics: dict[str, torch.Tensor] = {"structure_loss": loss}
        error = output - safe_target
        for index, name in enumerate(IBTRACS_STRUCTURE_TARGET_NAMES):
            selected = valid[:, index]
            if not selected.any():
                continue
            selected_error = error[selected, index]
            metrics[f"structure_{name}_mae_km"] = selected_error.abs().mean()
            metrics[f"structure_{name}_rmse_km"] = selected_error.square().mean().sqrt()
            metrics[f"structure_{name}_bias_km"] = selected_error.mean()
        return loss, metrics

    @staticmethod
    def _unet_image_structure_metrics(
        batch: Mapping[str, Any],
        reference: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        for name, image_key, target_index in zip(
            UNET_IMAGE_STRUCTURE_TARGET_NAMES,
            UNET_IMAGE_STRUCTURE_MANIFEST_COLUMNS,
            range(1, len(IBTRACS_STRUCTURE_TARGET_NAMES)),
        ):
            valid_key = f"{image_key}_valid"
            target_key = IBTRACS_STRUCTURE_MANIFEST_COLUMNS[target_index]
            target_valid_key = f"{target_key}_valid"
            if not {image_key, valid_key, target_key, target_valid_key}.issubset(batch):
                continue
            image_value = torch.as_tensor(batch[image_key], device=reference.device).to(
                reference
            )
            target = torch.as_tensor(batch[target_key], device=reference.device).to(
                reference
            )
            valid = (
                torch.as_tensor(batch[valid_key], device=reference.device).bool()
                & torch.as_tensor(
                    batch[target_valid_key], device=reference.device
                ).bool()
                & torch.isfinite(image_value)
                & torch.isfinite(target)
            )
            if sample_mask is not None:
                valid = valid & sample_mask.to(valid.device, dtype=torch.bool).reshape(
                    -1
                )
            if valid.any():
                error = image_value[valid] - target[valid]
                metrics[f"unet_image_{name}_mae_km"] = error.abs().mean()
                metrics[f"unet_image_{name}_rmse_km"] = error.square().mean().sqrt()
                metrics[f"unet_image_{name}_bias_km"] = error.mean()
        return metrics

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss, prediction = self._loss(batch)
        target = batch["target_wind_ms"].to(prediction.output_msw_ms)
        error = prediction.output_msw_ms - target
        raw_error = prediction.raw_unet_anchor_ms - target
        target_category = batch["target_category"].to(prediction.output_category)
        metrics = {
            "loss": loss,
            "mae_ms": error.abs().mean(),
            "rmse_ms": error.square().mean().sqrt(),
            "bias_ms": error.mean(),
            "raw_unet_mae_ms": raw_error.abs().mean(),
            "correction_mean_ms": prediction.correction_ms.mean(),
            "correction_mean_abs_ms": prediction.correction_ms.abs().mean(),
            "category_accuracy": (prediction.output_category == target_category)
            .float()
            .mean(),
            "category_within_one_accuracy": (
                (prediction.output_category - target_category).abs() <= 1
            )
            .float()
            .mean(),
        }
        _, structure_metrics = self._structure_loss_and_metrics(prediction, batch)
        metrics.update(structure_metrics)
        metrics.update(
            self._unet_image_structure_metrics(batch, prediction.output_msw_ms)
        )
        for name, value in metrics.items():
            self.log(
                f"train/{name}",
                value,
                on_step=name == "loss",
                on_epoch=True,
                prog_bar=name in {"loss", "mae_ms"},
                batch_size=int(target.numel()),
            )
        return loss

    def _evaluation_step(self, batch: Mapping[str, Any], prefix: str) -> None:
        loss, prediction = self._loss(batch)
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=int(prediction.output_msw_ms.numel()),
        )
        target = batch["target_wind_ms"].to(prediction.output_msw_ms)
        _, structure_metrics = self._structure_loss_and_metrics(prediction, batch)
        structure_metrics.update(
            self._unet_image_structure_metrics(batch, prediction.output_msw_ms)
        )
        for name, value in structure_metrics.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=int(prediction.output_msw_ms.numel()),
            )
        ri = torch.as_tensor(
            batch.get(
                "is_rapid_intensification",
                torch.zeros_like(prediction.output_msw_ms, dtype=torch.bool),
            ),
            device=prediction.output_msw_ms.device,
            dtype=torch.bool,
        ).reshape(-1)
        if ri.any():
            _, ri_structure_metrics = self._structure_loss_and_metrics(
                prediction, batch, sample_mask=ri
            )
            ri_structure_metrics.update(
                self._unet_image_structure_metrics(
                    batch, prediction.output_msw_ms, sample_mask=ri
                )
            )
            for name, value in ri_structure_metrics.items():
                self.log(
                    f"{prefix}_ri/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(ri.sum()),
                )
        target_category = batch["target_category"].to(prediction.output_category)
        self._evaluation_rows[prefix].extend(
            _as_rows(
                batch["sample_id"],
                batch["storm_id"],
                batch["observation_timestamp"],
                batch.get(
                    "intensity_target_source",
                    ["ibtracs"] * len(batch["sample_id"]),
                ),
                prediction.output_msw_ms,
                target,
                prediction.raw_unet_anchor_ms,
                prediction.raw_unet_max_wind_ms,
                prediction.raw_unet_robust_peak_ms,
                batch.get("ibtracs_target_ms", target).to(target),
                batch.get("sar_robust_peak_target_ms", target).to(target),
                batch.get(
                    "is_rapid_intensification",
                    torch.zeros_like(target, dtype=torch.bool),
                ).to(target.device),
                prediction.correction_ms,
                prediction.output_category,
                target_category,
            )
        )

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        del batch_idx
        self._evaluation_step(batch, "val")

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        del batch_idx
        self._evaluation_step(batch, "test")

    def on_validation_epoch_start(self) -> None:
        self._evaluation_rows["val"] = []

    def on_test_epoch_start(self) -> None:
        self._evaluation_rows["test"] = []

    @staticmethod
    def _distributed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return rows
        gathered: list[list[dict[str, Any]] | None] = [
            None for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather_object(gathered, rows)
        combined = [row for part in gathered if part is not None for row in part]
        # DistributedSampler may pad validation with repeated samples.
        return list({str(row["sample_id"]): row for row in combined}.values())

    def _log_evaluation(self, prefix: str) -> None:
        rows = self._distributed_rows(self._evaluation_rows[prefix])
        summary = summarize_intensity_rows(rows)
        scalar_metrics = {
            "mae_ms": summary["regression"]["mae_ms"],
            "rmse_ms": summary["regression"]["rmse_ms"],
            "bias_ms": summary["regression"]["bias_ms"],
            "storm_macro_mae_ms": summary["storm_macro_mae_ms"],
            "raw_unet_mae_ms": summary["raw_unet_baseline"]["mae_ms"],
            "raw_unet_rmse_ms": summary["raw_unet_baseline"]["rmse_ms"],
            "raw_unet_bias_ms": summary["raw_unet_baseline"]["bias_ms"],
            "correction_mean_ms": summary["correction"]["mean_ms"],
            "correction_mean_abs_ms": summary["correction"]["mean_abs_ms"],
            "category_accuracy": summary["category"]["accuracy"],
            "category_macro_f1": summary["category"]["macro_f1"],
            "category_within_one_accuracy": summary["category"]["within_one_accuracy"],
        }
        for name, value in scalar_metrics.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name in {"storm_macro_mae_ms", "mae_ms"},
                sync_dist=False,
            )
        for name, metrics in summary["category"]["per_category"].items():
            for metric_name in ("mae_ms", "accuracy", "f1"):
                self.log(
                    f"{prefix}/{name}_{metric_name}",
                    metrics[metric_name],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
        if prefix == "val":
            ri_rows = [
                row for row in rows if bool(row.get("is_rapid_intensification", False))
            ]
            self.log(
                "val_ri/samples",
                float(len(ri_rows)),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                "val_ri/storms",
                float(len({str(row["storm_id"]) for row in ri_rows})),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            for reference in ("ibtracs", "sar_robust_peak"):
                projected = rows_for_intensity_reference(ri_rows, reference)
                if not projected:
                    continue
                reference_summary = summarize_intensity_rows(projected)
                raw = torch.tensor(
                    [float(row["raw_unet_ms"]) for row in projected],
                    dtype=torch.float64,
                )
                target = torch.tensor(
                    [float(row["target_ms"]) for row in projected],
                    dtype=torch.float64,
                )
                raw_category = tropical_category_from_wind_ms_tensor(raw)
                target_category = tropical_category_from_wind_ms_tensor(target)
                reference_metrics = {
                    "mae_ms": reference_summary["regression"]["mae_ms"],
                    "rmse_ms": reference_summary["regression"]["rmse_ms"],
                    "bias_ms": reference_summary["regression"]["bias_ms"],
                    "storm_macro_mae_ms": reference_summary["storm_macro_mae_ms"],
                    "raw_unet_mae_ms": reference_summary["raw_unet_baseline"]["mae_ms"],
                    "raw_unet_rmse_ms": reference_summary["raw_unet_baseline"][
                        "rmse_ms"
                    ],
                    "raw_unet_bias_ms": reference_summary["raw_unet_baseline"][
                        "bias_ms"
                    ],
                    "raw_unet_storm_macro_mae_ms": float(
                        np.mean(
                            [
                                metrics["raw_unet_mae_ms"]
                                for metrics in reference_summary["per_storm"].values()
                            ]
                        )
                    ),
                    "category_accuracy": reference_summary["category"]["accuracy"],
                    "category_macro_f1": reference_summary["category"]["macro_f1"],
                    "category_within_one_accuracy": reference_summary["category"][
                        "within_one_accuracy"
                    ],
                    "raw_unet_category_accuracy": float(
                        (raw_category == target_category).double().mean()
                    ),
                    "raw_unet_category_macro_f1": float(
                        category_macro_f1_tensor(raw_category, target_category)
                    ),
                    "raw_unet_category_within_one_accuracy": float(
                        ((raw_category - target_category).abs() <= 1).double().mean()
                    ),
                }
                for name, value in reference_metrics.items():
                    self.log(
                        f"val_ri/{reference}_{name}",
                        value,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=False,
                    )
        if prefix == "val" and self.log_wandb_validation_media:
            log_wandb_intensity_evaluation(
                self,
                rows,
                summary,
                prefix=prefix,
                storm_count=self.validation_plot_storm_count,
                preferred_storm_ids=self.validation_plot_storm_ids,
            )

    def on_validation_epoch_end(self) -> None:
        self._log_evaluation("val")

    def on_test_epoch_end(self) -> None:
        self._log_evaluation("test")

    @torch.no_grad()
    def predict_step(
        self, batch: Mapping[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        del batch_idx, dataloader_idx
        prediction = self.predict_intensity(batch)
        result = {
            "sample_id": list(batch["sample_id"]),
            "storm_id": list(batch["storm_id"]),
            "observation_timestamp": list(batch["observation_timestamp"]),
            "raw_unet_max_wind_ms": prediction.raw_unet_max_wind_ms,
            "raw_unet_robust_peak_ms": prediction.raw_unet_robust_peak_ms,
            "raw_unet_anchor_ms": prediction.raw_unet_anchor_ms,
            "correction_ms": prediction.correction_ms,
            "output_msw_ms": prediction.output_msw_ms,
            "output_category": prediction.output_category,
        }
        if prediction.structure_prediction_km is not None:
            result["structure_prediction_km"] = prediction.structure_prediction_km
            for index, name in enumerate(IBTRACS_STRUCTURE_TARGET_NAMES):
                result[f"output_{name}_km"] = prediction.structure_prediction_km[
                    :, index
                ]
        return result

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=self.lr_scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.checkpoint_monitor,
            },
        }


__all__ = [
    "CATEGORY_NAMES",
    "IntensityPredictionBatch",
    "UNetIntensityCorrection",
    "WindFieldEncoder",
    "rows_for_intensity_reference",
    "summarize_intensity_rows",
]
