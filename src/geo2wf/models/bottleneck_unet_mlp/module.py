"""End-to-end U-Net reconstruction with a bottleneck intensity head."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from geo2wf.data.contracts import DataSpec
from geo2wf.data.joint_intensity import (
    IBTRACS_MAX_WIND_COMPANION,
    INTENSITY_TARGET_COMPANION,
)
from geo2wf.data.intensity import (
    category_macro_f1_tensor,
    tropical_category_from_wind_ms_tensor,
)
from geo2wf.models.base import (
    LossOutput,
    PredictionBatch,
    PredictionRequest,
    WindFieldLightningModule,
)
from geo2wf.tracking.reconstruction_media import log_wandb_reconstruction


def _group_count(channels: int, maximum: int = 8) -> int:
    upper_bound = min(maximum, max(channels // 2, 1))
    for groups in range(upper_bound, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    """Group-normalized residual block suitable for small image batches."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        hidden = F.silu(self.norm1(self.conv1(inputs)))
        hidden = self.norm2(self.conv2(hidden))
        return F.silu(hidden + residual)


@dataclass(frozen=True)
class JointUNetOutput:
    reconstruction_normalized: torch.Tensor
    intensity_prediction_ms: torch.Tensor
    bottleneck: torch.Tensor

    @property
    def ibtracs_max_wind_ms(self) -> torch.Tensor:
        """Compatibility alias for checkpoints and downstream callers."""

        return self.intensity_prediction_ms


@dataclass(frozen=True, kw_only=True)
class JointPredictionBatch(PredictionBatch):
    """Standard physical field prediction plus continuous scalar intensity."""

    intensity_prediction_ms: torch.Tensor

    def __post_init__(self) -> None:
        super().__post_init__()
        if tuple(self.intensity_prediction_ms.shape) != (
            self.central_physical.shape[0],
        ):
            raise ValueError("intensity_prediction_ms must have shape [B]")

    @property
    def ibtracs_max_wind_ms(self) -> torch.Tensor:
        """Compatibility alias for the former IBTrACS-specific output name."""

        return self.intensity_prediction_ms


class BottleneckUNetMLP(nn.Module):
    """Shared U-Net encoder with decoder and scalar bottleneck branches."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        intensity_hidden_features: int = 128,
        intensity_dropout: float = 0.1,
        initial_intensity_ms: float = 25.0,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or base_channels <= 0:
            raise ValueError("input and base channel counts must be positive")
        if not channel_mults or any(int(value) <= 0 for value in channel_mults):
            raise ValueError("channel_mults must contain positive integers")
        if intensity_hidden_features < 2:
            raise ValueError("intensity_hidden_features must be at least two")
        if not 0.0 <= intensity_dropout < 1.0:
            raise ValueError("intensity_dropout must be in [0, 1)")
        if initial_intensity_ms <= 0.0:
            raise ValueError("initial_intensity_ms must be positive")

        dimensions = [base_channels * int(value) for value in channel_mults]
        self.stem = nn.Conv2d(in_channels, dimensions[0], 3, padding=1)
        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, dimension in enumerate(dimensions):
            self.encoder.append(
                nn.Sequential(
                    ResidualBlock(dimension, dimension),
                    ResidualBlock(dimension, dimension),
                )
            )
            if index + 1 < len(dimensions):
                self.downsamples.append(
                    nn.Conv2d(
                        dimension,
                        dimensions[index + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )

        self.bottleneck = nn.Sequential(
            ResidualBlock(dimensions[-1], dimensions[-1]),
            ResidualBlock(dimensions[-1], dimensions[-1]),
        )
        pooled_features = 2 * dimensions[-1]
        second_hidden = max(intensity_hidden_features // 2, 1)
        self.intensity_mlp = nn.Sequential(
            nn.Linear(pooled_features, intensity_hidden_features),
            nn.LayerNorm(intensity_hidden_features),
            nn.SiLU(),
            nn.Dropout(intensity_dropout),
            nn.Linear(intensity_hidden_features, second_hidden),
            nn.SiLU(),
        )
        self.intensity_head = nn.Linear(second_hidden, 1)
        nn.init.normal_(self.intensity_head.weight, std=1.0e-3)
        nn.init.constant_(
            self.intensity_head.bias,
            math.log(math.expm1(float(initial_intensity_ms))),
        )

        self.decoder_projections = nn.ModuleList()
        self.decoder = nn.ModuleList()
        current_channels = dimensions[-1]
        for skip_channels in reversed(dimensions[:-1]):
            self.decoder_projections.append(
                nn.Conv2d(current_channels, skip_channels, kernel_size=1)
            )
            self.decoder.append(
                nn.Sequential(
                    ResidualBlock(2 * skip_channels, skip_channels),
                    ResidualBlock(skip_channels, skip_channels),
                )
            )
            current_channels = skip_channels
        self.reconstruction_head = nn.Conv2d(dimensions[0], 1, kernel_size=1)
        nn.init.normal_(self.reconstruction_head.weight, std=1.0e-3)
        nn.init.zeros_(self.reconstruction_head.bias)

    def forward(self, inputs: torch.Tensor) -> JointUNetOutput:
        if inputs.ndim != 4:
            raise ValueError("U-Net inputs must have shape [B,C,H,W]")
        hidden = self.stem(inputs)
        skips = []
        for index, block in enumerate(self.encoder):
            hidden = block(hidden)
            skips.append(hidden)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)

        bottleneck = self.bottleneck(hidden)
        flattened = bottleneck.flatten(2)
        pooled = torch.cat([flattened.mean(dim=2), flattened.amax(dim=2)], dim=1)
        intensity = F.softplus(self.intensity_head(self.intensity_mlp(pooled))).squeeze(
            1
        )

        hidden = bottleneck
        for projection, block, skip in zip(
            self.decoder_projections,
            self.decoder,
            reversed(skips[:-1]),
        ):
            hidden = F.interpolate(
                hidden,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            hidden = projection(hidden)
            hidden = block(torch.cat([hidden, skip], dim=1))
        reconstruction = torch.sigmoid(self.reconstruction_head(hidden))
        return JointUNetOutput(reconstruction, intensity, bottleneck)


def _huber_values(error: torch.Tensor, delta: float) -> torch.Tensor:
    absolute = error.abs()
    return torch.where(
        absolute <= delta,
        0.5 * error.square(),
        delta * (absolute - 0.5 * delta),
    )


def _masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    values = _huber_values(prediction - target, delta)
    mask = mask.to(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class BottleneckUNetMLPRegressor(WindFieldLightningModule):
    """Jointly reconstruct a SAR field and regress one scalar wind reference."""

    checkpoint_monitor = "val/loss"
    checkpoint_mode = "min"

    def __init__(
        self,
        condition_channels: int = 23,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        intensity_hidden_features: int = 128,
        intensity_dropout: float = 0.1,
        initial_intensity_ms: float = 25.0,
        image_huber_delta_ms: float = 2.0,
        intensity_huber_delta_ms: float = 5.0,
        image_loss_weight: float = 1.0,
        intensity_loss_weight: float = 1.0,
        lr: float = 2.0e-4,
        weight_decay: float = 1.0e-4,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 25,
        lr_scheduler_cooldown: int = 0,
        lr_scheduler_min_lr: float = 0.0,
        validation_reconstruction_batches: int = 1,
        log_reconstruction_images: bool = True,
        sar_robust_peak_fraction: float = 0.005,
    ) -> None:
        super().__init__()
        if condition_channels <= 0:
            raise ValueError("condition_channels must be positive")
        if image_huber_delta_ms <= 0.0 or intensity_huber_delta_ms <= 0.0:
            raise ValueError("Huber deltas must be positive")
        if image_loss_weight < 0.0 or intensity_loss_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if image_loss_weight + intensity_loss_weight <= 0.0:
            raise ValueError("at least one loss weight must be positive")
        if validation_reconstruction_batches < 1:
            raise ValueError("validation_reconstruction_batches must be positive")
        if not 0.0 < sar_robust_peak_fraction <= 1.0:
            raise ValueError("sar_robust_peak_fraction must be in (0, 1]")
        self.save_hyperparameters()
        self.condition_channels = int(condition_channels)
        self.image_huber_delta_ms = float(image_huber_delta_ms)
        self.intensity_huber_delta_ms = float(intensity_huber_delta_ms)
        self.image_loss_weight = float(image_loss_weight)
        self.intensity_loss_weight = float(intensity_loss_weight)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.lr_scheduler_factor = float(lr_scheduler_factor)
        self.lr_scheduler_patience = int(lr_scheduler_patience)
        self.lr_scheduler_cooldown = int(lr_scheduler_cooldown)
        self.lr_scheduler_min_lr = float(lr_scheduler_min_lr)
        self.validation_reconstruction_batches = int(validation_reconstruction_batches)
        self.log_reconstruction_images = bool(log_reconstruction_images)
        self.sar_robust_peak_fraction = float(sar_robust_peak_fraction)
        self.model = BottleneckUNetMLP(
            self.condition_channels + 1,
            base_channels,
            tuple(channel_mults),
            intensity_hidden_features,
            intensity_dropout,
            initial_intensity_ms,
        )
        self.register_buffer(
            "_validation_field_statistics",
            torch.zeros(5, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_validation_intensity_statistics",
            torch.zeros(5, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_test_field_statistics",
            torch.zeros(5, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_test_intensity_statistics",
            torch.zeros(5, dtype=torch.float64),
            persistent=False,
        )
        self._evaluation_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def validate_data_spec(self, spec: DataSpec) -> None:
        super().validate_data_spec(spec)
        if spec.target_channel_count != 1 or spec.target_channels != ("wind_speed",):
            raise ValueError(
                "BottleneckUNetMLPRegressor requires one wind_speed target, got "
                f"{spec.target_channels}"
            )
        if spec.target_units != "m s-1":
            raise ValueError(
                "BottleneckUNetMLPRegressor requires target units m s-1, got "
                f"{spec.target_units!r}"
            )
        if not {
            INTENSITY_TARGET_COMPANION,
            IBTRACS_MAX_WIND_COMPANION,
        }.intersection(spec.companions):
            raise ValueError("data must provide continuous scalar intensity labels")

    def forward(
        self, condition: torch.Tensor, condition_mask: torch.Tensor
    ) -> JointUNetOutput:
        if condition.ndim != 4 or condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"condition must have {self.condition_channels} channels, got "
                f"{tuple(condition.shape)}"
            )
        mask = self._single_channel(condition_mask, "condition_mask").to(condition)
        return self.model(torch.cat([condition * mask, mask], dim=1))

    def predict_normalized(self, batch: Mapping[str, Any]) -> JointUNetOutput:
        return self(batch["condition"], batch["condition_mask"])

    def predict_joint(self, batch: Mapping[str, Any]) -> JointPredictionBatch:
        output = self.predict_normalized(batch)
        offset, scale = self._normalization_parameters(
            batch, output.reconstruction_normalized
        )
        physical = output.reconstruction_normalized * scale + offset
        return JointPredictionBatch(
            samples_physical=physical.unsqueeze(1),
            central_physical=physical,
            intensity_prediction_ms=output.intensity_prediction_ms,
        )

    def predict_physical(self, batch: Mapping[str, Any]) -> torch.Tensor:
        return self.predict_joint(batch).central_physical

    def _losses(
        self, batch: Mapping[str, Any]
    ) -> tuple[
        torch.Tensor, torch.Tensor, JointPredictionBatch, torch.Tensor, torch.Tensor
    ]:
        prediction = self.predict_joint(batch)
        target, mask = self._field_target_and_mask(batch, prediction.central_physical)
        intensity_target = self._intensity_target(batch, prediction.central_physical)
        image_loss = _masked_huber_loss(
            prediction.central_physical,
            target,
            mask,
            self.image_huber_delta_ms,
        )
        intensity_loss = _huber_values(
            prediction.intensity_prediction_ms - intensity_target,
            self.intensity_huber_delta_ms,
        ).mean()
        return image_loss, intensity_loss, prediction, target, mask

    def compute_training_objective(self, batch: Mapping[str, Any]) -> LossOutput:
        image_loss, intensity_loss, prediction, target, mask = self._losses(batch)
        intensity_target = self._intensity_target(batch, prediction.central_physical)
        image_error = prediction.central_physical - target
        intensity_error = prediction.intensity_prediction_ms - intensity_target
        valid_count = mask.sum().clamp_min(1.0)
        masked_squared_image_error = (image_error.square() * mask).sum() / valid_count
        total = (
            self.image_loss_weight * image_loss
            + self.intensity_loss_weight * intensity_loss
        )
        return LossOutput(
            total,
            {
                "image_loss": image_loss,
                "intensity_loss": intensity_loss,
                "image_mae_ms": (image_error.abs() * mask).sum() / valid_count,
                "image_rmse_ms": masked_squared_image_error.sqrt(),
                "image_bias_ms": (image_error * mask).sum() / valid_count,
                "intensity_mae_ms": intensity_error.abs().mean(),
                "intensity_rmse_ms": intensity_error.square().mean().sqrt(),
                "intensity_bias_ms": intensity_error.mean(),
            },
        )

    @torch.no_grad()
    def predict_batch(
        self, batch: Mapping[str, Any], request: PredictionRequest
    ) -> JointPredictionBatch:
        prediction = self.predict_joint(batch)
        members = prediction.central_physical.unsqueeze(1).expand(
            -1, request.ensemble_size, -1, -1, -1
        )
        return JointPredictionBatch(
            samples_physical=members,
            central_physical=prediction.central_physical,
            intensity_prediction_ms=prediction.intensity_prediction_ms,
        )

    @torch.no_grad()
    def validation_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx == 1:
            if self.log_reconstruction_images and batch_idx == 0:
                self._log_reconstruction(batch, "images/train_reconstruction")
            return
        if dataloader_idx:
            return
        _, _, prediction, target, mask = self._losses(batch)
        intensity_target = self._intensity_target(batch, prediction.central_physical)
        self._accumulate(
            self._validation_field_statistics,
            prediction.central_physical,
            target,
            mask,
            self.image_huber_delta_ms,
        )
        self._accumulate(
            self._validation_intensity_statistics,
            prediction.intensity_prediction_ms,
            intensity_target,
            torch.ones_like(intensity_target),
            self.intensity_huber_delta_ms,
        )
        ibtracs_target = batch.get("ibtracs_target_ms", intensity_target).to(
            prediction.intensity_prediction_ms
        )
        sar_target = batch.get("sar_robust_peak_target_ms", intensity_target).to(
            prediction.intensity_prediction_ms
        )
        is_ri = batch.get(
            "is_rapid_intensification",
            torch.zeros_like(intensity_target, dtype=torch.bool),
        ).bool()
        storm_ids = [str(item.get("storm_id", "")) for item in batch["meta"]]
        condition_valid = self._single_channel(
            batch["condition_mask"], "condition_mask"
        ).bool() & torch.isfinite(prediction.central_physical)
        for index, sample_id in enumerate(batch["sample_id"]):
            field_values = prediction.central_physical[index][condition_valid[index]]
            field_error = (prediction.central_physical[index] - target[index])[
                mask[index].bool()
            ]
            raw_max = float(field_values.max().detach().cpu())
            robust_count = max(
                1,
                int(math.ceil(field_values.numel() * self.sar_robust_peak_fraction)),
            )
            raw_robust_peak = float(
                torch.topk(field_values, robust_count, sorted=False)
                .values.mean()
                .detach()
                .cpu()
            )
            self._evaluation_rows["val"].append(
                {
                    "sample_id": str(sample_id),
                    "storm_id": storm_ids[index],
                    "prediction_ms": float(
                        prediction.intensity_prediction_ms[index].detach().cpu()
                    ),
                    "ibtracs_target_ms": float(ibtracs_target[index].detach().cpu()),
                    "sar_robust_peak_target_ms": float(
                        sar_target[index].detach().cpu()
                    ),
                    "raw_unet_max_ms": raw_max,
                    "raw_unet_robust_peak_ms": raw_robust_peak,
                    "field_valid_pixels": int(field_error.numel()),
                    "field_absolute_error_sum": float(
                        field_error.abs().sum().detach().cpu()
                    ),
                    "field_squared_error_sum": float(
                        field_error.square().sum().detach().cpu()
                    ),
                    "field_signed_error_sum": float(field_error.sum().detach().cpu()),
                    "is_rapid_intensification": bool(is_ri[index].detach().cpu()),
                }
            )
        if (
            self.log_reconstruction_images
            and batch_idx < self.validation_reconstruction_batches
        ):
            self._log_reconstruction(batch, "images/val_reconstruction")

    def on_validation_epoch_start(self) -> None:
        self._validation_field_statistics.zero_()
        self._validation_intensity_statistics.zero_()
        self._evaluation_rows["val"] = []

    def on_validation_epoch_end(self) -> None:
        self._log_statistics(
            "val",
            self._validation_field_statistics,
            self._validation_intensity_statistics,
        )
        self._log_ri_statistics()

    @torch.no_grad()
    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        del batch_idx
        _, _, prediction, target, mask = self._losses(batch)
        intensity_target = self._intensity_target(batch, prediction.central_physical)
        self._accumulate(
            self._test_field_statistics,
            prediction.central_physical,
            target,
            mask,
            self.image_huber_delta_ms,
        )
        self._accumulate(
            self._test_intensity_statistics,
            prediction.intensity_prediction_ms,
            intensity_target,
            torch.ones_like(intensity_target),
            self.intensity_huber_delta_ms,
        )

    def on_test_epoch_start(self) -> None:
        self._test_field_statistics.zero_()
        self._test_intensity_statistics.zero_()

    def on_test_epoch_end(self) -> None:
        self._log_statistics(
            "test", self._test_field_statistics, self._test_intensity_statistics
        )

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            cooldown=self.lr_scheduler_cooldown,
            min_lr=self.lr_scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.checkpoint_monitor,
            },
        }

    def _field_target_and_mask(self, batch: Mapping[str, Any], reference: torch.Tensor):
        target = self._single_channel(batch["target_physical"], "target_physical").to(
            reference
        )
        target_mask = self._single_channel(batch["target_mask"], "target_mask")
        condition_mask = self._single_channel(batch["condition_mask"], "condition_mask")
        return target, (target_mask.bool() & condition_mask.bool()).to(reference)

    @staticmethod
    def _intensity_target(
        batch: Mapping[str, Any], reference: torch.Tensor
    ) -> torch.Tensor:
        if "intensity_target_ms" not in batch:
            raise KeyError("joint batch is missing continuous intensity_target_ms")
        target = batch["intensity_target_ms"]
        if not torch.is_tensor(target):
            raise TypeError("intensity_target_ms must be a tensor")
        target = target.to(reference).reshape(-1)
        if target.shape != (reference.shape[0],):
            raise ValueError("intensity_target_ms must have one value per sample")
        if not torch.isfinite(target).all() or bool((target < 0.0).any()):
            raise ValueError("intensity_target_ms must be finite and non-negative")
        return target

    @staticmethod
    def _normalization_parameters(batch: Mapping[str, Any], reference: torch.Tensor):
        offset = batch["target_norm_offset"].to(reference)
        scale = batch["target_norm_scale"].to(reference)
        if offset.ndim == 1:
            offset, scale = offset.reshape(1, -1, 1, 1), scale.reshape(1, -1, 1, 1)
        elif offset.ndim == 2:
            offset, scale = offset[:, :, None, None], scale[:, :, None, None]
        elif offset.ndim == 3:
            offset, scale = offset.unsqueeze(0), scale.unsqueeze(0)
        if offset.shape[1:] != (1, 1, 1) or scale.shape != offset.shape:
            raise ValueError("target normalization must describe one channel")
        if offset.shape[0] not in {1, reference.shape[0]}:
            raise ValueError("target normalization batch dimension is incompatible")
        return offset, scale

    @staticmethod
    def _accumulate(
        statistics: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        delta: float,
    ) -> None:
        error = prediction - target
        mask = mask.to(error)
        statistics.add_(
            torch.stack(
                [
                    mask.sum(),
                    (error.abs() * mask).sum(),
                    (error.square() * mask).sum(),
                    (error * mask).sum(),
                    (_huber_values(error, delta) * mask).sum(),
                ]
            ).to(statistics)
        )

    def _log_statistics(
        self,
        prefix: str,
        field_statistics: torch.Tensor,
        intensity_statistics: torch.Tensor,
    ) -> None:
        field = field_statistics.clone()
        intensity = intensity_statistics.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(field)
            torch.distributed.all_reduce(intensity)
        field_count = field[0].clamp_min(1.0)
        intensity_count = intensity[0].clamp_min(1.0)
        image_loss = field[4] / field_count
        intensity_loss = intensity[4] / intensity_count
        metrics = {
            "loss": self.image_loss_weight * image_loss
            + self.intensity_loss_weight * intensity_loss,
            "image_loss": image_loss,
            "intensity_loss": intensity_loss,
            "image_mae_ms": field[1] / field_count,
            "image_rmse_ms": torch.sqrt(field[2] / field_count),
            "image_bias_ms": field[3] / field_count,
            "intensity_mae_ms": intensity[1] / intensity_count,
            "intensity_rmse_ms": torch.sqrt(intensity[2] / intensity_count),
            "intensity_bias_ms": intensity[3] / intensity_count,
        }
        for name, value in metrics.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name in {"loss", "image_rmse_ms", "intensity_mae_ms"},
                sync_dist=False,
            )

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

    def _log_ri_statistics(self) -> None:
        rows = self._distributed_rows(self._evaluation_rows["val"])
        rows = [row for row in rows if bool(row.get("is_rapid_intensification", False))]
        self.log("val_ri/samples", float(len(rows)), on_epoch=True, sync_dist=False)
        self.log(
            "val_ri/storms",
            float(len({str(row["storm_id"]) for row in rows})),
            on_epoch=True,
            sync_dist=False,
        )
        for reference, target_key, baseline_key in (
            ("ibtracs", "ibtracs_target_ms", "raw_unet_max_ms"),
            (
                "sar_robust_peak",
                "sar_robust_peak_target_ms",
                "raw_unet_robust_peak_ms",
            ),
        ):
            selected = [
                row
                for row in rows
                if math.isfinite(float(row.get(target_key, math.nan)))
            ]
            if not selected:
                continue
            prediction = torch.tensor(
                [float(row["prediction_ms"]) for row in selected], dtype=torch.float64
            )
            target = torch.tensor(
                [float(row[target_key]) for row in selected], dtype=torch.float64
            )
            baseline = torch.tensor(
                [float(row[baseline_key]) for row in selected], dtype=torch.float64
            )
            error = prediction - target
            baseline_error = baseline - target
            storm_mae = []
            baseline_storm_mae = []
            for storm_id in sorted({str(row["storm_id"]) for row in selected}):
                indices = [
                    index
                    for index, row in enumerate(selected)
                    if str(row["storm_id"]) == storm_id
                ]
                storm_mae.append(error[indices].abs().mean())
                baseline_storm_mae.append(baseline_error[indices].abs().mean())
            predicted_category = tropical_category_from_wind_ms_tensor(prediction)
            baseline_category = tropical_category_from_wind_ms_tensor(baseline)
            target_category = tropical_category_from_wind_ms_tensor(target)
            metrics = {
                "mae_ms": error.abs().mean(),
                "rmse_ms": error.square().mean().sqrt(),
                "bias_ms": error.mean(),
                "storm_macro_mae_ms": torch.stack(storm_mae).mean(),
                "category_accuracy": (predicted_category == target_category)
                .double()
                .mean(),
                "category_macro_f1": category_macro_f1_tensor(
                    predicted_category, target_category
                ),
                "category_within_one_accuracy": (
                    (predicted_category - target_category).abs() <= 1
                )
                .double()
                .mean(),
                "raw_unet_mae_ms": baseline_error.abs().mean(),
                "raw_unet_rmse_ms": baseline_error.square().mean().sqrt(),
                "raw_unet_bias_ms": baseline_error.mean(),
                "raw_unet_storm_macro_mae_ms": torch.stack(baseline_storm_mae).mean(),
                "raw_unet_category_accuracy": (baseline_category == target_category)
                .double()
                .mean(),
                "raw_unet_category_macro_f1": category_macro_f1_tensor(
                    baseline_category, target_category
                ),
                "raw_unet_category_within_one_accuracy": (
                    (baseline_category - target_category).abs() <= 1
                )
                .double()
                .mean(),
            }
            for name, value in metrics.items():
                self.log(
                    f"val_ri/{reference}_{name}",
                    value.to(self.device),
                    on_epoch=True,
                    sync_dist=False,
                )
        field_count = sum(int(row.get("field_valid_pixels", 0)) for row in rows)
        if field_count:
            for name, value in {
                "field_mae_ms": sum(
                    float(row["field_absolute_error_sum"]) for row in rows
                )
                / field_count,
                "field_rmse_ms": math.sqrt(
                    sum(float(row["field_squared_error_sum"]) for row in rows)
                    / field_count
                ),
                "field_bias_ms": sum(
                    float(row["field_signed_error_sum"]) for row in rows
                )
                / field_count,
            }.items():
                self.log(
                    f"val_ri/{name}",
                    torch.tensor(value, device=self.device),
                    on_epoch=True,
                    sync_dist=False,
                )

    def _log_reconstruction(self, batch: Mapping[str, Any], wandb_key: str) -> None:
        prediction = self.predict_joint(batch)
        log_wandb_reconstruction(
            self,
            batch,
            prediction.central_physical,
            wandb_key=wandb_key,
            target_batch=batch["target_physical"],
            intensity_prediction_batch=prediction.intensity_prediction_ms,
            intensity_target_batch=batch["intensity_target_ms"],
            physical_output_units="m s-1",
        )

    @staticmethod
    def _single_channel(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError(f"{name} must have shape [batch, 1, height, width]")
        return tensor


__all__ = [
    "BottleneckUNetMLP",
    "BottleneckUNetMLPRegressor",
    "JointPredictionBatch",
    "JointUNetOutput",
]
