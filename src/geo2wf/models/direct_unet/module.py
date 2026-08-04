from __future__ import annotations

from collections.abc import Sequence
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from geo2wf.data.contracts import DataSpec
from geo2wf.models.base import (
    LossOutput,
    PredictionBatch,
    PredictionRequest,
    WindFieldLightningModule,
)
from geo2wf.models.deterministic_residual import ResidualUNet
from geo2wf.models.deterministic_residual.module import masked_huber_loss
from geo2wf.tracking.reconstruction_media import log_wandb_reconstruction


class DirectUNetRegressor(WindFieldLightningModule):
    """Predict one bounded image channel and optimize it in physical units."""

    checkpoint_monitor = "val/rmse_k"
    checkpoint_mode = "min"

    def __init__(
        self,
        condition_channels: int = 23,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        huber_delta_k: float = 2.0,
        lr: float = 2.0e-4,
        weight_decay: float = 1.0e-4,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 25,
        lr_scheduler_monitor: str = "val/rmse_k",
        lr_scheduler_cooldown: int = 0,
        lr_scheduler_min_lr: float = 0.0,
        validation_reconstruction_batches: int = 1,
        log_reconstruction_images: bool = True,
    ) -> None:
        super().__init__()
        if condition_channels <= 0 or huber_delta_k <= 0:
            raise ValueError("channel count and Huber delta must be positive")
        if validation_reconstruction_batches < 1:
            raise ValueError("validation_reconstruction_batches must be positive")
        self.save_hyperparameters()
        self.condition_channels, self.huber_delta_k = int(condition_channels), float(
            huber_delta_k
        )
        self.lr, self.weight_decay = float(lr), float(weight_decay)
        self.lr_scheduler_factor, self.lr_scheduler_patience = float(
            lr_scheduler_factor
        ), int(lr_scheduler_patience)
        self.lr_scheduler_monitor, self.lr_scheduler_cooldown = str(
            lr_scheduler_monitor
        ), int(lr_scheduler_cooldown)
        self.lr_scheduler_min_lr = float(lr_scheduler_min_lr)
        self.validation_reconstruction_batches = int(validation_reconstruction_batches)
        self.log_reconstruction_images = bool(log_reconstruction_images)
        self.model = ResidualUNet(
            self.condition_channels + 1, base_channels, tuple(channel_mults)
        )
        self.register_buffer(
            "_validation_statistics",
            torch.zeros(5, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_test_statistics", torch.zeros(5, dtype=torch.float64), persistent=False
        )

    def validate_data_spec(self, spec: DataSpec) -> None:
        super().validate_data_spec(spec)
        if spec.target_channel_count != 1:
            raise ValueError(
                f"DirectUNetRegressor requires one target: {spec.target_channels}"
            )
        if spec.target_units != "K":
            raise ValueError(
                f"DirectUNetRegressor requires target units K, got {spec.target_units!r}"
            )

    def forward(
        self, condition: torch.Tensor, condition_mask: torch.Tensor
    ) -> torch.Tensor:
        if condition.ndim != 4 or condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"condition must have {self.condition_channels} channels, got {tuple(condition.shape)}"
            )
        mask = self._single_channel(condition_mask, "condition_mask").to(condition)
        return torch.sigmoid(self.model(torch.cat([condition * mask, mask], dim=1)))

    def predict_normalized(self, batch) -> torch.Tensor:
        return self(batch["condition"], batch["condition_mask"])

    def predict_physical(self, batch) -> torch.Tensor:
        prediction = self.predict_normalized(batch)
        offset, scale = self._normalization_parameters(batch, prediction)
        return prediction * scale + offset

    def compute_training_objective(self, batch) -> LossOutput:
        prediction = self.predict_physical(batch)
        target, mask = self._target_and_mask(batch, prediction)
        loss = masked_huber_loss(prediction, target, mask, delta=self.huber_delta_k)
        count, error = mask.sum().clamp_min(1.0), prediction - target
        return LossOutput(
            loss,
            {
                "mae_k": (error.abs() * mask).sum() / count,
                "bias_k": (error * mask).sum() / count,
            },
        )

    @torch.no_grad()
    def predict_batch(self, batch, request: PredictionRequest) -> PredictionBatch:
        prediction = self.predict_physical(batch)
        members = prediction.unsqueeze(1).expand(-1, request.ensemble_size, -1, -1, -1)
        return PredictionBatch(members, prediction)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if dataloader_idx == 1:
            if self.log_reconstruction_images and batch_idx == 0:
                self._log_reconstruction(batch, "images/train_reconstruction")
            return
        if dataloader_idx:
            return
        prediction = self.predict_physical(batch)
        target, mask = self._target_and_mask(batch, prediction)
        self._accumulate(self._validation_statistics, prediction, target, mask)
        if (
            self.log_reconstruction_images
            and batch_idx < self.validation_reconstruction_batches
        ):
            self._log_reconstruction(batch, "images/val_reconstruction")

    def on_validation_epoch_start(self) -> None:
        self._validation_statistics.zero_()

    def on_validation_epoch_end(self) -> None:
        self._log_statistics("val", self._validation_statistics)

    @torch.no_grad()
    def test_step(self, batch, batch_idx: int) -> None:
        del batch_idx
        prediction = self.predict_physical(batch)
        target, mask = self._target_and_mask(batch, prediction)
        self._accumulate(self._test_statistics, prediction, target, mask)

    def on_test_epoch_start(self) -> None:
        self._test_statistics.zero_()

    def on_test_epoch_end(self) -> None:
        self._log_statistics("test", self._test_statistics)

    def configure_optimizers(self) -> dict:
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
                "monitor": self.lr_scheduler_monitor,
            },
        }

    def _target_and_mask(self, batch, reference):
        target = self._single_channel(batch["target_physical"], "target_physical").to(
            reference
        )
        target_mask = self._single_channel(batch["target_mask"], "target_mask")
        condition_mask = self._single_channel(batch["condition_mask"], "condition_mask")
        return target, (target_mask.bool() & condition_mask.bool()).to(reference)

    @staticmethod
    def _normalization_parameters(batch, reference):
        offset, scale = batch["target_norm_offset"].to(reference), batch[
            "target_norm_scale"
        ].to(reference)
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

    def _accumulate(self, statistics, prediction, target, mask) -> None:
        error, absolute = prediction - target, (prediction - target).abs()
        huber = torch.where(
            absolute <= self.huber_delta_k,
            0.5 * error.square(),
            self.huber_delta_k * (absolute - 0.5 * self.huber_delta_k),
        )
        statistics.add_(
            torch.stack(
                [
                    mask.sum(),
                    (absolute * mask).sum(),
                    (error.square() * mask).sum(),
                    (error * mask).sum(),
                    (huber * mask).sum(),
                ]
            ).to(statistics)
        )

    def _log_statistics(self, prefix: str, statistics: torch.Tensor) -> None:
        values = statistics.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values)
        count = values[0].clamp_min(1.0)
        metrics = {
            "loss": values[4] / count,
            "mae_k": values[1] / count,
            "rmse_k": torch.sqrt(values[2] / count),
            "bias_k": values[3] / count,
        }
        for name, value in metrics.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name in {"loss", "rmse_k"},
                sync_dist=False,
            )

    def _log_reconstruction(self, batch, wandb_key: str) -> None:
        log_wandb_reconstruction(
            self,
            batch,
            self.predict_physical(batch),
            wandb_key=wandb_key,
            target_batch=batch["target_physical"],
            physical_output_units="K",
        )

    @staticmethod
    def _single_channel(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError(f"{name} must have shape [batch, 1, height, width]")
        return tensor
