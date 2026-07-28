from __future__ import annotations

from collections.abc import Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .reconstruction_logging import log_wandb_reconstruction
from .wind_metrics import RADIAL_METRIC_NAMES, radial_wind_metric_statistics


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return a group count with at least two channels per group when possible."""
    upper_bound = min(maximum, max(channels // 2, 1))
    for groups in range(upper_bound, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    """Small GroupNorm residual block that is stable for tiny image batches."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class ResidualUNet(nn.Module):
    """Compact deterministic U-Net used only by the ERA5 residual baseline."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        if not channel_mults or any(multiplier <= 0 for multiplier in channel_mults):
            raise ValueError("channel_mults must contain positive integers")

        dimensions = [base_channels * int(multiplier) for multiplier in channel_mults]
        self.stem = nn.Conv2d(in_channels, dimensions[0], kernel_size=3, padding=1)
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

        self.head = nn.Conv2d(dimensions[0], 1, kernel_size=1)
        # The first prediction is exactly the ERA5 field. This makes the baseline
        # useful before training and asks the network to learn only corrections.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = []
        for index, block in enumerate(self.encoder):
            x = block(x)
            skips.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)

        x = self.bottleneck(x)
        for projection, block, skip in zip(
            self.decoder_projections,
            self.decoder,
            reversed(skips[:-1]),
        ):
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = projection(x)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    """Huber loss averaged over valid pixels, retaining a zero-loss gradient."""
    if delta <= 0:
        raise ValueError("delta must be positive")
    mask = mask.to(device=prediction.device, dtype=prediction.dtype)
    absolute_error = (prediction - target).abs()
    pointwise = torch.where(
        absolute_error <= delta,
        0.5 * absolute_error.square(),
        delta * (absolute_error - 0.5 * delta),
    )
    return (pointwise * mask).sum() / mask.sum().clamp_min(1.0)


class ERA5ResidualRegressor(pl.LightningModule):
    """Predict a deterministic physical-wind correction to the ERA5 field.

    The data set supplies an ERA5 wind speed transformed like the SAR target for
    network input, plus its physical m/s value for the residual connection. Loss
    and validation metrics are calculated in m/s over the joint valid mask.
    """

    checkpoint_monitor = "val/eye_structure_score"
    checkpoint_mode = "min"

    # count, |error|, error^2, signed error, Huber, ERA5 |error|,
    # ERA5 error^2, high-wind count, high-wind |error|, high-wind ERA5 |error|
    _STAT_COUNT = 10

    def __init__(
        self,
        condition_channels: int = 19,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        huber_delta_ms: float = 2.0,
        off_swath_anchor_weight: float = 0.05,
        high_wind_threshold_ms: float = 17.0,
        prediction_min_ms: float | None = 0.0,
        prediction_max_ms: float | None = None,
        psnr_data_range_ms: float = 79.8,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 10,
        validation_reconstruction_batches: int = 1,
    ) -> None:
        super().__init__()
        if condition_channels <= 0:
            raise ValueError("condition_channels must be positive")
        if psnr_data_range_ms <= 0:
            raise ValueError("psnr_data_range_ms must be positive")
        if off_swath_anchor_weight < 0:
            raise ValueError("off_swath_anchor_weight must be non-negative")
        if validation_reconstruction_batches < 1:
            raise ValueError("validation_reconstruction_batches must be positive")
        self.save_hyperparameters()

        self.condition_channels = int(condition_channels)
        self.huber_delta_ms = float(huber_delta_ms)
        self.off_swath_anchor_weight = float(off_swath_anchor_weight)
        self.high_wind_threshold_ms = float(high_wind_threshold_ms)
        self.prediction_min_ms = prediction_min_ms
        self.prediction_max_ms = prediction_max_ms
        self.psnr_data_range_ms = float(psnr_data_range_ms)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.lr_scheduler_factor = float(lr_scheduler_factor)
        self.lr_scheduler_patience = int(lr_scheduler_patience)
        self.validation_reconstruction_batches = int(
            validation_reconstruction_batches
        )

        # Raw condition + condition-valid mask + explicit ERA5 wind + ERA5 mask.
        self.model = ResidualUNet(
            in_channels=self.condition_channels + 3,
            base_channels=base_channels,
            channel_mults=tuple(channel_mults),
        )
        self.register_buffer(
            "_validation_statistics",
            torch.zeros(self._STAT_COUNT, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_test_statistics",
            torch.zeros(self._STAT_COUNT, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_validation_radial_statistics",
            torch.zeros((len(RADIAL_METRIC_NAMES), 2), dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_test_radial_statistics",
            torch.zeros((len(RADIAL_METRIC_NAMES), 2), dtype=torch.float64),
            persistent=False,
        )

    def forward(
        self,
        condition: torch.Tensor,
        condition_mask: torch.Tensor,
        era5_wind_speed: torch.Tensor,
        era5_wind_speed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the learned residual in physical m/s."""
        if condition.ndim != 4:
            raise ValueError("condition must have shape [batch, channel, height, width]")
        if condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"expected {self.condition_channels} condition channels, "
                f"got {condition.shape[1]}"
            )
        condition_mask = self._single_channel(
            condition_mask, "condition_mask", collapse_mask=True
        )
        era5_wind_speed = self._single_channel(
            era5_wind_speed, "era5_wind_speed"
        )
        era5_wind_speed_mask = self._single_channel(
            era5_wind_speed_mask,
            "era5_wind_speed_mask",
            collapse_mask=True,
        )
        condition_mask = condition_mask.to(condition.dtype)
        era5_wind_speed_mask = era5_wind_speed_mask.to(condition.dtype)
        features = torch.cat(
            [
                condition * condition_mask,
                condition_mask,
                era5_wind_speed.to(condition.dtype) * era5_wind_speed_mask,
                era5_wind_speed_mask,
            ],
            dim=1,
        )
        return self.model(features)

    def predict_residual_ms(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict the physical ERA5 correction for one collated batch."""
        self._require_prediction_keys(batch)
        return self(
            batch["condition"],
            batch["condition_mask"],
            batch["era5_wind_speed"],
            batch["era5_wind_speed_mask"],
        )

    def predict_physical(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return one deterministic wind-speed reconstruction in m/s."""
        residual = self.predict_residual_ms(batch)
        era5_physical = self._single_channel(
            batch["era5_wind_speed_physical"],
            "era5_wind_speed_physical",
        ).to(device=residual.device, dtype=residual.dtype)
        return self._bound_prediction(era5_physical + residual)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        prediction, target, valid_mask, era5 = self._batch_outputs(batch)
        reconstruction_loss = masked_huber_loss(
            prediction,
            target,
            valid_mask,
            delta=self.huber_delta_ms,
        )
        anchor_mask = self._off_swath_mask(batch, prediction)
        anchor_loss = masked_huber_loss(
            prediction - era5,
            torch.zeros_like(prediction),
            anchor_mask,
            delta=self.huber_delta_ms,
        )
        loss = (
            reconstruction_loss
            + self.off_swath_anchor_weight * anchor_loss
        )
        valid_count = valid_mask.sum().clamp_min(1)
        mae = ((self._bound_prediction(prediction) - target).abs() * valid_mask).sum()
        mae = mae / valid_count
        era5_mae = ((self._bound_prediction(era5) - target).abs() * valid_mask).sum()
        era5_mae = era5_mae / valid_count
        batch_size = int(target.shape[0])
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train/reconstruction_loss",
            reconstruction_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train/off_swath_anchor_loss",
            anchor_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train/mae_ms",
            mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train/era5_mae_ms",
            era5_mae,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    @torch.no_grad()
    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Loader 1 is the fixed train preview used only for image logging.
        if dataloader_idx == 1:
            if batch_idx == 0:
                self._log_reconstruction(
                    batch,
                    self.predict_physical(batch),
                    wandb_key="images/train_reconstruction",
                )
            return None
        if dataloader_idx != 0:
            return None
        prediction, target, valid_mask, era5 = self._batch_outputs(batch)
        bounded_prediction = self._bound_prediction(prediction)
        self._accumulate_statistics(
            self._validation_statistics,
            bounded_prediction,
            target,
            valid_mask,
            self._bound_prediction(era5),
            raw_prediction=prediction,
        )
        self._accumulate_radial_statistics(
            self._validation_radial_statistics,
            self._bound_prediction(prediction),
            target,
            valid_mask,
            batch,
        )
        if batch_idx < self.validation_reconstruction_batches:
            self._log_reconstruction(batch, bounded_prediction)
        return None

    def on_validation_epoch_start(self) -> None:
        self._validation_statistics.zero_()
        self._validation_radial_statistics.zero_()

    def on_validation_epoch_end(self) -> None:
        self._log_statistics("val", self._validation_statistics)
        self._log_radial_statistics("val", self._validation_radial_statistics)

    @torch.no_grad()
    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        del batch_idx
        prediction, target, valid_mask, era5 = self._batch_outputs(batch)
        self._accumulate_statistics(
            self._test_statistics,
            self._bound_prediction(prediction),
            target,
            valid_mask,
            self._bound_prediction(era5),
            raw_prediction=prediction,
        )
        self._accumulate_radial_statistics(
            self._test_radial_statistics,
            self._bound_prediction(prediction),
            target,
            valid_mask,
            batch,
        )
        return None

    def on_test_epoch_start(self) -> None:
        self._test_statistics.zero_()
        self._test_radial_statistics.zero_()

    def on_test_epoch_end(self) -> None:
        self._log_statistics("test", self._test_statistics)
        self._log_radial_statistics("test", self._test_radial_statistics)

    @torch.no_grad()
    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        del batch_idx, dataloader_idx
        return self.predict_physical(batch)

    def _log_reconstruction(
        self,
        batch: dict[str, torch.Tensor],
        prediction: torch.Tensor,
        *,
        wandb_key: str = "images/val_reconstruction",
    ) -> None:
        """Log physical-wind reconstructions through the shared W&B helper."""
        log_wandb_reconstruction(
            self,
            batch,
            prediction,
            wandb_key=wandb_key,
            target_batch=batch["target_physical"],
        )

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.checkpoint_monitor,
            },
        }

    def _batch_outputs(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._require_supervised_batch_keys(batch)
        residual = self.predict_residual_ms(batch)
        target = self._single_channel(
            batch["target_physical"], "target_physical"
        ).to(device=residual.device, dtype=residual.dtype)
        era5 = self._single_channel(
            batch["era5_wind_speed_physical"],
            "era5_wind_speed_physical",
        ).to(device=residual.device, dtype=residual.dtype)
        prediction = era5 + residual
        valid_mask = self._valid_mask(batch, prediction)
        return prediction, target, valid_mask, era5

    def _valid_mask(
        self, batch: dict[str, torch.Tensor], reference: torch.Tensor
    ) -> torch.Tensor:
        target_mask = self._single_channel(
            batch["target_mask"], "target_mask", collapse_mask=True
        )
        condition_mask = self._single_channel(
            batch["condition_mask"], "condition_mask", collapse_mask=True
        )
        era5_mask = self._single_channel(
            batch["era5_wind_speed_mask"],
            "era5_wind_speed_mask",
            collapse_mask=True,
        )
        return (
            target_mask.bool() & condition_mask.bool() & era5_mask.bool()
        ).to(device=reference.device, dtype=reference.dtype)

    def _off_swath_mask(
        self, batch: dict[str, torch.Tensor], reference: torch.Tensor
    ) -> torch.Tensor:
        """Select valid ERA5/GEO pixels outside the observed SAR swath."""
        target_mask = self._single_channel(
            batch["target_mask"], "target_mask", collapse_mask=True
        )
        condition_mask = self._single_channel(
            batch["condition_mask"], "condition_mask", collapse_mask=True
        )
        era5_mask = self._single_channel(
            batch["era5_wind_speed_mask"],
            "era5_wind_speed_mask",
            collapse_mask=True,
        )
        return (
            ~target_mask.bool() & condition_mask.bool() & era5_mask.bool()
        ).to(device=reference.device, dtype=reference.dtype)

    def _bound_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.prediction_min_ms is not None:
            prediction = prediction.clamp_min(float(self.prediction_min_ms))
        if self.prediction_max_ms is not None:
            prediction = prediction.clamp_max(float(self.prediction_max_ms))
        return prediction

    def _accumulate_statistics(
        self,
        statistics: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        era5: torch.Tensor,
        *,
        raw_prediction: torch.Tensor,
    ) -> None:
        mask = valid_mask.to(prediction.dtype)
        error = prediction - target
        era5_error = era5 - target
        absolute_error = error.abs()
        era5_absolute_error = era5_error.abs()
        raw_absolute_error = (raw_prediction - target).abs()
        huber = torch.where(
            raw_absolute_error <= self.huber_delta_ms,
            0.5 * raw_absolute_error.square(),
            self.huber_delta_ms
            * (raw_absolute_error - 0.5 * self.huber_delta_ms),
        )
        high_wind_mask = mask * (target >= self.high_wind_threshold_ms).to(mask.dtype)
        additions = torch.stack(
            [
                mask.sum(),
                (absolute_error * mask).sum(),
                (error.square() * mask).sum(),
                (error * mask).sum(),
                (huber * mask).sum(),
                (era5_absolute_error * mask).sum(),
                (era5_error.square() * mask).sum(),
                high_wind_mask.sum(),
                (absolute_error * high_wind_mask).sum(),
                (era5_absolute_error * high_wind_mask).sum(),
            ]
        )
        statistics.add_(additions.to(statistics))

    def _log_statistics(self, prefix: str, statistics: torch.Tensor) -> None:
        statistics = self._distributed_sum(statistics)
        count = statistics[0]
        if count <= 0:
            return
        mae = statistics[1] / count
        mse = statistics[2] / count
        era5_mae = statistics[5] / count
        era5_mse = statistics[6] / count
        metrics = {
            f"{prefix}/loss": statistics[4] / count,
            f"{prefix}/mae_ms": mae,
            f"{prefix}/rmse_ms": mse.sqrt(),
            f"{prefix}/bias_ms": statistics[3] / count,
            f"{prefix}/psnr_db": 20.0
            * torch.log10(
                mae.new_tensor(self.psnr_data_range_ms)
                / mse.clamp_min(1e-12).sqrt()
            ),
            f"{prefix}/era5_mae_ms": era5_mae,
            f"{prefix}/era5_rmse_ms": era5_mse.sqrt(),
            f"{prefix}/mae_skill_vs_era5": 1.0
            - mae / era5_mae.clamp_min(1e-12),
        }
        high_count = statistics[7]
        if high_count > 0:
            metrics[f"{prefix}/high_wind_mae_ms"] = statistics[8] / high_count
            metrics[f"{prefix}/high_wind_era5_mae_ms"] = (
                statistics[9] / high_count
            )
        for name, value in metrics.items():
            self.log(
                name,
                value.to(dtype=torch.float32),
                on_step=False,
                on_epoch=True,
                prog_bar=name in {
                    f"{prefix}/mae_ms",
                    f"{prefix}/era5_mae_ms",
                },
                logger=True,
                sync_dist=False,
            )

    def _accumulate_radial_statistics(
        self,
        statistics: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> None:
        """Accumulate fixed-shape radial statistics on every rank."""
        if not {"center", "target_bounds"}.issubset(batch):
            return
        additions = radial_wind_metric_statistics(
            prediction,
            target,
            valid_mask,
            batch["center"],
            batch["target_bounds"],
        )
        statistics.add_(additions.to(statistics))

    def _log_radial_statistics(
        self, prefix: str, statistics: torch.Tensor
    ) -> None:
        """All-reduce once, then log the globally available radial means."""
        statistics = self._distributed_sum(statistics)
        means = {}
        for index, name in enumerate(RADIAL_METRIC_NAMES):
            count = statistics[index, 1]
            if count <= 0:
                continue
            value = (statistics[index, 0] / count).to(dtype=torch.float32)
            means[name] = value
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )
        score_terms = {
            "eye_mae_ms": 0.5,
            "inner_core_mae_ms": 1.0,
            "radial_profile_mae_ms": 1.0,
            "rmw_error_km": 0.1,
            "eye_to_eyewall_contrast_error_ms": 1.0,
        }
        if score_terms.keys() <= means.keys():
            score = sum(means[name] * weight for name, weight in score_terms.items())
            self.log(
                f"{prefix}/eye_structure_score",
                score,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=False,
            )

    def _distributed_sum(self, statistics: torch.Tensor) -> torch.Tensor:
        trainer = getattr(self, "_trainer", None)
        if trainer is None or trainer.world_size <= 1:
            return statistics
        gathered = self.all_gather(statistics)
        return gathered.reshape(-1, *statistics.shape).sum(dim=0)

    @staticmethod
    def _single_channel(
        tensor: torch.Tensor,
        name: str,
        *,
        collapse_mask: bool = False,
    ) -> torch.Tensor:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [batch, channel, height, width]")
        if tensor.shape[1] != 1:
            if not collapse_mask:
                raise ValueError(f"{name} must contain exactly one channel")
            tensor = tensor.bool().all(dim=1, keepdim=True)
        return tensor

    @staticmethod
    def _require_prediction_keys(batch: dict[str, torch.Tensor]) -> None:
        required = {
            "condition",
            "condition_mask",
            "era5_wind_speed",
            "era5_wind_speed_physical",
            "era5_wind_speed_mask",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(
                "ERA5 residual batches require: " + ", ".join(missing)
            )

    @classmethod
    def _require_supervised_batch_keys(
        cls, batch: dict[str, torch.Tensor]
    ) -> None:
        cls._require_prediction_keys(batch)
        missing = sorted({"target_mask", "target_physical"}.difference(batch))
        if missing:
            raise KeyError(
                "ERA5 residual training batches require: " + ", ".join(missing)
            )
