"""Encoder-only IBTrACS intensity regression."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from geo2wf.data.contracts import DataSpec
from geo2wf.data.intensity import (
    category_macro_f1_tensor,
    tropical_category_from_wind_ms_tensor,
)
from geo2wf.data.joint_intensity import (
    IBTRACS_STRUCTURE_COMPANION,
    IBTRACS_STRUCTURE_TARGET_NAMES,
    INTENSITY_TARGET_COMPANION,
    ibtracs_structure_targets,
)

from .module import BottleneckEncoderMLP, EncoderMLPOutput, _huber_values


class BottleneckEncoderMLPRegressor(pl.LightningModule):
    """Regress IBTrACS USA_WIND from a U-Net encoder without a decoder."""

    checkpoint_monitor = "val/intensity_mae_ms"
    checkpoint_mode = "min"

    def __init__(
        self,
        condition_channels: int = 23,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        intensity_hidden_features: int = 128,
        intensity_dropout: float = 0.1,
        initial_intensity_ms: float = 25.0,
        intensity_huber_delta_ms: float = 5.0,
        structure_head_enabled: bool = False,
        structure_loss_weight: float = 0.0,
        structure_huber_delta_km: float = 20.0,
        lr: float = 2.0e-4,
        weight_decay: float = 1.0e-4,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 25,
        lr_scheduler_cooldown: int = 0,
        lr_scheduler_min_lr: float = 0.0,
    ) -> None:
        super().__init__()
        if condition_channels <= 0:
            raise ValueError("condition_channels must be positive")
        if intensity_huber_delta_ms <= 0.0:
            raise ValueError("intensity_huber_delta_ms must be positive")
        if structure_loss_weight < 0.0 or structure_huber_delta_km <= 0.0:
            raise ValueError(
                "structure loss weight/delta must be non-negative/positive"
            )
        if structure_loss_weight > 0.0 and not structure_head_enabled:
            raise ValueError("positive structure loss requires structure_head_enabled")
        self.save_hyperparameters()
        self.condition_channels = int(condition_channels)
        self.intensity_huber_delta_ms = float(intensity_huber_delta_ms)
        self.structure_head_enabled = bool(structure_head_enabled)
        self.structure_loss_weight = float(structure_loss_weight)
        self.structure_huber_delta_km = float(structure_huber_delta_km)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.lr_scheduler_factor = float(lr_scheduler_factor)
        self.lr_scheduler_patience = int(lr_scheduler_patience)
        self.lr_scheduler_cooldown = int(lr_scheduler_cooldown)
        self.lr_scheduler_min_lr = float(lr_scheduler_min_lr)
        self.model = BottleneckEncoderMLP(
            self.condition_channels + 1,
            base_channels,
            tuple(channel_mults),
            intensity_hidden_features,
            intensity_dropout,
            initial_intensity_ms,
            len(IBTRACS_STRUCTURE_TARGET_NAMES) if self.structure_head_enabled else 0,
        )
        self._category_rows: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
            "val": [],
            "test": [],
        }

    def validate_data_spec(self, spec: DataSpec) -> None:
        if spec.condition_channel_count != self.condition_channels:
            raise ValueError(
                f"model expects {self.condition_channels} condition channels but "
                f"data provides {spec.condition_channel_count}"
            )
        if spec.target_channel_count:
            raise ValueError(
                "encoder-only intensity data must not provide raster targets"
            )
        if INTENSITY_TARGET_COMPANION not in spec.companions:
            raise ValueError("data must provide continuous IBTrACS intensity labels")
        if (
            self.structure_loss_weight > 0.0
            and IBTRACS_STRUCTURE_COMPANION not in spec.companions
        ):
            raise ValueError("structure-supervised encoder requires IBTrACS radii")

    @staticmethod
    def _single_channel(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError(f"{name} must have shape [B,1,H,W]")
        return tensor

    def forward(
        self, condition: torch.Tensor, condition_mask: torch.Tensor
    ) -> EncoderMLPOutput:
        if condition.ndim != 4 or condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"condition must have {self.condition_channels} channels, got "
                f"{tuple(condition.shape)}"
            )
        mask = self._single_channel(condition_mask, "condition_mask").to(condition)
        return self.model(torch.cat([condition * mask, mask], dim=1))

    def predict_batch(self, batch: Mapping[str, Any]) -> EncoderMLPOutput:
        return self(batch["condition"], batch["condition_mask"])

    def _target(
        self, batch: Mapping[str, Any], reference: torch.Tensor
    ) -> torch.Tensor:
        if "intensity_target_ms" not in batch:
            raise KeyError("batch is missing intensity_target_ms")
        target = torch.as_tensor(batch["intensity_target_ms"], device=reference.device)
        target = target.to(reference).reshape(-1)
        if target.shape != reference.shape:
            raise ValueError("intensity_target_ms must have one value per sample")
        if not torch.isfinite(target).all() or bool((target < 0.0).any()):
            raise ValueError("intensity_target_ms must be finite and non-negative")
        return target

    def _metrics(
        self, batch: Mapping[str, Any]
    ) -> tuple[EncoderMLPOutput, torch.Tensor, dict[str, torch.Tensor]]:
        output = self.predict_batch(batch)
        target = self._target(batch, output.intensity_prediction_ms)
        error = output.intensity_prediction_ms - target
        intensity_loss = _huber_values(error, self.intensity_huber_delta_ms).mean()
        structure_loss = target.sum() * 0.0
        metrics = {
            "intensity_loss": intensity_loss,
            "intensity_mae_ms": error.abs().mean(),
            "intensity_rmse_ms": error.square().mean().sqrt(),
            "intensity_bias_ms": error.mean(),
        }
        if output.structure_prediction_km is not None:
            targets = ibtracs_structure_targets(batch, output.structure_prediction_km)
            if targets is None:
                if self.structure_loss_weight > 0.0:
                    raise KeyError(
                        "structure-supervised encoder requires IBTrACS targets"
                    )
            else:
                structure_target, structure_valid = targets
                safe_target = torch.where(
                    structure_valid,
                    structure_target,
                    output.structure_prediction_km.detach(),
                )
                values = torch.nn.functional.smooth_l1_loss(
                    output.structure_prediction_km,
                    safe_target,
                    reduction="none",
                    beta=self.structure_huber_delta_km,
                )
                weights = structure_valid.to(values)
                structure_loss = (values * weights).sum() / weights.sum().clamp_min(1.0)
                metrics["structure_loss"] = structure_loss
                structure_error = output.structure_prediction_km - safe_target
                for index, name in enumerate(IBTRACS_STRUCTURE_TARGET_NAMES):
                    selected = structure_valid[:, index]
                    if selected.any():
                        selected_error = structure_error[selected, index]
                        metrics[f"structure_{name}_mae_km"] = (
                            selected_error.abs().mean()
                        )
                        metrics[f"structure_{name}_rmse_km"] = (
                            selected_error.square().mean().sqrt()
                        )
                        metrics[f"structure_{name}_bias_km"] = selected_error.mean()
        metrics["loss"] = intensity_loss + self.structure_loss_weight * structure_loss
        return output, target, metrics

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        _, _, metrics = self._metrics(batch)
        batch_size = int(batch["condition"].shape[0])
        for name, value in metrics.items():
            self.log(
                f"train/{name}",
                value,
                on_step=name == "loss",
                on_epoch=True,
                prog_bar=name == "loss",
                sync_dist=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
        return metrics["loss"]

    def _evaluation_step(self, split: str, batch: Mapping[str, Any]) -> None:
        output, target, metrics = self._metrics(batch)
        batch_size = int(target.numel())
        prediction_category = tropical_category_from_wind_ms_tensor(
            output.intensity_prediction_ms.detach()
        )
        target_category = tropical_category_from_wind_ms_tensor(target.detach())
        self._category_rows[split].append(
            (prediction_category.cpu(), target_category.cpu())
        )
        for name, value in metrics.items():
            self.log(
                f"{split}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name == "intensity_mae_ms",
                sync_dist=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
        ri = torch.as_tensor(
            batch.get(
                "is_rapid_intensification",
                torch.zeros_like(target, dtype=torch.bool),
            ),
            device=target.device,
            dtype=torch.bool,
        ).reshape(-1)
        if ri.any():
            prefix = f"{split}_ri"
            ri_error = output.intensity_prediction_ms[ri] - target[ri]
            ri_metrics = {
                "intensity_mae_ms": ri_error.abs().mean(),
                "intensity_rmse_ms": ri_error.square().mean().sqrt(),
                "intensity_bias_ms": ri_error.mean(),
            }
            if output.structure_prediction_km is not None:
                targets = ibtracs_structure_targets(
                    batch, output.structure_prediction_km
                )
                if targets is not None:
                    structure_target, structure_valid = targets
                    structure_valid = structure_valid & ri[:, None]
                    safe_target = torch.where(
                        structure_valid,
                        structure_target,
                        output.structure_prediction_km.detach(),
                    )
                    structure_error = output.structure_prediction_km - safe_target
                    for index, name in enumerate(IBTRACS_STRUCTURE_TARGET_NAMES):
                        selected = structure_valid[:, index]
                        if selected.any():
                            selected_error = structure_error[selected, index]
                            ri_metrics[f"structure_{name}_mae_km"] = (
                                selected_error.abs().mean()
                            )
                            ri_metrics[f"structure_{name}_rmse_km"] = (
                                selected_error.square().mean().sqrt()
                            )
                            ri_metrics[f"structure_{name}_bias_km"] = (
                                selected_error.mean()
                            )
            for name, value in ri_metrics.items():
                self.log(
                    f"{prefix}/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=int(ri.sum()),
                    add_dataloader_idx=False,
                )
            self.log(
                f"{prefix}/samples",
                ri.sum().to(torch.float32),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                reduce_fx="sum",
                add_dataloader_idx=False,
            )

    def validation_step(
        self, batch: Mapping[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        del batch_idx
        if dataloader_idx == 0:
            self._evaluation_step("val", batch)

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        del batch_idx
        self._evaluation_step("test", batch)

    def on_validation_epoch_start(self) -> None:
        self._category_rows["val"] = []

    def on_validation_epoch_end(self) -> None:
        self._log_category_metrics("val")

    def on_test_epoch_start(self) -> None:
        self._category_rows["test"] = []

    def on_test_epoch_end(self) -> None:
        self._log_category_metrics("test")

    def _log_category_metrics(self, split: str) -> None:
        rows = self._category_rows[split]
        if not rows:
            return
        prediction = torch.cat([row[0] for row in rows])
        target = torch.cat([row[1] for row in rows])
        self.log(
            f"{split}/category_accuracy",
            (prediction == target).float().mean(),
            sync_dist=True,
        )
        self.log(
            f"{split}/category_macro_f1",
            category_macro_f1_tensor(prediction, target).float(),
            sync_dist=True,
        )

    def predict_step(
        self, batch: Mapping[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> EncoderMLPOutput:
        del batch_idx, dataloader_idx
        return self.predict_batch(batch)

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


__all__ = ["BottleneckEncoderMLPRegressor"]
