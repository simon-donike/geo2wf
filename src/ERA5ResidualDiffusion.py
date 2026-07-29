from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .ERA5Residual import ERA5ResidualRegressor
from .PixelDiffusion import PixelDiffusionConditional


_BASELINE_PHYSICAL = "_residual_diffusion_baseline_physical"
_BASELINE_NORMALIZED = "_residual_diffusion_baseline_normalized"
_BASELINE_MASK = "_residual_diffusion_baseline_mask"


class ERA5ResidualDiffusion(PixelDiffusionConditional):
    """Diffuse a signed correction to a dense wind-speed baseline.

    The generated variable is not absolute normalized wind. It is an odd,
    bounded transform of ``SAR wind - baseline wind`` in physical m/s. The
    baseline can be raw ERA5 speed or a frozen :class:`ERA5ResidualRegressor`.
    At sampling time the transform is inverted and the residual is added back
    to the same baseline, producing the normalized absolute-wind tensors used
    by the existing physical and eye-structure metrics.
    """

    def __init__(
        self,
        *,
        base_condition_channels: int,
        generated_channels: int = 1,
        baseline_source: str = "era5",
        baseline_model: ERA5ResidualRegressor | None = None,
        residual_soft_scale_ms: float = 5.0,
        residual_clip_ms: float = 80.0,
        prediction_min_ms: float | None = 0.0,
        prediction_max_ms: float | None = 80.0,
        **diffusion_kwargs: Any,
    ) -> None:
        if base_condition_channels <= 0:
            raise ValueError("base_condition_channels must be positive")
        if generated_channels != 1:
            raise ValueError("ERA5 residual diffusion currently generates one channel")
        if residual_soft_scale_ms <= 0:
            raise ValueError("residual_soft_scale_ms must be positive")
        if residual_clip_ms <= 0:
            raise ValueError("residual_clip_ms must be positive")
        if residual_clip_ms <= residual_soft_scale_ms:
            raise ValueError(
                "residual_clip_ms must exceed residual_soft_scale_ms"
            )
        if (
            prediction_min_ms is not None
            and prediction_max_ms is not None
            and prediction_max_ms <= prediction_min_ms
        ):
            raise ValueError("prediction_max_ms must exceed prediction_min_ms")

        baseline_source = str(baseline_source).strip().lower()
        if baseline_source not in {"era5", "deterministic"}:
            raise ValueError("baseline_source must be 'era5' or 'deterministic'")
        if baseline_source == "deterministic" and baseline_model is None:
            raise ValueError(
                "a frozen deterministic baseline_model is required when "
                "baseline_source='deterministic'"
            )
        if baseline_source == "era5" and baseline_model is not None:
            raise ValueError(
                "baseline_model must be omitted when baseline_source='era5'"
            )

        # The inherited condition consists of normalized GEO/ERA fields and one
        # aggregate validity mask. Add the exact target-scaled baseline and its
        # own mask so the denoiser does not need to rediscover the residual
        # connection from a differently normalized ERA channel.
        diffusion_condition_channels = int(base_condition_channels) + 2
        expected_model_channels = generated_channels + diffusion_condition_channels
        configured_model_channels = diffusion_kwargs.get("model_channels")
        if (
            configured_model_channels is not None
            and int(configured_model_channels) != expected_model_channels
        ):
            raise ValueError(
                "residual diffusion U-Net channels must equal one noisy residual "
                f"plus {diffusion_condition_channels} condition channels "
                f"({expected_model_channels} total), got {configured_model_channels}"
            )

        super().__init__(
            condition_channels=diffusion_condition_channels,
            generated_channels=generated_channels,
            **diffusion_kwargs,
        )
        self.base_condition_channels = int(base_condition_channels)
        self.baseline_source = baseline_source
        self.residual_soft_scale_ms = float(residual_soft_scale_ms)
        self.residual_clip_ms = float(residual_clip_ms)
        self.prediction_min_ms = prediction_min_ms
        self.prediction_max_ms = prediction_max_ms
        self.baseline_model = baseline_model
        if self.baseline_model is not None:
            self.baseline_model.requires_grad_(False)
            self.baseline_model.eval()
        self.register_buffer(
            "_validation_baseline_statistics",
            torch.zeros(3, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_test_baseline_statistics",
            torch.zeros(3, dtype=torch.float64),
            persistent=False,
        )

    def train(self, mode: bool = True):
        """Keep the optional deterministic baseline frozen in evaluation mode."""
        module = super().train(mode)
        if self.baseline_model is not None:
            self.baseline_model.eval()
        return module

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor], *, initial_noise=None):
        """Sample and return an absolute normalized wind reconstruction."""
        prepared = self._prepare_batch_context(batch)
        condition = self._prepare_condition(prepared["condition"], prepared)
        process = (
            self.ema_model
            if not self.training
            and self.ema_model is not None
            and self.ema_use_for_eval
            else self.model
        )
        residual_sample = process(condition, initial_noise=initial_noise)
        return self._sample_to_prediction(residual_sample, prepared)

    def encode_residual(self, residual_ms: torch.Tensor) -> torch.Tensor:
        """Map a physical signed residual to a bounded diffusion variable."""
        clipped = residual_ms.clamp(
            -self.residual_clip_ms, self.residual_clip_ms
        )
        denominator = torch.asinh(
            clipped.new_tensor(
                self.residual_clip_ms / self.residual_soft_scale_ms
            )
        )
        return torch.asinh(clipped / self.residual_soft_scale_ms) / denominator

    def decode_residual(self, encoded: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`encode_residual` back to physical m/s."""
        encoded = encoded.clamp(-1.0, 1.0)
        denominator = torch.asinh(
            encoded.new_tensor(
                self.residual_clip_ms / self.residual_soft_scale_ms
            )
        )
        return self.residual_soft_scale_ms * torch.sinh(encoded * denominator)

    def _prepare_batch_context(self, batch):
        if not isinstance(batch, dict):
            raise TypeError("ERA5 residual diffusion requires dictionary batches")
        if {
            _BASELINE_PHYSICAL,
            _BASELINE_NORMALIZED,
            _BASELINE_MASK,
        }.issubset(batch):
            return batch

        required = {
            "condition",
            "condition_mask",
            "era5_wind_speed",
            "era5_wind_speed_physical",
            "era5_wind_speed_mask",
            "target_norm_offset",
            "target_norm_scale",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(
                "ERA5 residual diffusion requires: " + ", ".join(missing)
            )

        prepared = dict(batch)
        baseline_physical = self._baseline_physical(prepared)
        baseline_mask = self._prepare_target_mask(
            prepared["era5_wind_speed_mask"], baseline_physical
        ).bool()
        finite = baseline_physical.isfinite().all(dim=1, keepdim=True)
        baseline_mask = baseline_mask & finite
        baseline_physical = torch.nan_to_num(
            baseline_physical, nan=0.0, posinf=0.0, neginf=0.0
        )
        offset, scale = self._target_affine(prepared, baseline_physical)
        baseline_normalized = ((baseline_physical - offset) / scale).clamp(0.0, 1.0)
        baseline_normalized = baseline_normalized * baseline_mask.to(
            baseline_normalized.dtype
        )
        prepared[_BASELINE_PHYSICAL] = baseline_physical
        prepared[_BASELINE_NORMALIZED] = baseline_normalized
        prepared[_BASELINE_MASK] = baseline_mask
        return prepared

    def _baseline_physical(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.baseline_source == "era5":
            return batch["era5_wind_speed_physical"].detach().to(
                device=batch["condition"].device,
                dtype=batch["condition"].dtype,
            )
        assert self.baseline_model is not None
        self.baseline_model.eval()
        with torch.no_grad():
            prediction = self.baseline_model.predict_physical(batch)
        # Clone to ensure a normal no-grad tensor is supplied to trainable
        # convolutions even if a future baseline uses inference-mode tensors.
        return prediction.detach().clone()

    def _prepare_condition(self, condition, batch):
        prepared = super()._prepare_condition(condition, batch)
        baseline = batch[_BASELINE_NORMALIZED].to(
            device=prepared.device, dtype=prepared.dtype
        )
        baseline_mask = batch[_BASELINE_MASK].to(
            device=prepared.device, dtype=prepared.dtype
        )
        baseline_feature = self.input_T(baseline) * baseline_mask
        result = torch.cat([prepared, baseline_feature, baseline_mask], dim=1)
        expected = self.base_condition_channels + 2
        if result.shape[1] != expected:
            raise ValueError(
                f"prepared residual condition has {result.shape[1]} channels; "
                f"expected {expected}"
            )
        return result

    def _prepare_diffusion_target(self, target, target_mask, batch):
        del target
        required = {"target_physical", "target_mask"}
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(
                "residual target construction requires: " + ", ".join(missing)
            )
        target_physical = batch["target_physical"].to(
            device=batch[_BASELINE_PHYSICAL].device,
            dtype=batch[_BASELINE_PHYSICAL].dtype,
        )
        observed = self._prepare_target_mask(
            target_mask, target_physical
        ).bool()
        baseline_valid = batch[_BASELINE_MASK].to(
            device=target_physical.device
        ).bool()
        jointly_observed = observed & baseline_valid
        encoded = self.encode_residual(
            target_physical - batch[_BASELINE_PHYSICAL]
        )
        diffusion_target = torch.where(
            jointly_observed, encoded, torch.zeros_like(encoded)
        )
        anchor_only = ~observed & baseline_valid
        loss_weight = jointly_observed.to(target_physical.dtype)
        if self.unobserved_loss_weight > 0:
            loss_weight = loss_weight + (
                anchor_only.to(target_physical.dtype)
                * self.unobserved_loss_weight
            )
        return diffusion_target, loss_weight

    def _target_to_model_space(self, target, batch):
        del batch
        # encode_residual already returns the diffusion process's [-1, 1] range.
        return target.clamp(-1.0, 1.0)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._validation_baseline_statistics.zero_()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self._log_baseline_statistics(
            "val", self._validation_baseline_statistics
        )

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._test_baseline_statistics.zero_()

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        self._log_baseline_statistics("test", self._test_baseline_statistics)

    def _accumulate_physical_statistics(self, statistics, prediction, batch):
        super()._accumulate_physical_statistics(statistics, prediction, batch)
        if statistics is self._validation_physical_statistics:
            baseline_statistics = self._validation_baseline_statistics
        elif statistics is self._test_physical_statistics:
            baseline_statistics = self._test_baseline_statistics
        else:
            return

        prepared = self._prepare_batch_context(batch)
        offset, scale = self._target_affine(prepared, prediction)
        prediction_ms = prediction * scale + offset
        target_ms = prepared["target_physical"].to(
            device=prediction.device, dtype=prediction.dtype
        )
        baseline_ms = prepared[_BASELINE_PHYSICAL].to(
            device=prediction.device, dtype=prediction.dtype
        )
        target_mask = self._prepare_target_mask(
            prepared["target_mask"], prediction
        )
        baseline_mask = prepared[_BASELINE_MASK].to(
            device=prediction.device, dtype=prediction.dtype
        )
        valid = target_mask * baseline_mask
        additions = prediction.new_zeros(3)
        additions[0] = valid.sum()
        additions[1] = ((prediction_ms - target_ms).abs() * valid).sum()
        additions[2] = ((baseline_ms - target_ms).abs() * valid).sum()
        baseline_statistics.add_(additions.to(baseline_statistics))

    def _log_baseline_statistics(self, prefix, statistics):
        statistics = self._distributed_sum_statistics(statistics)
        count = statistics[0]
        if count <= 0:
            return
        reconstruction_mae = statistics[1] / count
        baseline_mae = statistics[2] / count
        metrics = {
            "baseline_mae_ms": baseline_mae,
            "mae_skill_vs_baseline": (
                1.0
                - reconstruction_mae / baseline_mae.clamp_min(1e-12)
            ),
        }
        for name, value in metrics.items():
            self.log(
                f"{prefix}/{name}",
                value.to(torch.float32),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )

    def _sample_to_prediction(self, sample, batch):
        residual_ms = self.decode_residual(sample)
        prediction_ms = batch[_BASELINE_PHYSICAL].to(
            device=sample.device, dtype=sample.dtype
        ) + residual_ms
        if self.prediction_min_ms is not None:
            prediction_ms = prediction_ms.clamp_min(self.prediction_min_ms)
        if self.prediction_max_ms is not None:
            prediction_ms = prediction_ms.clamp_max(self.prediction_max_ms)
        offset, scale = self._target_affine(batch, prediction_ms)
        return ((prediction_ms - offset) / scale).clamp(0.0, 1.0)

    def _log_val_reconstruction(
        self, batch, pred_batch, *, wandb_key="images/val_reconstruction"
    ):
        """Plot baseline, refinement, and truth on one physical wind scale."""
        from .reconstruction_logging import log_wandb_reconstruction

        prepared = self._prepare_batch_context(batch)
        offset, scale = self._target_affine(prepared, pred_batch)
        refined_physical = pred_batch * scale + offset
        log_wandb_reconstruction(
            self,
            prepared,
            refined_physical,
            wandb_key=wandb_key,
            condition_batch=prepared["condition"],
            target_batch=prepared["target_physical"],
            baseline_batch=prepared[_BASELINE_PHYSICAL],
            physical_wind_output=True,
        )

    @staticmethod
    def _target_affine(batch, reference):
        offset = batch["target_norm_offset"].to(
            device=reference.device, dtype=reference.dtype
        )
        scale = batch["target_norm_scale"].to(
            device=reference.device, dtype=reference.dtype
        )
        while offset.ndim < reference.ndim:
            offset = offset.unsqueeze(-1)
            scale = scale.unsqueeze(-1)
        return offset, scale.clamp_min(1e-6)


def load_frozen_deterministic_baseline(
    checkpoint_path: str | Path,
) -> ERA5ResidualRegressor:
    """Load and freeze the deterministic baseline used by residual diffusion."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"deterministic baseline checkpoint does not exist: {path}"
        )
    model = ERA5ResidualRegressor.load_from_checkpoint(
        str(path), map_location="cpu"
    )
    model.requires_grad_(False)
    model.eval()
    return model
