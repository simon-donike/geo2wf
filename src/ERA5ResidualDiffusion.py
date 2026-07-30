from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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
        gradient_loss_weight: float = 0.0,
        spectrum_loss_weight: float = 0.0,
        low_frequency_loss_weight: float = 0.0,
        smoothness_loss_weight: float = 0.0,
        auxiliary_max_timestep_fraction: float = 0.5,
        high_wind_threshold_ms: float = 17.0,
        high_wind_loss_weight: float = 1.0,
        inner_core_radius_km: float = 100.0,
        inner_core_loss_weight: float = 1.0,
        high_gradient_threshold_ms: float = 2.0,
        high_gradient_loss_weight: float = 1.0,
        low_frequency_kernel_size: int = 9,
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
        for name, value in {
            "gradient_loss_weight": gradient_loss_weight,
            "spectrum_loss_weight": spectrum_loss_weight,
            "low_frequency_loss_weight": low_frequency_loss_weight,
            "smoothness_loss_weight": smoothness_loss_weight,
        }.items():
            if float(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 < float(auxiliary_max_timestep_fraction) <= 1.0:
            raise ValueError("auxiliary_max_timestep_fraction must be in (0, 1]")
        if (
            high_wind_loss_weight < 1.0
            or inner_core_loss_weight < 1.0
            or high_gradient_loss_weight < 1.0
        ):
            raise ValueError("structural loss weights must be at least 1")
        if inner_core_radius_km <= 0:
            raise ValueError("inner_core_radius_km must be positive")
        if high_gradient_threshold_ms < 0:
            raise ValueError("high_gradient_threshold_ms must be non-negative")
        if low_frequency_kernel_size < 1 or low_frequency_kernel_size % 2 == 0:
            raise ValueError("low_frequency_kernel_size must be a positive odd integer")
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
        self.gradient_loss_weight = float(gradient_loss_weight)
        self.spectrum_loss_weight = float(spectrum_loss_weight)
        self.low_frequency_loss_weight = float(low_frequency_loss_weight)
        self.smoothness_loss_weight = float(smoothness_loss_weight)
        self.auxiliary_max_timestep_fraction = float(
            auxiliary_max_timestep_fraction
        )
        self.high_wind_threshold_ms = float(high_wind_threshold_ms)
        self.high_wind_loss_weight = float(high_wind_loss_weight)
        self.inner_core_radius_km = float(inner_core_radius_km)
        self.inner_core_loss_weight = float(inner_core_loss_weight)
        self.high_gradient_threshold_ms = float(high_gradient_threshold_ms)
        self.high_gradient_loss_weight = float(high_gradient_loss_weight)
        self.low_frequency_kernel_size = int(low_frequency_kernel_size)
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
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        initial_noise=None,
        guidance_scale=None,
    ):
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
        selected_guidance = (
            self.guidance_scale if guidance_scale is None else float(guidance_scale)
        )
        unconditional_condition = (
            torch.zeros_like(condition) if selected_guidance != 1.0 else None
        )
        residual_sample = process(
            condition,
            initial_noise=initial_noise,
            guidance_scale=selected_guidance,
            unconditional_condition=unconditional_condition,
        )
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

    def _diffusion_loss(
        self, model_target, condition, loss_weight, batch, *, stage
    ):
        """Optimize calibrated residual noise plus weak structural objectives."""
        del loss_weight
        prepared = self._prepare_batch_context(batch)
        if stage == "train":
            condition = self._condition_for_training(condition)
        details = self.model.diffusion_terms(model_target, condition)
        timestep_weight = self.model.min_snr_weight(
            details["snr"], self.min_snr_gamma
        )

        target_physical = prepared["target_physical"].to(
            device=model_target.device, dtype=model_target.dtype
        )
        baseline_physical = prepared[_BASELINE_PHYSICAL].to(
            device=model_target.device, dtype=model_target.dtype
        )
        baseline_valid = prepared[_BASELINE_MASK].to(
            device=model_target.device
        ).bool()
        observed = self._prepare_target_mask(
            prepared["target_mask"], target_physical
        ).bool()
        jointly_observed = observed & baseline_valid
        anchor_only = ~observed & baseline_valid
        target_residual_ms = target_physical - baseline_physical

        target_gradient, _ = self._gradient_magnitude(
            target_residual_ms, jointly_observed.to(model_target.dtype)
        )
        structural_weight = torch.ones_like(model_target)
        structural_weight = torch.where(
            target_physical >= self.high_wind_threshold_ms,
            structural_weight * self.high_wind_loss_weight,
            structural_weight,
        )
        inner_core = self._inner_core_mask(prepared, model_target)
        structural_weight = torch.where(
            inner_core,
            structural_weight * self.inner_core_loss_weight,
            structural_weight,
        )
        structural_weight = torch.where(
            target_gradient >= self.high_gradient_threshold_ms,
            structural_weight * self.high_gradient_loss_weight,
            structural_weight,
        )
        observation_weight = (
            jointly_observed.to(model_target.dtype) * structural_weight
        )
        observation_loss = self.model.weighted_loss(
            details["squared_error"], observation_weight, timestep_weight
        )
        anchor_loss = self.model.weighted_loss(
            details["squared_error"],
            anchor_only.to(model_target.dtype),
            timestep_weight,
        )

        auxiliary_enabled = any(
            weight > 0
            for weight in (
                self.gradient_loss_weight,
                self.spectrum_loss_weight,
                self.low_frequency_loss_weight,
                self.smoothness_loss_weight,
            )
        )
        if auxiliary_enabled:
            clean_residual_ms = self.decode_residual(
                details["clean_prediction"].clamp(-1.0, 1.0)
            )
            maximum_auxiliary_timestep = int(
                (self.model.num_timesteps - 1)
                * self.auxiliary_max_timestep_fraction
            )
            auxiliary_gate = (
                details["t"] <= maximum_auxiliary_timestep
            ).to(model_target.dtype).view(-1, 1, 1, 1)
            auxiliary_mask = (
                jointly_observed.to(model_target.dtype) * auxiliary_gate
            )
            predicted_gradient, gradient_mask = self._gradient_magnitude(
                clean_residual_ms, auxiliary_mask
            )
            auxiliary_target_gradient, _ = self._gradient_magnitude(
                target_residual_ms, auxiliary_mask
            )
            gradient_loss = self.model.weighted_loss(
                (predicted_gradient - auxiliary_target_gradient).abs(),
                gradient_mask,
            )
            # Weak total variation on the physical residual suppresses
            # pixel-scale ringing while leaving the broad baseline untouched.
            smoothness_loss = self.model.weighted_loss(
                predicted_gradient, gradient_mask
            )
            spectrum_loss = self._masked_log_spectrum_loss(
                clean_residual_ms, target_residual_ms, auxiliary_mask
            )
            low_frequency_loss = self._masked_low_frequency_loss(
                clean_residual_ms, target_residual_ms, auxiliary_mask
            )
        else:
            zero = details["squared_error"].sum() * 0.0
            gradient_loss = zero
            spectrum_loss = zero
            low_frequency_loss = zero
            smoothness_loss = zero
        loss = (
            observation_loss
            + self.unobserved_loss_weight * anchor_loss
            + self.gradient_loss_weight * gradient_loss
            + self.spectrum_loss_weight * spectrum_loss
            + self.low_frequency_loss_weight * low_frequency_loss
            + self.smoothness_loss_weight * smoothness_loss
        )
        metrics = {
            "diffusion_observed_loss": observation_loss,
            "diffusion_anchor_loss": anchor_loss,
            "gradient_loss": gradient_loss,
            "spectrum_loss": spectrum_loss,
            "low_frequency_loss": low_frequency_loss,
            "smoothness_loss": smoothness_loss,
        }
        batch_size = int(model_target.shape[0])
        for name, value in metrics.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        return loss

    def _inner_core_mask(self, batch, reference):
        """Return pixels within the configured storm-centered physical radius."""
        if not {"center", "target_bounds"}.issubset(batch):
            return torch.zeros_like(reference, dtype=torch.bool)
        center = torch.as_tensor(
            batch["center"], device=reference.device, dtype=reference.dtype
        )
        bounds = torch.as_tensor(
            batch["target_bounds"],
            device=reference.device,
            dtype=reference.dtype,
        )
        if center.ndim == 1:
            center = center.unsqueeze(0)
        if bounds.ndim == 1:
            bounds = bounds.unsqueeze(0)
        if center.shape != (reference.shape[0], 2) or bounds.shape != (
            reference.shape[0],
            4,
        ):
            return torch.zeros_like(reference, dtype=torch.bool)
        height, width = reference.shape[-2:]
        x_fraction = (
            torch.arange(width, device=reference.device, dtype=reference.dtype)
            + 0.5
        ) / width
        y_fraction = (
            torch.arange(height, device=reference.device, dtype=reference.dtype)
            + 0.5
        ) / height
        left, right, bottom, top = bounds.unbind(dim=1)
        longitude = left[:, None, None] + x_fraction[None, None, :] * (
            right - left
        )[:, None, None]
        latitude = top[:, None, None] - y_fraction[None, :, None] * (
            top - bottom
        )[:, None, None]
        center_latitude = center[:, 0, None, None]
        center_longitude = center[:, 1, None, None]
        delta_longitude = torch.remainder(
            longitude - center_longitude + 180.0, 360.0
        ) - 180.0
        north_km = (latitude - center_latitude) * 111.32
        east_km = (
            delta_longitude
            * 111.32
            * torch.cos(torch.deg2rad(center_latitude)).clamp_min(1e-6)
        )
        radius_km = torch.sqrt(north_km.square() + east_km.square())
        valid_center = torch.isfinite(center).all(dim=1).view(-1, 1, 1)
        valid_bounds = torch.isfinite(bounds).all(dim=1).view(-1, 1, 1)
        result = (radius_km <= self.inner_core_radius_km) & valid_center & valid_bounds
        return result.unsqueeze(1).expand_as(reference)

    def _masked_low_frequency_loss(self, prediction, target, mask):
        kernel = self.low_frequency_kernel_size
        padding = kernel // 2
        pooled_weight = F.avg_pool2d(
            mask, kernel, stride=1, padding=padding
        )
        prediction_mean = F.avg_pool2d(
            prediction * mask, kernel, stride=1, padding=padding
        ) / pooled_weight.clamp_min(1e-6)
        target_mean = F.avg_pool2d(
            target * mask, kernel, stride=1, padding=padding
        ) / pooled_weight.clamp_min(1e-6)
        valid = (pooled_weight >= 0.5).to(prediction.dtype)
        return self.model.weighted_loss(
            (prediction_mean - target_mean).abs(), valid
        )

    @staticmethod
    def _masked_log_spectrum_loss(prediction, target, mask):
        count = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        prediction_mean = (prediction * mask).sum(
            dim=(-2, -1), keepdim=True
        ) / count
        target_mean = (target * mask).sum(
            dim=(-2, -1), keepdim=True
        ) / count
        prediction_spectrum = torch.fft.rfft2(
            (prediction - prediction_mean) * mask, norm="ortho"
        ).abs()
        target_spectrum = torch.fft.rfft2(
            (target - target_mean) * mask, norm="ortho"
        ).abs()
        per_sample = (
            torch.log1p(prediction_spectrum)
            - torch.log1p(target_spectrum)
        ).abs().mean(dim=(-3, -2, -1))
        active = (mask.sum(dim=(-3, -2, -1)) > 0).to(per_sample.dtype)
        return (per_sample * active).sum() / active.sum().clamp_min(1.0)

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
