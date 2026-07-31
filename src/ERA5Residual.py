from __future__ import annotations

import math
from collections.abc import Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .reconstruction_logging import log_wandb_reconstruction
from .wind_metrics import (
    EARTH_RADIUS_KM,
    RADIAL_METRIC_NAMES,
    radial_wind_metric_statistics,
)


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


def continuous_high_wind_pixel_weights(
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    start_ms: float,
    full_ms: float,
    maximum_weight: float,
) -> torch.Tensor:
    """Return a smooth target-wind weighting ramp over valid pixels.

    Weights are one through ``start_ms`` and reach ``maximum_weight`` at
    ``full_ms``. A cubic smoothstep makes both transitions continuous without
    introducing discrete wind-category boundaries. Invalid pixels receive zero
    weight so the result can be passed directly to :func:`masked_huber_loss`.
    """
    if full_ms <= start_ms:
        raise ValueError("full_ms must be greater than start_ms")
    if maximum_weight < 1.0:
        raise ValueError("maximum_weight must be at least one")
    if target.shape != mask.shape:
        raise ValueError("mask must match target")
    mask = mask.to(device=target.device, dtype=target.dtype)
    ramp = ((target - start_ms) / (full_ms - start_ms)).clamp(0.0, 1.0)
    ramp = ramp.square() * (3.0 - 2.0 * ramp)
    return mask * (1.0 + (maximum_weight - 1.0) * ramp)


def storm_radius_grid_km(
    reference: torch.Tensor,
    center: torch.Tensor,
    bounds: torch.Tensor,
) -> torch.Tensor:
    """Return a storm-centered radius grid for each image in ``reference``.

    ``center`` uses ``[latitude, longitude]`` and ``bounds`` uses
    ``[left, right, bottom, top]`` degrees, matching the dataset metadata and
    the shared radial evaluation metrics.
    """
    if reference.ndim != 4 or reference.shape[1] != 1:
        raise ValueError("reference must have shape [batch, 1, height, width]")
    geometry_dtype = (
        reference.dtype if reference.dtype == torch.float64 else torch.float32
    )
    center = torch.as_tensor(center, device=reference.device, dtype=geometry_dtype)
    bounds = torch.as_tensor(bounds, device=reference.device, dtype=geometry_dtype)
    if center.ndim == 1:
        center = center.unsqueeze(0)
    if bounds.ndim == 1:
        bounds = bounds.unsqueeze(0)
    if center.shape != (reference.shape[0], 2):
        raise ValueError("center must have shape [batch, 2]")
    if bounds.shape != (reference.shape[0], 4):
        raise ValueError("bounds must have shape [batch, 4]")

    _, _, height, width = reference.shape
    row_fraction = (
        torch.arange(height, device=reference.device, dtype=geometry_dtype) + 0.5
    ) / height
    column_fraction = (
        torch.arange(width, device=reference.device, dtype=geometry_dtype) + 0.5
    ) / width
    center_lat = center[:, 0, None, None]
    center_lon = center[:, 1, None, None]
    left = bounds[:, 0, None, None]
    right = bounds[:, 1, None, None]
    bottom = bounds[:, 2, None, None]
    top = bounds[:, 3, None, None]
    latitudes = top - row_fraction[None, :, None] * (top - bottom)
    longitudes = left + column_fraction[None, None, :] * (right - left)
    delta_lat = latitudes - center_lat
    delta_lon = torch.remainder(longitudes - center_lon + 180.0, 360.0) - 180.0
    north_km = torch.deg2rad(delta_lat) * EARTH_RADIUS_KM
    east_km = (
        torch.deg2rad(delta_lon)
        * EARTH_RADIUS_KM
        * torch.cos(torch.deg2rad(center_lat)).clamp_min(1e-6)
    )
    radius = torch.sqrt(north_km.square() + east_km.square())
    geometry_is_valid = (
        torch.isfinite(center).all(dim=1)
        & torch.isfinite(bounds).all(dim=1)
        & (bounds[:, 1] > bounds[:, 0])
        & (bounds[:, 3] > bounds[:, 2])
    )
    return radius.masked_fill(~geometry_is_valid[:, None, None], torch.inf)


def _robust_top_fraction_peak_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    top_fraction: float,
    minimum_pixels: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have matching shapes")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    if minimum_pixels < 1:
        raise ValueError("minimum_pixels must be positive")

    peaks = []
    for sample_prediction, sample_target, sample_mask in zip(prediction, target, mask):
        valid = (
            sample_mask.bool()
            & torch.isfinite(sample_prediction)
            & torch.isfinite(sample_target)
        )
        sample_prediction = sample_prediction[valid]
        sample_target = sample_target[valid]
        if sample_target.numel() == 0:
            continue
        count = min(
            sample_target.numel(),
            max(minimum_pixels, math.ceil(sample_target.numel() * top_fraction)),
        )
        prediction_peak = sample_prediction.topk(count).values.mean()
        target_peak = sample_target.topk(count).values.mean()
        peaks.append((prediction_peak, target_peak))
    return peaks


def robust_top_fraction_peak_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    top_fraction: float,
    minimum_pixels: int,
    delta_ms: float,
) -> torch.Tensor:
    """Huber loss between per-image robust maxima.

    A robust maximum is the mean of the largest target fraction of pixels.
    Prediction and target ranks are selected independently, making this an
    intensity objective rather than a location objective.
    """
    if delta_ms <= 0:
        raise ValueError("delta_ms must be positive")
    peaks = _robust_top_fraction_peak_values(
        prediction,
        target,
        mask,
        top_fraction=top_fraction,
        minimum_pixels=minimum_pixels,
    )
    if not peaks:
        return prediction.nan_to_num().sum() * 0.0
    differences = torch.stack(
        [prediction_peak - target_peak for prediction_peak, target_peak in peaks]
    )
    absolute_differences = differences.abs()
    return torch.where(
        absolute_differences <= delta_ms,
        0.5 * differences.square(),
        delta_ms * (absolute_differences - 0.5 * delta_ms),
    ).mean()


def radial_profile_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    radius_km: torch.Tensor,
    *,
    max_radius_km: float,
    radial_bin_km: float,
    minimum_bin_pixels: int,
    delta_ms: float,
) -> torch.Tensor:
    """Compare storm-centered annular-mean wind profiles in physical units."""
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have matching shapes")
    if radius_km.shape != (prediction.shape[0], *prediction.shape[-2:]):
        raise ValueError("radius_km must have shape [batch, height, width]")
    if max_radius_km <= 0 or radial_bin_km <= 0:
        raise ValueError("radial profile radii must be positive")
    if minimum_bin_pixels < 1:
        raise ValueError("minimum_bin_pixels must be positive")
    if delta_ms <= 0:
        raise ValueError("delta_ms must be positive")

    batch_size = prediction.shape[0]
    bin_count = math.ceil(max_radius_km / radial_bin_km)
    accumulation_dtype = (
        torch.float32
        if prediction.dtype in {torch.float16, torch.bfloat16}
        else prediction.dtype
    )
    flat_prediction = prediction[:, 0].to(accumulation_dtype).flatten()
    flat_target = target[:, 0].to(accumulation_dtype).flatten()
    flat_radius = radius_km.flatten()
    flat_valid = (
        mask[:, 0].bool().flatten()
        & torch.isfinite(flat_prediction)
        & torch.isfinite(flat_target)
        & torch.isfinite(flat_radius)
        & (flat_radius >= 0)
        & (flat_radius < max_radius_km)
    )
    safe_radius = torch.nan_to_num(
        flat_radius,
        nan=max_radius_km,
        posinf=max_radius_km,
        neginf=0.0,
    )
    flat_bin = torch.floor(safe_radius / radial_bin_km).long()
    flat_bin = flat_bin.clamp(0, bin_count - 1)
    pixels_per_sample = prediction.shape[-2] * prediction.shape[-1]
    sample_index = torch.arange(batch_size, device=prediction.device)
    sample_index = sample_index.repeat_interleave(pixels_per_sample)
    flat_bin = flat_bin + sample_index * bin_count

    group_count = batch_size * bin_count
    counts = torch.zeros(
        group_count,
        device=prediction.device,
        dtype=accumulation_dtype,
    ).scatter_add(
        0,
        flat_bin[flat_valid],
        torch.ones_like(flat_prediction[flat_valid]),
    )
    prediction_sums = torch.zeros(
        group_count,
        device=prediction.device,
        dtype=accumulation_dtype,
    ).scatter_add(
        0,
        flat_bin[flat_valid],
        flat_prediction[flat_valid],
    )
    target_sums = torch.zeros(
        group_count,
        device=prediction.device,
        dtype=accumulation_dtype,
    ).scatter_add(
        0,
        flat_bin[flat_valid],
        flat_target[flat_valid],
    )
    valid_bins = counts >= minimum_bin_pixels
    if not valid_bins.any():
        return prediction.nan_to_num().sum() * 0.0
    difference = (prediction_sums[valid_bins] - target_sums[valid_bins]) / counts[
        valid_bins
    ]
    absolute_difference = difference.abs()
    return torch.where(
        absolute_difference <= delta_ms,
        0.5 * difference.square(),
        delta_ms * (absolute_difference - 0.5 * delta_ms),
    ).mean()


def soft_exceedance_area_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    thresholds_ms: Sequence[float],
    temperature_ms: float,
) -> torch.Tensor:
    """L1 loss between soft target and predicted exceedance areas.

    Applying the same sigmoid approximation to both fields makes the loss zero
    for an exact reconstruction and avoids biasing values that equal a category
    threshold upward merely because the hard target indicator includes them.
    """
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have matching shapes")
    if temperature_ms <= 0:
        raise ValueError("temperature_ms must be positive")
    thresholds = tuple(float(threshold) for threshold in thresholds_ms)
    if not thresholds:
        raise ValueError("thresholds_ms must not be empty")

    losses = []
    for sample_prediction, sample_target, sample_mask in zip(prediction, target, mask):
        valid = (
            sample_mask.bool()
            & torch.isfinite(sample_prediction)
            & torch.isfinite(sample_target)
        )
        sample_prediction = sample_prediction[valid]
        sample_target = sample_target[valid]
        if sample_target.numel() == 0:
            continue
        for threshold in thresholds:
            predicted_area = torch.sigmoid(
                (sample_prediction - threshold) / temperature_ms
            ).mean()
            target_area = torch.sigmoid(
                (sample_target - threshold) / temperature_ms
            ).mean()
            losses.append((predicted_area - target_area).abs())
    if not losses:
        return prediction.nan_to_num().sum() * 0.0
    return torch.stack(losses).mean()


class ERA5ResidualRegressor(pl.LightningModule):
    """Predict a deterministic physical-wind correction to the ERA5 field.

    The data set supplies an ERA5 wind speed transformed like the SAR target for
    network input, plus its physical m/s value for the residual connection. Loss
    and validation metrics are calculated in m/s over the joint valid mask.
    """

    checkpoint_monitor = "val/eye_structure_score"
    checkpoint_mode = "min"

    # count, |error|, error^2, signed error, Huber, ERA5 |error|,
    # ERA5 error^2, high-wind count, high-wind |error|, high-wind ERA5 |error|,
    # robust-peak sample count, robust-peak |error|, signed error, ERA5 |error|
    _STAT_COUNT = 14

    def __init__(
        self,
        condition_channels: int = 20,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        huber_delta_ms: float = 2.0,
        off_swath_anchor_weight: float = 0.05,
        high_wind_threshold_ms: float = 17.0,
        high_wind_weight_max: float = 1.0,
        high_wind_weight_start_ms: float = 25.0,
        high_wind_weight_full_ms: float = 43.0,
        peak_loss_weight: float = 0.0,
        peak_top_fraction: float = 0.005,
        peak_min_pixels: int = 4,
        peak_inner_core_radius_km: float | None = 100.0,
        radial_profile_loss_weight: float = 0.0,
        radial_profile_max_radius_km: float = 200.0,
        radial_profile_bin_km: float = 10.0,
        radial_profile_min_bin_pixels: int = 4,
        exceedance_area_loss_weight: float = 0.0,
        exceedance_area_thresholds_ms: Sequence[float] = (17.0, 33.0, 43.0),
        exceedance_area_temperature_ms: float = 2.0,
        peak_structure_radial_weight: float = 0.25,
        peak_structure_area_weight: float = 10.0,
        prediction_min_ms: float | None = 0.0,
        prediction_max_ms: float | None = None,
        psnr_data_range_ms: float = 79.8,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 10,
        lr_scheduler_monitor: str = "val/eye_structure_score",
        lr_scheduler_cooldown: int = 0,
        lr_scheduler_min_lr: float = 0.0,
        validation_reconstruction_batches: int = 1,
        log_reconstruction_images: bool = True,
    ) -> None:
        super().__init__()
        if condition_channels <= 0:
            raise ValueError("condition_channels must be positive")
        if huber_delta_ms <= 0:
            raise ValueError("huber_delta_ms must be positive")
        if psnr_data_range_ms <= 0:
            raise ValueError("psnr_data_range_ms must be positive")
        if off_swath_anchor_weight < 0:
            raise ValueError("off_swath_anchor_weight must be non-negative")
        if high_wind_weight_max < 1.0:
            raise ValueError("high_wind_weight_max must be at least one")
        if high_wind_weight_full_ms <= high_wind_weight_start_ms:
            raise ValueError(
                "high_wind_weight_full_ms must exceed high_wind_weight_start_ms"
            )
        if peak_loss_weight < 0:
            raise ValueError("peak_loss_weight must be non-negative")
        if not 0.0 < peak_top_fraction <= 1.0:
            raise ValueError("peak_top_fraction must be in (0, 1]")
        if peak_min_pixels < 1:
            raise ValueError("peak_min_pixels must be positive")
        if peak_inner_core_radius_km is not None and peak_inner_core_radius_km <= 0:
            raise ValueError("peak_inner_core_radius_km must be positive or None")
        if radial_profile_loss_weight < 0:
            raise ValueError("radial_profile_loss_weight must be non-negative")
        if radial_profile_max_radius_km <= 0 or radial_profile_bin_km <= 0:
            raise ValueError("radial profile radii must be positive")
        if radial_profile_min_bin_pixels < 1:
            raise ValueError("radial_profile_min_bin_pixels must be positive")
        if exceedance_area_loss_weight < 0:
            raise ValueError("exceedance_area_loss_weight must be non-negative")
        if exceedance_area_temperature_ms <= 0:
            raise ValueError("exceedance_area_temperature_ms must be positive")
        if peak_structure_radial_weight < 0:
            raise ValueError("peak_structure_radial_weight must be non-negative")
        if peak_structure_area_weight < 0:
            raise ValueError("peak_structure_area_weight must be non-negative")
        exceedance_area_thresholds_ms = tuple(
            float(threshold) for threshold in exceedance_area_thresholds_ms
        )
        if not exceedance_area_thresholds_ms:
            raise ValueError("exceedance_area_thresholds_ms must not be empty")
        if any(
            not math.isfinite(threshold) for threshold in exceedance_area_thresholds_ms
        ):
            raise ValueError("exceedance area thresholds must be finite")
        if any(
            upper <= lower
            for lower, upper in zip(
                exceedance_area_thresholds_ms[:-1],
                exceedance_area_thresholds_ms[1:],
            )
        ):
            raise ValueError("exceedance area thresholds must be strictly increasing")
        if validation_reconstruction_batches < 1:
            raise ValueError("validation_reconstruction_batches must be positive")
        self.save_hyperparameters()

        self.condition_channels = int(condition_channels)
        self.huber_delta_ms = float(huber_delta_ms)
        self.off_swath_anchor_weight = float(off_swath_anchor_weight)
        self.high_wind_threshold_ms = float(high_wind_threshold_ms)
        self.high_wind_weight_max = float(high_wind_weight_max)
        self.high_wind_weight_start_ms = float(high_wind_weight_start_ms)
        self.high_wind_weight_full_ms = float(high_wind_weight_full_ms)
        self.peak_loss_weight = float(peak_loss_weight)
        self.peak_top_fraction = float(peak_top_fraction)
        self.peak_min_pixels = int(peak_min_pixels)
        self.peak_inner_core_radius_km = peak_inner_core_radius_km
        self.radial_profile_loss_weight = float(radial_profile_loss_weight)
        self.radial_profile_max_radius_km = float(radial_profile_max_radius_km)
        self.radial_profile_bin_km = float(radial_profile_bin_km)
        self.radial_profile_min_bin_pixels = int(radial_profile_min_bin_pixels)
        self.exceedance_area_loss_weight = float(exceedance_area_loss_weight)
        self.exceedance_area_thresholds_ms = exceedance_area_thresholds_ms
        self.exceedance_area_temperature_ms = float(exceedance_area_temperature_ms)
        self.peak_structure_radial_weight = float(peak_structure_radial_weight)
        self.peak_structure_area_weight = float(peak_structure_area_weight)
        self.prediction_min_ms = prediction_min_ms
        self.prediction_max_ms = prediction_max_ms
        self.psnr_data_range_ms = float(psnr_data_range_ms)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.lr_scheduler_factor = float(lr_scheduler_factor)
        self.lr_scheduler_patience = int(lr_scheduler_patience)
        self.lr_scheduler_monitor = str(lr_scheduler_monitor)
        self.lr_scheduler_cooldown = int(lr_scheduler_cooldown)
        self.lr_scheduler_min_lr = float(lr_scheduler_min_lr)
        self.validation_reconstruction_batches = int(validation_reconstruction_batches)
        self.log_reconstruction_images = bool(log_reconstruction_images)

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
        exceedance_statistic_shape = (
            len(self.exceedance_area_thresholds_ms),
            4,
        )
        self.register_buffer(
            "_validation_exceedance_statistics",
            torch.zeros(exceedance_statistic_shape, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_test_exceedance_statistics",
            torch.zeros(exceedance_statistic_shape, dtype=torch.float64),
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
            raise ValueError(
                "condition must have shape [batch, channel, height, width]"
            )
        if condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"expected {self.condition_channels} condition channels, "
                f"got {condition.shape[1]}"
            )
        condition_mask = self._single_channel(
            condition_mask, "condition_mask", collapse_mask=True
        )
        era5_wind_speed = self._single_channel(era5_wind_speed, "era5_wind_speed")
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
        unweighted_reconstruction_loss = masked_huber_loss(
            prediction,
            target,
            valid_mask,
            delta=self.huber_delta_ms,
        )
        weighted_mask = continuous_high_wind_pixel_weights(
            target,
            valid_mask,
            start_ms=self.high_wind_weight_start_ms,
            full_ms=self.high_wind_weight_full_ms,
            maximum_weight=self.high_wind_weight_max,
        )
        reconstruction_loss = masked_huber_loss(
            prediction,
            target,
            weighted_mask,
            delta=self.huber_delta_ms,
        )
        anchor_mask = self._off_swath_mask(batch, prediction)
        anchor_loss = masked_huber_loss(
            prediction - era5,
            torch.zeros_like(prediction),
            anchor_mask,
            delta=self.huber_delta_ms,
        )

        structural_prediction = self._bound_prediction(prediction)
        zero_loss = structural_prediction.nan_to_num().sum() * 0.0
        radius_km = None
        peak_loss = zero_loss
        if self.peak_loss_weight > 0:
            peak_mask = valid_mask
            if self.peak_inner_core_radius_km is not None:
                radius_km = self._training_radius_grid(batch, structural_prediction)
                peak_mask = peak_mask * (
                    radius_km[:, None] <= self.peak_inner_core_radius_km
                ).to(peak_mask.dtype)
            peak_loss = robust_top_fraction_peak_loss(
                structural_prediction,
                target,
                peak_mask,
                top_fraction=self.peak_top_fraction,
                minimum_pixels=self.peak_min_pixels,
                delta_ms=self.huber_delta_ms,
            )

        radial_profile_loss = zero_loss
        if self.radial_profile_loss_weight > 0:
            if radius_km is None:
                radius_km = self._training_radius_grid(batch, structural_prediction)
            radial_profile_loss = radial_profile_huber_loss(
                structural_prediction,
                target,
                valid_mask,
                radius_km,
                max_radius_km=self.radial_profile_max_radius_km,
                radial_bin_km=self.radial_profile_bin_km,
                minimum_bin_pixels=self.radial_profile_min_bin_pixels,
                delta_ms=self.huber_delta_ms,
            )

        exceedance_area_loss = zero_loss
        if self.exceedance_area_loss_weight > 0:
            exceedance_area_loss = soft_exceedance_area_loss(
                structural_prediction,
                target,
                valid_mask,
                thresholds_ms=self.exceedance_area_thresholds_ms,
                temperature_ms=self.exceedance_area_temperature_ms,
            )
        loss = (
            reconstruction_loss
            + self.off_swath_anchor_weight * anchor_loss
            + self.peak_loss_weight * peak_loss
            + self.radial_profile_loss_weight * radial_profile_loss
            + self.exceedance_area_loss_weight * exceedance_area_loss
        )
        valid_count = valid_mask.sum().clamp_min(1)
        mean_pixel_weight = weighted_mask.sum() / valid_count
        mae = ((structural_prediction - target).abs() * valid_mask).sum()
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
            "train/unweighted_reconstruction_loss",
            unweighted_reconstruction_loss,
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
        for name, value in {
            "high_wind_pixel_weight_mean": mean_pixel_weight,
            "robust_peak_loss": peak_loss,
            "radial_profile_loss": radial_profile_loss,
            "exceedance_area_loss": exceedance_area_loss,
        }.items():
            self.log(
                f"train/{name}",
                value,
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
            if self.log_reconstruction_images and batch_idx == 0:
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
        bounded_era5 = self._bound_prediction(era5)
        self._accumulate_statistics(
            self._validation_statistics,
            bounded_prediction,
            target,
            valid_mask,
            bounded_era5,
            raw_prediction=prediction,
            peak_mask=self._peak_metric_mask(batch, valid_mask, bounded_prediction),
        )
        self._accumulate_exceedance_statistics(
            self._validation_exceedance_statistics,
            bounded_prediction,
            target,
            valid_mask,
            bounded_era5,
        )
        self._accumulate_radial_statistics(
            self._validation_radial_statistics,
            self._bound_prediction(prediction),
            target,
            valid_mask,
            batch,
        )
        if (
            self.log_reconstruction_images
            and batch_idx < self.validation_reconstruction_batches
        ):
            self._log_reconstruction(batch, bounded_prediction)
        return None

    def on_validation_epoch_start(self) -> None:
        self._validation_statistics.zero_()
        self._validation_radial_statistics.zero_()
        self._validation_exceedance_statistics.zero_()

    def on_validation_epoch_end(self) -> None:
        self._log_statistics("val", self._validation_statistics)
        self._log_radial_statistics("val", self._validation_radial_statistics)
        self._log_exceedance_statistics("val", self._validation_exceedance_statistics)
        self._log_peak_structure_score(
            "val",
            self._validation_statistics,
            self._validation_radial_statistics,
            self._validation_exceedance_statistics,
        )

    @torch.no_grad()
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        del batch_idx
        prediction, target, valid_mask, era5 = self._batch_outputs(batch)
        bounded_prediction = self._bound_prediction(prediction)
        bounded_era5 = self._bound_prediction(era5)
        self._accumulate_statistics(
            self._test_statistics,
            bounded_prediction,
            target,
            valid_mask,
            bounded_era5,
            raw_prediction=prediction,
            peak_mask=self._peak_metric_mask(batch, valid_mask, bounded_prediction),
        )
        self._accumulate_exceedance_statistics(
            self._test_exceedance_statistics,
            bounded_prediction,
            target,
            valid_mask,
            bounded_era5,
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
        self._test_exceedance_statistics.zero_()

    def on_test_epoch_end(self) -> None:
        self._log_statistics("test", self._test_statistics)
        self._log_radial_statistics("test", self._test_radial_statistics)
        self._log_exceedance_statistics("test", self._test_exceedance_statistics)
        self._log_peak_structure_score(
            "test",
            self._test_statistics,
            self._test_radial_statistics,
            self._test_exceedance_statistics,
        )

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
            physical_wind_output=True,
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

    def _batch_outputs(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._require_supervised_batch_keys(batch)
        residual = self.predict_residual_ms(batch)
        target = self._single_channel(batch["target_physical"], "target_physical").to(
            device=residual.device, dtype=residual.dtype
        )
        era5 = self._single_channel(
            batch["era5_wind_speed_physical"],
            "era5_wind_speed_physical",
        ).to(device=residual.device, dtype=residual.dtype)
        prediction = era5 + residual
        valid_mask = self._valid_mask(batch, prediction)
        return prediction, target, valid_mask, era5

    @staticmethod
    def _training_radius_grid(
        batch: dict[str, torch.Tensor], reference: torch.Tensor
    ) -> torch.Tensor:
        missing = sorted({"center", "target_bounds"}.difference(batch))
        if missing:
            raise KeyError(
                "storm-centered Stage 1 losses require: " + ", ".join(missing)
            )
        metadata_radius = storm_radius_grid_km(
            reference,
            batch["center"],
            batch["target_bounds"],
        )
        # The dataset appends distance-to-center immediately before its three
        # solar-time channels. Unlike center/bounds metadata, this raster is
        # transformed by random augmentation, so prefer it during training.
        condition = batch.get("condition")
        if condition is None or condition.ndim != 4 or condition.shape[1] < 4:
            return metadata_radius
        normalized_radius = condition[:, -4].to(
            device=reference.device, dtype=metadata_radius.dtype
        )
        if (
            not torch.isfinite(normalized_radius).all()
            or (normalized_radius < -1e-4).any()
            or (normalized_radius > 1.0001).any()
        ):
            return metadata_radius
        maximum_radius = metadata_radius.flatten(1).amax(dim=1)
        return normalized_radius.clamp(0.0, 1.0) * maximum_radius[:, None, None]

    def _peak_metric_mask(
        self,
        batch: dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Restrict robust-peak evaluation to the inner core when available."""
        if self.peak_inner_core_radius_km is None:
            return valid_mask
        if not {"center", "target_bounds"}.issubset(batch):
            return valid_mask
        radius_km = storm_radius_grid_km(
            reference,
            batch["center"],
            batch["target_bounds"],
        )
        return valid_mask * (radius_km[:, None] <= self.peak_inner_core_radius_km).to(
            valid_mask.dtype
        )

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
        return (target_mask.bool() & condition_mask.bool() & era5_mask.bool()).to(
            device=reference.device, dtype=reference.dtype
        )

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
        return (~target_mask.bool() & condition_mask.bool() & era5_mask.bool()).to(
            device=reference.device, dtype=reference.dtype
        )

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
        peak_mask: torch.Tensor,
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
            self.huber_delta_ms * (raw_absolute_error - 0.5 * self.huber_delta_ms),
        )
        high_wind_mask = mask * (target >= self.high_wind_threshold_ms).to(mask.dtype)
        prediction_peaks = _robust_top_fraction_peak_values(
            prediction,
            target,
            peak_mask,
            top_fraction=self.peak_top_fraction,
            minimum_pixels=self.peak_min_pixels,
        )
        era5_peaks = _robust_top_fraction_peak_values(
            era5,
            target,
            peak_mask,
            top_fraction=self.peak_top_fraction,
            minimum_pixels=self.peak_min_pixels,
        )
        if prediction_peaks:
            peak_errors = torch.stack(
                [
                    prediction_peak - target_peak
                    for prediction_peak, target_peak in prediction_peaks
                ]
            )
            era5_peak_errors = torch.stack(
                [era5_peak - target_peak for era5_peak, target_peak in era5_peaks]
            )
            peak_additions = (
                peak_errors.new_tensor(float(len(prediction_peaks))),
                peak_errors.abs().sum(),
                peak_errors.sum(),
                era5_peak_errors.abs().sum(),
            )
        else:
            peak_additions = tuple(prediction.new_zeros(()) for _ in range(4))
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
                *peak_additions,
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
                mae.new_tensor(self.psnr_data_range_ms) / mse.clamp_min(1e-12).sqrt()
            ),
            f"{prefix}/era5_mae_ms": era5_mae,
            f"{prefix}/era5_rmse_ms": era5_mse.sqrt(),
            f"{prefix}/mae_skill_vs_era5": 1.0 - mae / era5_mae.clamp_min(1e-12),
        }
        high_count = statistics[7]
        if high_count > 0:
            metrics[f"{prefix}/high_wind_mae_ms"] = statistics[8] / high_count
            metrics[f"{prefix}/high_wind_era5_mae_ms"] = statistics[9] / high_count
        peak_count = statistics[10]
        if peak_count > 0:
            metrics[f"{prefix}/robust_peak_mae_ms"] = statistics[11] / peak_count
            metrics[f"{prefix}/robust_peak_bias_ms"] = statistics[12] / peak_count
            metrics[f"{prefix}/era5_robust_peak_mae_ms"] = statistics[13] / peak_count
        for name, value in metrics.items():
            self.log(
                name,
                value.to(dtype=torch.float32),
                on_step=False,
                on_epoch=True,
                prog_bar=name
                in {
                    f"{prefix}/mae_ms",
                    f"{prefix}/era5_mae_ms",
                },
                logger=True,
                sync_dist=False,
            )

    def _accumulate_exceedance_statistics(
        self,
        statistics: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        era5: torch.Tensor,
    ) -> None:
        """Accumulate per-image hard exceedance-area errors by wind threshold."""
        additions = prediction.new_zeros(statistics.shape)
        for threshold_index, threshold in enumerate(self.exceedance_area_thresholds_ms):
            for sample_prediction, sample_target, sample_mask, sample_era5 in zip(
                prediction, target, valid_mask, era5
            ):
                valid = (
                    sample_mask.bool()
                    & torch.isfinite(sample_prediction)
                    & torch.isfinite(sample_target)
                    & torch.isfinite(sample_era5)
                )
                if not valid.any():
                    continue
                target_area = (
                    (sample_target[valid] >= threshold).to(target.dtype).mean()
                )
                prediction_area = (
                    (sample_prediction[valid] >= threshold).to(prediction.dtype).mean()
                )
                era5_area = (sample_era5[valid] >= threshold).to(era5.dtype).mean()
                error = prediction_area - target_area
                additions[threshold_index, 0] += 1.0
                additions[threshold_index, 1] += error.abs()
                additions[threshold_index, 2] += error
                additions[threshold_index, 3] += (era5_area - target_area).abs()
        statistics.add_(additions.to(statistics))

    def _log_exceedance_statistics(self, prefix: str, statistics: torch.Tensor) -> None:
        statistics = self._distributed_sum(statistics)
        for threshold_index, threshold in enumerate(self.exceedance_area_thresholds_ms):
            count = statistics[threshold_index, 0]
            if count <= 0:
                continue
            label = f"{threshold:g}".replace("-", "m").replace(".", "p")
            metrics = {
                f"{prefix}/r{label}_area_mae_fraction": (
                    statistics[threshold_index, 1] / count
                ),
                f"{prefix}/r{label}_area_bias_fraction": (
                    statistics[threshold_index, 2] / count
                ),
                f"{prefix}/era5_r{label}_area_mae_fraction": (
                    statistics[threshold_index, 3] / count
                ),
            }
            for name, value in metrics.items():
                self.log(
                    name,
                    value.to(dtype=torch.float32),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )

    def _log_peak_structure_score(
        self,
        prefix: str,
        statistics: torch.Tensor,
        radial_statistics: torch.Tensor,
        exceedance_statistics: torch.Tensor,
    ) -> None:
        """Log one intensity-and-structure score for checkpoint selection."""
        statistics = self._distributed_sum(statistics)
        radial_statistics = self._distributed_sum(radial_statistics)
        exceedance_statistics = self._distributed_sum(exceedance_statistics)
        peak_count = statistics[10]
        radial_index = RADIAL_METRIC_NAMES.index("radial_profile_mae_ms")
        radial_count = radial_statistics[radial_index, 1]
        area_counts = exceedance_statistics[:, 0]
        available_areas = area_counts > 0
        if peak_count <= 0 or radial_count <= 0 or not available_areas.any():
            return
        peak_mae = statistics[11] / peak_count
        radial_mae = radial_statistics[radial_index, 0] / radial_count
        area_mae = (
            exceedance_statistics[available_areas, 1] / area_counts[available_areas]
        ).mean()
        score = (
            peak_mae
            + self.peak_structure_radial_weight * radial_mae
            + self.peak_structure_area_weight * area_mae
        )
        self.log(
            f"{prefix}/peak_structure_score",
            score.to(dtype=torch.float32),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
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

    def _log_radial_statistics(self, prefix: str, statistics: torch.Tensor) -> None:
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
            raise KeyError("ERA5 residual batches require: " + ", ".join(missing))

    @classmethod
    def _require_supervised_batch_keys(cls, batch: dict[str, torch.Tensor]) -> None:
        cls._require_prediction_keys(batch)
        missing = sorted({"target_mask", "target_physical"}.difference(batch))
        if missing:
            raise KeyError(
                "ERA5 residual training batches require: " + ", ".join(missing)
            )
