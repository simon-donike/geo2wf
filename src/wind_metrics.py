"""Storm-centric wind-field metrics shared by deterministic and diffusion models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


EARTH_RADIUS_KM = 6371.0
RADIAL_METRIC_NAMES = (
    "eye_mae_ms",
    "eye_mean_wind_error_ms",
    "inner_core_mae_ms",
    "radial_profile_mae_ms",
    "rmw_error_km",
    "eye_to_eyewall_contrast_error_ms",
    "eye_center_displacement_km",
)

EYE_CENTER_SEARCH_RADIUS_KM = 100.0
EYE_CENTER_TARGET_RADIUS_KM = 50.0
EYE_CENTER_RING_INNER_KM = 20.0
EYE_CENTER_RING_OUTER_KM = 60.0
EYE_CENTER_MIN_COVERAGE = 0.8
EYE_CENTER_MIN_RING_WIND_MS = 17.0
EYE_CENTER_MIN_CONTRAST_MS = 5.0
EYE_CENTER_MIN_SMOOTHING_PIXELS = 8


def _masked_smooth_3x3(
    values: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a masked 3x3 mean and locations with at least 8/9 support."""
    kernel = values.new_ones((1, 1, 3, 3))
    valid_float = valid.to(dtype=values.dtype)[None, None]
    clean_values = torch.where(valid, values, torch.zeros_like(values))[None, None]
    support = F.conv2d(valid_float, kernel, padding=1)[0, 0]
    smoothed = F.conv2d(clean_values, kernel, padding=1)[0, 0]
    smoothed = smoothed / support.clamp_min(1.0)
    return smoothed, support >= EYE_CENTER_MIN_SMOOTHING_PIXELS


def _eye_center_displacement(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    radius_km: torch.Tensor,
    east_km: torch.Tensor,
    north_km: torch.Tensor,
) -> torch.Tensor | None:
    """Locate a well-observed target eye and measure prediction displacement."""
    search_region = radius_km <= EYE_CENTER_SEARCH_RADIUS_KM
    ring_region = (
        (radius_km >= EYE_CENTER_RING_INNER_KM)
        & (radius_km < EYE_CENTER_RING_OUTER_KM)
    )
    if not search_region.any() or not ring_region.any():
        return None
    search_coverage = valid[search_region].to(target.dtype).mean()
    ring_coverage = valid[ring_region].to(target.dtype).mean()
    if (
        search_coverage < EYE_CENTER_MIN_COVERAGE
        or ring_coverage < EYE_CENTER_MIN_COVERAGE
    ):
        return None

    ring_valid = valid & ring_region
    if not ring_valid.any():
        return None
    target_ring_mean = target[ring_valid].mean()
    if target_ring_mean < EYE_CENTER_MIN_RING_WIND_MS:
        return None

    target_smooth, smoothing_valid = _masked_smooth_3x3(target, valid)
    prediction_smooth, prediction_smoothing_valid = _masked_smooth_3x3(
        prediction, valid
    )
    shared_smoothing_valid = smoothing_valid & prediction_smoothing_valid
    target_candidates = shared_smoothing_valid & search_region
    prediction_candidates = shared_smoothing_valid & search_region
    if not target_candidates.any() or not prediction_candidates.any():
        return None

    target_flat_index = torch.argmin(
        target_smooth.masked_fill(~target_candidates, torch.inf)
    )
    if (
        radius_km.flatten()[target_flat_index]
        > EYE_CENTER_TARGET_RADIUS_KM
    ):
        return None
    target_minimum = target_smooth.flatten()[target_flat_index]
    if target_ring_mean - target_minimum < EYE_CENTER_MIN_CONTRAST_MS:
        return None
    prediction_flat_index = torch.argmin(
        prediction_smooth.masked_fill(~prediction_candidates, torch.inf)
    )

    target_east = east_km.flatten()[target_flat_index]
    target_north = north_km.flatten()[target_flat_index]
    prediction_east = east_km.flatten()[prediction_flat_index]
    prediction_north = north_km.flatten()[prediction_flat_index]
    return torch.sqrt(
        (prediction_east - target_east).square()
        + (prediction_north - target_north).square()
    )


def radial_wind_metric_statistics(
    prediction_ms: torch.Tensor,
    target_ms: torch.Tensor,
    target_mask: torch.Tensor,
    center: torch.Tensor,
    bounds: torch.Tensor,
    *,
    eye_radius_km: float = 25.0,
    inner_core_radius_km: float = 100.0,
    max_radius_km: float = 200.0,
    radial_bin_km: float = 10.0,
    min_bin_pixels: int = 4,
) -> torch.Tensor:
    """Return fixed ``[sum, count]`` rows for storm-centric metrics.

    ``center`` is ``[latitude, longitude]`` and ``bounds`` is
    ``[left, right, bottom, top]`` in degrees. Metrics use only observed target
    pixels, so partially sampled SAR swaths do not get silently filled with ERA5.
    Unavailable metrics (for example, an eye with no observed pixels) are omitted.
    """
    if prediction_ms.ndim != 4 or prediction_ms.shape[1] != 1:
        raise ValueError("prediction_ms must have shape [batch, 1, height, width]")
    if target_ms.shape != prediction_ms.shape:
        raise ValueError("target_ms must match prediction_ms")
    if target_mask.ndim == 3:
        target_mask = target_mask.unsqueeze(1)
    if target_mask.shape != prediction_ms.shape:
        raise ValueError("target_mask must match prediction_ms")
    if eye_radius_km <= 0 or inner_core_radius_km <= eye_radius_km:
        raise ValueError("radii must satisfy 0 < eye_radius_km < inner_core_radius_km")
    if max_radius_km <= inner_core_radius_km or radial_bin_km <= 0:
        raise ValueError("max_radius_km and radial_bin_km must be positive")
    if min_bin_pixels < 1:
        raise ValueError("min_bin_pixels must be positive")

    center = torch.as_tensor(center, device=prediction_ms.device)
    bounds = torch.as_tensor(bounds, device=prediction_ms.device)
    if center.ndim == 1:
        center = center.unsqueeze(0)
    if bounds.ndim == 1:
        bounds = bounds.unsqueeze(0)
    if center.shape != (prediction_ms.shape[0], 2):
        raise ValueError("center must have shape [batch, 2]")
    if bounds.shape != (prediction_ms.shape[0], 4):
        raise ValueError("bounds must have shape [batch, 4]")

    dtype = prediction_ms.dtype
    device = prediction_ms.device
    center = center.to(dtype=dtype)
    bounds = bounds.to(dtype=dtype)
    metric_values: dict[str, list[torch.Tensor]] = {
        name: [] for name in RADIAL_METRIC_NAMES
    }

    _, _, height, width = prediction_ms.shape
    row_fraction = (
        torch.arange(height, device=device, dtype=dtype) + 0.5
    ) / height
    column_fraction = (
        torch.arange(width, device=device, dtype=dtype) + 0.5
    ) / width
    bin_edges = torch.arange(
        0.0,
        max_radius_km + radial_bin_km,
        radial_bin_km,
        device=device,
        dtype=dtype,
    )

    for index in range(prediction_ms.shape[0]):
        sample_center = center[index]
        sample_bounds = bounds[index]
        if not torch.isfinite(sample_center).all() or not torch.isfinite(
            sample_bounds
        ).all():
            continue
        center_lat, center_lon = sample_center
        left, right, bottom, top = sample_bounds
        if right <= left or top <= bottom:
            continue

        latitudes = top - row_fraction * (top - bottom)
        longitudes = left + column_fraction * (right - left)
        latitude_grid, longitude_grid = torch.meshgrid(
            latitudes, longitudes, indexing="ij"
        )
        # Wrap the longitude delta so storms close to the dateline retain a
        # compact radial grid even when longitudes use a different convention.
        delta_lon = torch.remainder(longitude_grid - center_lon + 180.0, 360.0) - 180.0
        delta_lat = latitude_grid - center_lat
        north_km = torch.deg2rad(delta_lat) * EARTH_RADIUS_KM
        east_km = (
            torch.deg2rad(delta_lon)
            * EARTH_RADIUS_KM
            * torch.cos(torch.deg2rad(center_lat)).clamp_min(1e-6)
        )
        radius_km = torch.sqrt(north_km.square() + east_km.square())

        prediction = prediction_ms[index, 0]
        target = target_ms[index, 0]
        valid = target_mask[index, 0].bool()
        valid = valid & torch.isfinite(prediction) & torch.isfinite(target)

        eye_mask = valid & (radius_km < eye_radius_km)
        if eye_mask.any():
            eye_error = prediction[eye_mask] - target[eye_mask]
            metric_values["eye_mae_ms"].append(eye_error.abs().mean())
            metric_values["eye_mean_wind_error_ms"].append(
                (prediction[eye_mask].mean() - target[eye_mask].mean()).abs()
            )

        inner_core_mask = valid & (radius_km < inner_core_radius_km)
        if inner_core_mask.any():
            metric_values["inner_core_mae_ms"].append(
                (prediction[inner_core_mask] - target[inner_core_mask]).abs().mean()
            )

        eye_center_displacement = _eye_center_displacement(
            prediction,
            target,
            valid,
            radius_km,
            east_km,
            north_km,
        )
        if eye_center_displacement is not None:
            metric_values["eye_center_displacement_km"].append(
                eye_center_displacement
            )

        prediction_profile = []
        target_profile = []
        profile_radius = []
        for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
            bin_mask = valid & (radius_km >= lower) & (radius_km < upper)
            if int(bin_mask.sum().detach()) < min_bin_pixels:
                continue
            prediction_profile.append(prediction[bin_mask].mean())
            target_profile.append(target[bin_mask].mean())
            profile_radius.append((lower + upper) * 0.5)
        if len(prediction_profile) < 2:
            continue

        prediction_profile = torch.stack(prediction_profile)
        target_profile = torch.stack(target_profile)
        profile_radius = torch.stack(profile_radius)
        metric_values["radial_profile_mae_ms"].append(
            (prediction_profile - target_profile).abs().mean()
        )
        prediction_peak = prediction_profile.argmax()
        target_peak = target_profile.argmax()
        metric_values["rmw_error_km"].append(
            (profile_radius[prediction_peak] - profile_radius[target_peak]).abs()
        )
        if eye_mask.any():
            prediction_contrast = (
                prediction_profile[prediction_peak] - prediction[eye_mask].mean()
            )
            target_contrast = target_profile[target_peak] - target[eye_mask].mean()
            metric_values["eye_to_eyewall_contrast_error_ms"].append(
                (prediction_contrast - target_contrast).abs()
            )

    statistics = []
    for name in RADIAL_METRIC_NAMES:
        values = metric_values[name]
        if values:
            metric_sum = torch.stack(values).sum()
            metric_count = metric_sum.new_tensor(float(len(values)))
        else:
            metric_sum = prediction_ms.new_zeros(())
            metric_count = prediction_ms.new_zeros(())
        statistics.append(torch.stack([metric_sum, metric_count]))
    return torch.stack(statistics)


def radial_wind_metrics(
    prediction_ms: torch.Tensor,
    target_ms: torch.Tensor,
    target_mask: torch.Tensor,
    center: torch.Tensor,
    bounds: torch.Tensor,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Return available storm-centric metric means for direct evaluation."""
    statistics = radial_wind_metric_statistics(
        prediction_ms,
        target_ms,
        target_mask,
        center,
        bounds,
        **kwargs,
    )
    return {
        name: statistics[index, 0] / statistics[index, 1]
        for index, name in enumerate(RADIAL_METRIC_NAMES)
        if bool((statistics[index, 1] > 0).detach())
    }
