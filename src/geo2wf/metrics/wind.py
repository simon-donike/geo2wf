"""Storm-centric wind-field metrics shared by deterministic and diffusion models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
IBTRACS_RADIUS_NAMES = ("rmw", "r34", "r50", "r64")
IBTRACS_WIND_RADIUS_THRESHOLDS_MS = {
    "r34": 34.0 * 0.514444,
    "r50": 50.0 * 0.514444,
    "r64": 64.0 * 0.514444,
}


def ibtracs_radius_targets(
    batch: Mapping[str, Any], reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Stack optional scalar IBTrACS radius companions in metric order."""
    values = []
    validity = []
    for name in IBTRACS_RADIUS_NAMES:
        source_name = name if name == "rmw" else f"{name}_equivalent"
        value_key = f"ibtracs_{source_name}_km"
        valid_key = f"{value_key}_valid"
        if value_key not in batch or valid_key not in batch:
            return None
        value = torch.as_tensor(batch[value_key], device=reference.device).reshape(-1)
        valid = torch.as_tensor(
            batch[valid_key], device=reference.device, dtype=torch.bool
        ).reshape(-1)
        if value.shape != (reference.shape[0],) or valid.shape != value.shape:
            raise ValueError(f"{value_key} must have one value per generated image")
        values.append(value.to(dtype=reference.dtype))
        validity.append(valid)
    return torch.stack(values, dim=1), torch.stack(validity, dim=1)


EYE_CENTER_SEARCH_RADIUS_KM = 100.0
EYE_CENTER_TARGET_RADIUS_KM = 50.0
EYE_CENTER_RING_INNER_KM = 20.0
EYE_CENTER_RING_OUTER_KM = 60.0
EYE_CENTER_MIN_COVERAGE = 0.8
EYE_CENTER_MIN_RING_WIND_MS = 17.0
EYE_CENTER_MIN_CONTRAST_MS = 5.0
EYE_CENTER_MIN_SMOOTHING_PIXELS = 8


def ibtracs_radius_metric_statistics(
    prediction_ms: torch.Tensor,
    prediction_mask: torch.Tensor,
    center: torch.Tensor,
    bounds: torch.Tensor,
    target_radii_km: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    radial_bin_km: float = 10.0,
    min_bin_pixels: int = 4,
) -> torch.Tensor:
    """Compare radii derived from generated fields with scalar IBTrACS radii.

    The returned rows follow :data:`IBTRACS_RADIUS_NAMES`; columns contain
    predicted sum, target sum, absolute-error sum, squared-error sum,
    signed-error sum, and count.
    Wind radii are equivalent-circle radii calculated from the 34-, 50-, or
    64-knot exceedance area. RMW is the peak annular-mean bin. Only the complete
    circular domain supported by the generated image is used.
    """
    if prediction_ms.ndim != 4 or prediction_ms.shape[1] != 1:
        raise ValueError("prediction_ms must have shape [batch, 1, height, width]")
    if prediction_mask.ndim == 3:
        prediction_mask = prediction_mask.unsqueeze(1)
    if prediction_mask.shape != prediction_ms.shape:
        raise ValueError("prediction_mask must match prediction_ms")
    if radial_bin_km <= 0.0 or min_bin_pixels < 1:
        raise ValueError("radial_bin_km and min_bin_pixels must be positive")

    batch_size = prediction_ms.shape[0]
    device = prediction_ms.device
    geometry_dtype = (
        prediction_ms.dtype if prediction_ms.dtype == torch.float64 else torch.float32
    )
    center = torch.as_tensor(center, device=device, dtype=geometry_dtype)
    bounds = torch.as_tensor(bounds, device=device, dtype=geometry_dtype)
    target_radii_km = torch.as_tensor(
        target_radii_km, device=device, dtype=geometry_dtype
    )
    target_valid = torch.as_tensor(target_valid, device=device, dtype=torch.bool)
    if center.ndim == 1:
        center = center.unsqueeze(0)
    if bounds.ndim == 1:
        bounds = bounds.unsqueeze(0)
    expected_radii_shape = (batch_size, len(IBTRACS_RADIUS_NAMES))
    if center.shape != (batch_size, 2) or bounds.shape != (batch_size, 4):
        raise ValueError("center and bounds have incompatible batch shapes")
    if target_radii_km.shape != expected_radii_shape:
        raise ValueError(f"target_radii_km must have shape {expected_radii_shape}")
    if target_valid.shape != expected_radii_shape:
        raise ValueError(f"target_valid must have shape {expected_radii_shape}")

    statistics = prediction_ms.new_zeros((len(IBTRACS_RADIUS_NAMES), 6))
    _, _, height, width = prediction_ms.shape
    row_fraction = (
        torch.arange(height, device=device, dtype=geometry_dtype) + 0.5
    ) / height
    column_fraction = (
        torch.arange(width, device=device, dtype=geometry_dtype) + 0.5
    ) / width

    for sample_index in range(batch_size):
        sample_center = center[sample_index]
        sample_bounds = bounds[sample_index]
        if (
            not torch.isfinite(sample_center).all()
            or not torch.isfinite(sample_bounds).all()
        ):
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
        delta_lon = torch.remainder(longitude_grid - center_lon + 180.0, 360.0) - 180.0
        north_km = torch.deg2rad(latitude_grid - center_lat) * EARTH_RADIUS_KM
        east_km = (
            torch.deg2rad(delta_lon)
            * EARTH_RADIUS_KM
            * torch.cos(torch.deg2rad(center_lat)).clamp_min(1e-6)
        )
        radius_km = torch.sqrt(north_km.square() + east_km.square())
        field = prediction_ms[sample_index, 0]
        valid = (
            prediction_mask[sample_index, 0].bool()
            & torch.isfinite(field)
            & torch.isfinite(radius_km)
        )
        if not valid.any():
            continue

        directional_extents = torch.stack(
            [
                north_km[valid].max(),
                -north_km[valid].min(),
                east_km[valid].max(),
                -east_km[valid].min(),
            ]
        )
        if bool((directional_extents <= 0.0).any()):
            continue
        complete_radius = directional_extents.min()
        lower_edges = torch.arange(
            0.0,
            float(complete_radius.detach()),
            radial_bin_km,
            device=device,
            dtype=geometry_dtype,
        )
        profile_radii = []
        profile_winds = []
        for lower in lower_edges:
            upper = lower + radial_bin_km
            if upper > complete_radius:
                continue
            annulus = valid & (radius_km >= lower) & (radius_km < upper)
            if int(annulus.sum().detach()) < min_bin_pixels:
                continue
            profile_radii.append(lower + radial_bin_km / 2.0)
            profile_winds.append(field[annulus].to(geometry_dtype).mean())
        if not profile_winds:
            continue
        profile_radii_tensor = torch.stack(profile_radii)
        profile_winds_tensor = torch.stack(profile_winds)
        predicted = {
            "rmw": profile_radii_tensor[profile_winds_tensor.argmax()],
        }
        latitude_edges = torch.linspace(
            top,
            bottom,
            height + 1,
            device=device,
            dtype=geometry_dtype,
        )
        longitude_width_rad = torch.deg2rad((right - left).abs() / width)
        row_area_km2 = (
            EARTH_RADIUS_KM**2
            * longitude_width_rad
            * (
                torch.sin(torch.deg2rad(latitude_edges[:-1]))
                - torch.sin(torch.deg2rad(latitude_edges[1:]))
            ).abs()
        )
        pixel_area_km2 = row_area_km2[:, None].expand(height, width)
        complete_domain = valid & (radius_km <= complete_radius)
        for name, threshold_ms in IBTRACS_WIND_RADIUS_THRESHOLDS_MS.items():
            exceedance_area_km2 = pixel_area_km2[
                complete_domain & (field >= threshold_ms)
            ].sum()
            predicted[name] = torch.sqrt(exceedance_area_km2.clamp_min(0.0) / torch.pi)

        for radius_index, name in enumerate(IBTRACS_RADIUS_NAMES):
            target = target_radii_km[sample_index, radius_index]
            if (
                not target_valid[sample_index, radius_index]
                or not torch.isfinite(target)
                or target < 0.0
                or target > complete_radius
            ):
                continue
            estimate = predicted[name]
            error = estimate - target
            statistics[radius_index] += torch.stack(
                [
                    estimate,
                    target,
                    error.abs(),
                    error.square(),
                    error,
                    error.new_ones(()),
                ]
            ).to(statistics)
    return statistics


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
    ring_region = (radius_km >= EYE_CENTER_RING_INNER_KM) & (
        radius_km < EYE_CENTER_RING_OUTER_KM
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
    if radius_km.flatten()[target_flat_index] > EYE_CENTER_TARGET_RADIUS_KM:
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
    row_fraction = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    column_fraction = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
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
        if (
            not torch.isfinite(sample_center).all()
            or not torch.isfinite(sample_bounds).all()
        ):
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
            metric_values["eye_center_displacement_km"].append(eye_center_displacement)

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
