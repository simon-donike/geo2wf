from __future__ import annotations

import torch

from src.wind_metrics import (
    IBTRACS_RADIUS_NAMES,
    RADIAL_METRIC_NAMES,
    ibtracs_radius_metric_statistics,
    radial_wind_metric_statistics,
    radial_wind_metrics,
)


def _ring_field(radius_km: torch.Tensor, peak_radius_km: float) -> torch.Tensor:
    return 10.0 + 35.0 * torch.exp(
        -0.5 * ((radius_km - peak_radius_km) / 10.0).square()
    )


def _radial_grid(size: int = 41) -> tuple[torch.Tensor, torch.Tensor]:
    # At the equator, one degree is about 111 km. These are pixel-center
    # coordinates for the [-2, 2] degree test bounds.
    coordinates = 2.0 - (torch.arange(size) + 0.5) * (4.0 / size)
    latitudes, longitudes = torch.meshgrid(coordinates, -coordinates, indexing="ij")
    radius_km = 6371.0 * torch.deg2rad(
        torch.sqrt(latitudes.square() + longitudes.square())
    )
    return radius_km, torch.tensor([[-2.0, 2.0, -2.0, 2.0]])


def _local_xy_grid(
    size: int = 81,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinates = 2.0 - (torch.arange(size) + 0.5) * (4.0 / size)
    latitudes, longitudes = torch.meshgrid(coordinates, -coordinates, indexing="ij")
    north_km = 6371.0 * torch.deg2rad(latitudes)
    east_km = 6371.0 * torch.deg2rad(longitudes)
    return east_km, north_km, torch.tensor([[-2.0, 2.0, -2.0, 2.0]])


def _eye_field(
    east_km: torch.Tensor,
    north_km: torch.Tensor,
    *,
    center_east_km: float = 0.0,
) -> torch.Tensor:
    radius_km = torch.sqrt((east_km - center_east_km).square() + north_km.square())
    eye = 15.0 * torch.exp(-0.5 * (radius_km / 10.0).square())
    eyewall = 25.0 * torch.exp(-0.5 * ((radius_km - 45.0) / 10.0).square())
    return 20.0 - eye + eyewall


def test_identical_fields_have_zero_storm_centric_errors() -> None:
    radius_km, bounds = _radial_grid()
    target = _ring_field(radius_km, 55.0)[None, None]
    mask = torch.ones_like(target, dtype=torch.bool)

    metrics = radial_wind_metrics(
        target,
        target,
        mask,
        torch.tensor([[0.0, 0.0]]),
        bounds,
    )

    assert {
        "eye_mae_ms",
        "inner_core_mae_ms",
        "radial_profile_mae_ms",
        "rmw_error_km",
        "eye_to_eyewall_contrast_error_ms",
    }.issubset(metrics)
    assert all(torch.allclose(value, torch.tensor(0.0)) for value in metrics.values())


def test_eye_center_displacement_tracks_shifted_low_wind_eye() -> None:
    east_km, north_km, bounds = _local_xy_grid()
    target = _eye_field(east_km, north_km)[None, None]
    prediction = _eye_field(east_km, north_km, center_east_km=30.0)[None, None]

    metrics = radial_wind_metrics(
        prediction,
        target,
        torch.ones_like(target, dtype=torch.bool),
        torch.tensor([[0.0, 0.0]]),
        bounds,
    )

    assert torch.isclose(
        metrics["eye_center_displacement_km"],
        torch.tensor(30.0),
        atol=7.0,
    )


def test_eye_center_displacement_skips_target_without_eye_contrast() -> None:
    _, _, bounds = _local_xy_grid()
    target = torch.full((1, 1, 81, 81), 20.0)

    metrics = radial_wind_metrics(
        target,
        target,
        torch.ones_like(target, dtype=torch.bool),
        torch.tensor([[0.0, 0.0]]),
        bounds,
    )

    assert "eye_center_displacement_km" not in metrics


def test_eye_center_displacement_skips_outer_target_minimum() -> None:
    east_km, north_km, bounds = _local_xy_grid()
    outer_low = 18.0 * torch.exp(
        -0.5 * (((east_km - 80.0) / 8.0).square() + (north_km / 8.0).square())
    )
    target = (_eye_field(east_km, north_km) - outer_low)[None, None]

    metrics = radial_wind_metrics(
        target,
        target,
        torch.ones_like(target, dtype=torch.bool),
        torch.tensor([[0.0, 0.0]]),
        bounds,
    )

    # The global low-wind point is outside the plausible 50 km target-eye
    # radius. Skipping avoids assigning a nonzero displacement to an identical
    # prediction merely because its search radius extends to 100 km.
    assert "eye_center_displacement_km" not in metrics


def test_eye_center_displacement_skips_incomplete_inner_core() -> None:
    east_km, north_km, bounds = _local_xy_grid()
    target = _eye_field(east_km, north_km)[None, None]
    mask = (east_km >= 0.0)[None, None]

    metrics = radial_wind_metrics(
        target,
        target,
        mask,
        torch.tensor([[0.0, 0.0]]),
        bounds,
    )

    assert "eye_center_displacement_km" not in metrics


def test_shifted_eyewall_has_nonzero_rmw_and_profile_error() -> None:
    radius_km, bounds = _radial_grid()
    target = _ring_field(radius_km, 45.0)[None, None]
    prediction = _ring_field(radius_km, 85.0)[None, None]

    metrics = radial_wind_metrics(
        prediction,
        target,
        torch.ones_like(target, dtype=torch.bool),
        torch.tensor([[0.0, 0.0]]),
        bounds,
    )

    assert metrics["rmw_error_km"] >= 30.0
    assert metrics["radial_profile_mae_ms"] > 0.0


def test_missing_center_skips_metrics_without_fabricating_location() -> None:
    target = torch.zeros(1, 1, 16, 16)
    metrics = radial_wind_metrics(
        target,
        target,
        torch.ones_like(target, dtype=torch.bool),
        torch.tensor([[float("nan"), float("nan")]]),
        torch.tensor([[-1.0, 1.0, -1.0, 1.0]]),
    )

    assert metrics == {}


def test_statistics_keep_fixed_shape_when_metrics_are_unavailable() -> None:
    target = torch.zeros(1, 1, 16, 16)
    statistics = radial_wind_metric_statistics(
        target,
        target,
        torch.ones_like(target, dtype=torch.bool),
        torch.tensor([[float("nan"), float("nan")]]),
        torch.tensor([[-1.0, 1.0, -1.0, 1.0]]),
    )

    assert statistics.shape == (len(RADIAL_METRIC_NAMES), 2)
    assert torch.count_nonzero(statistics) == 0


def test_generated_field_radii_are_compared_with_combined_ibtracs_targets() -> None:
    radius_km, bounds = _radial_grid(size=161)
    field = torch.where(
        radius_km < 40.0,
        torch.tensor(40.0),
        torch.where(
            radius_km < 80.0,
            torch.tensor(30.0),
            torch.where(radius_km < 120.0, torch.tensor(20.0), torch.tensor(10.0)),
        ),
    )[None, None]
    mask = torch.ones_like(field, dtype=torch.bool)
    center = torch.tensor([[0.0, 0.0]])
    valid = torch.ones((1, len(IBTRACS_RADIUS_NAMES)), dtype=torch.bool)

    initial = ibtracs_radius_metric_statistics(
        field,
        mask,
        center,
        bounds,
        torch.zeros((1, len(IBTRACS_RADIUS_NAMES))),
        valid,
    )
    predicted = initial[:, 0]

    assert initial[:, 5].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert predicted[1] > predicted[2] > predicted[3] > predicted[0]

    matched = ibtracs_radius_metric_statistics(
        field,
        mask,
        center,
        bounds,
        predicted[None],
        valid,
    )
    assert torch.count_nonzero(matched[:, 2:5]) == 0


def test_generated_field_radius_metrics_respect_ground_truth_validity() -> None:
    radius_km, bounds = _radial_grid()
    field = _ring_field(radius_km, 55.0)[None, None]
    valid = torch.tensor([[True, False, True, False]])
    statistics = ibtracs_radius_metric_statistics(
        field,
        torch.ones_like(field, dtype=torch.bool),
        torch.tensor([[0.0, 0.0]]),
        bounds,
        torch.full((1, len(IBTRACS_RADIUS_NAMES)), 50.0),
        valid,
    )

    assert statistics[:, 5].tolist() == [1.0, 0.0, 1.0, 0.0]
