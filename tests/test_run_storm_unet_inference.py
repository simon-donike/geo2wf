import math

import pytest
import torch

from scripts.run_storm_unet_inference import (
    CROP_SIZE,
    _bounds,
    _output_metrics,
    _physical_distance_km,
)


def test_output_metrics_extracts_threshold_radii_from_predicted_image() -> None:
    center = (15.0, -55.0)
    bounds = _bounds(*center)
    distance_km = _physical_distance_km(bounds, center)
    field = torch.where(
        distance_km < 40.0,
        40.0,
        torch.where(
            distance_km < 80.0,
            30.0,
            torch.where(distance_km < 120.0, 20.0, 10.0),
        ),
    )
    output = field[None, None]
    valid = torch.ones((1, 1, CROP_SIZE, CROP_SIZE), dtype=torch.bool)

    metrics = _output_metrics(output, valid, distance_km, bounds, center)

    assert metrics["r34"] == pytest.approx(120.0, abs=4.0)
    assert metrics["r50"] == pytest.approx(80.0, abs=4.0)
    assert metrics["r64"] == pytest.approx(40.0, abs=4.0)
    assert metrics["r34"] > metrics["r50"] > metrics["r64"]


def test_output_metrics_returns_nan_radii_when_image_coverage_is_too_low() -> None:
    center = (15.0, -55.0)
    bounds = _bounds(*center)
    distance_km = _physical_distance_km(bounds, center)
    output = torch.full((1, 1, CROP_SIZE, CROP_SIZE), 40.0)
    valid = torch.zeros_like(output, dtype=torch.bool)
    valid[..., :10, :10] = True

    metrics = _output_metrics(output, valid, distance_km, bounds, center)

    assert all(math.isnan(metrics[name]) for name in ("rmw", "r34", "r50", "r64"))


def test_output_metrics_returns_nan_radii_when_center_is_outside_footprint() -> None:
    image_center = (15.0, -55.0)
    storm_center = (25.0, -55.0)
    bounds = _bounds(*image_center)
    distance_km = _physical_distance_km(bounds, storm_center)
    output = torch.full((1, 1, CROP_SIZE, CROP_SIZE), 40.0)
    valid = torch.ones_like(output, dtype=torch.bool)

    metrics = _output_metrics(output, valid, distance_km, bounds, storm_center)

    assert math.isfinite(metrics["msw"])
    assert all(math.isnan(metrics[name]) for name in ("rmw", "r34", "r50", "r64"))
