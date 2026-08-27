from __future__ import annotations

import pytest
import torch

from src.DenoisingDiffusionProcess.beta_schedules import (
    cosine_beta_schedule,
    get_beta_schedule,
    linear_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
)


@pytest.mark.parametrize(
    ("name", "schedule_fn"),
    [
        ("linear", linear_beta_schedule),
        ("quadratic", quadratic_beta_schedule),
        ("sigmoid", sigmoid_beta_schedule),
        ("cosine", cosine_beta_schedule),
    ],
)
def test_beta_schedules_return_valid_timestep_vector(name, schedule_fn) -> None:
    betas = schedule_fn(16)

    assert betas.shape == (16,)
    assert betas.dtype == torch.float32
    assert torch.isfinite(betas).all()
    assert torch.all((betas > 0) & (betas < 1))
    assert torch.allclose(get_beta_schedule(name, 16), betas)


@pytest.mark.parametrize(
    "schedule_fn",
    [linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule],
)
def test_monotonic_beta_schedules_increase_noise(schedule_fn) -> None:
    betas = schedule_fn(32)

    assert torch.all(betas[1:] >= betas[:-1])


def test_linear_beta_schedule_uses_expected_endpoints() -> None:
    betas = linear_beta_schedule(5)

    assert betas[0] == pytest.approx(0.0001)
    assert betas[-1] == pytest.approx(0.02)


def test_cosine_beta_schedule_clips_extreme_final_value() -> None:
    betas = cosine_beta_schedule(8)

    assert betas[-1] == pytest.approx(0.9999)


def test_unknown_beta_schedule_raises_helpful_error() -> None:
    with pytest.raises(NotImplementedError, match="Unknown beta schedule"):
        get_beta_schedule("not-a-schedule", 4)
