from __future__ import annotations

import pytest
import torch

from geo2wf.metrics.image_quality import (
    WIND_SPEED_DATA_RANGE_MS,
    masked_ssim_sum_count,
    psnr_db_from_mse,
)


def test_physical_psnr_uses_fixed_wind_range() -> None:
    assert psnr_db_from_mse(4.0) == pytest.approx(
        20.0 * torch.log10(torch.tensor(WIND_SPEED_DATA_RANGE_MS / 2.0)).item()
    )


def test_masked_ssim_is_one_for_identical_valid_windows() -> None:
    target = torch.linspace(0.2, 80.0, 15 * 15).reshape(1, 1, 15, 15)
    mask = torch.ones_like(target, dtype=torch.bool)

    score_sum, scenes = masked_ssim_sum_count(target, target, mask)

    assert scenes == 1
    assert score_sum == pytest.approx(1.0)


def test_masked_ssim_ignores_changes_outside_complete_windows() -> None:
    target = torch.full((1, 1, 15, 15), 20.0)
    prediction = target.clone()
    prediction[:, :, :3, :] = 80.0
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, :, :3, :] = False

    score_sum, scenes = masked_ssim_sum_count(prediction, target, mask)

    assert scenes == 1
    assert score_sum == pytest.approx(1.0)
