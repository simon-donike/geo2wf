from __future__ import annotations

import pytest
import torch

from src.DenoisingDiffusionProcess.forward import GaussianForwardProcess


def test_gaussian_forward_process_registers_expected_buffers() -> None:
    process = GaussianForwardProcess(num_timesteps=7, schedule="linear")

    for name in (
        "betas",
        "betas_sqrt",
        "alphas",
        "alphas_cumprod",
        "alphas_cumprod_sqrt",
        "alphas_one_minus_cumprod_sqrt",
        "alphas_sqrt",
    ):
        assert name in dict(process.named_buffers())
        assert getattr(process, name).shape == (7,)


def test_forward_diffusion_returns_sample_and_noise_that_match_formula() -> None:
    process = GaussianForwardProcess(num_timesteps=4, schedule="linear")
    x_0 = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).view(2, 3, 2, 2)
    t = torch.tensor([0, 3])

    torch.manual_seed(123)
    output, noise = process(x_0, t, return_noise=True)

    expected = (
        x_0 * process.alphas_cumprod_sqrt[t].view(2, 1, 1, 1)
        + noise * process.alphas_one_minus_cumprod_sqrt[t].view(2, 1, 1, 1)
    )
    assert output.shape == x_0.shape
    assert noise.shape == x_0.shape
    assert torch.allclose(output, expected)


def test_forward_diffusion_can_return_only_noisy_sample() -> None:
    process = GaussianForwardProcess(num_timesteps=3, schedule="linear")
    x_0 = torch.ones(1, 1, 2, 2)

    output = process(x_0, torch.tensor([1]))

    assert output.shape == x_0.shape


def test_forward_diffusion_rejects_out_of_range_timestep() -> None:
    process = GaussianForwardProcess(num_timesteps=3, schedule="linear")

    with pytest.raises(AssertionError):
        process(torch.ones(1, 1, 2, 2), torch.tensor([3]))


def test_step_uses_current_sample_shape_and_broadcasts_batch_timesteps() -> None:
    process = GaussianForwardProcess(num_timesteps=5, schedule="linear")
    x_t = torch.ones(2, 3, 4, 5)
    t = torch.tensor([0, 4])

    output, noise = process.step(x_t, t, return_noise=True)

    expected = (
        process.alphas_sqrt[t].view(2, 1, 1, 1) * x_t
        + process.betas_sqrt[t].view(2, 1, 1, 1) * noise
    )
    assert output.shape == x_t.shape
    assert noise.shape == x_t.shape
    assert torch.allclose(output, expected)
