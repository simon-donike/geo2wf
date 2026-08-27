from __future__ import annotations

import pytest
import torch

from src.DenoisingDiffusionProcess.samplers.DDIM import DDIM_Sampler
from src.DenoisingDiffusionProcess.samplers.DDPM import DDPM_Sampler


def test_ddpm_exposes_descending_training_timesteps() -> None:
    sampler = DDPM_Sampler(num_timesteps=5, schedule="linear")

    assert sampler.train_timesteps == 5
    assert torch.equal(sampler.timesteps, torch.tensor([4, 3, 2, 1, 0]))


def test_ddpm_posterior_params_match_exact_closed_form() -> None:
    sampler = DDPM_Sampler(
        num_timesteps=5,
        schedule="linear",
        clip_sample=False,
    )
    x_t = torch.full((2, 1, 2, 2), 0.75)
    noise_pred = torch.full_like(x_t, 0.25)
    t = torch.tensor([1, 4])

    mean, std = sampler.posterior_params(x_t, t, noise_pred)

    alpha_bar = sampler.alphas_cumprod[t].view(2, 1, 1, 1)
    alpha_bar_prev = sampler.alphas_cumprod_prev[t].view(2, 1, 1, 1)
    alpha = sampler.alphas[t].view(2, 1, 1, 1)
    beta = sampler.betas[t].view(2, 1, 1, 1)
    x_0 = (x_t - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
    expected_mean = (
        beta * alpha_bar_prev.sqrt() / (1 - alpha_bar) * x_0
        + (1 - alpha_bar_prev) * alpha.sqrt() / (1 - alpha_bar) * x_t
    )
    expected_std = (
        beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
    ).sqrt()

    assert torch.allclose(mean, expected_mean)
    assert torch.allclose(std, expected_std)


def test_ddpm_clips_predicted_origin_before_posterior_mean() -> None:
    clipped = DDPM_Sampler(num_timesteps=5, clip_sample=True)
    unclipped = DDPM_Sampler(num_timesteps=5, clip_sample=False)
    x_t = torch.full((1, 1, 2, 2), 20.0)
    noise_pred = torch.zeros_like(x_t)
    t = torch.tensor([0])

    clipped_mean, _ = clipped.posterior_params(x_t, t, noise_pred)
    unclipped_mean, _ = unclipped.posterior_params(x_t, t, noise_pred)

    assert clipped_mean.max() <= 1.0
    assert unclipped_mean.min() > 1.0


def test_ddpm_step_is_deterministic_at_timestep_zero() -> None:
    sampler = DDPM_Sampler(num_timesteps=5, schedule="linear")
    x_t = torch.full((1, 1, 2, 2), 0.75)
    noise_pred = torch.full_like(x_t, 0.25)
    t = torch.tensor([0])

    mean, _ = sampler.posterior_params(x_t, t, noise_pred)

    assert torch.allclose(sampler.step(x_t, t, noise_pred), mean)


def test_ddpm_generator_reproduces_stochastic_step() -> None:
    sampler = DDPM_Sampler(num_timesteps=5, schedule="linear")
    x_t = torch.full((1, 1, 4, 4), 0.5)
    noise_pred = torch.full_like(x_t, 0.1)
    t = torch.tensor([4])

    first = sampler.step(
        x_t, t, noise_pred, generator=torch.Generator().manual_seed(123)
    )
    second = sampler.step(
        x_t, t, noise_pred, generator=torch.Generator().manual_seed(123)
    )
    different = sampler.step(
        x_t, t, noise_pred, generator=torch.Generator().manual_seed(456)
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_ddim_uses_endpoint_inclusive_actual_training_timesteps() -> None:
    sampler = DDIM_Sampler(num_timesteps=4, train_timesteps=8)

    assert torch.equal(sampler.timesteps, torch.tensor([7, 5, 2, 0]))
    assert torch.equal(sampler.previous_timesteps, torch.tensor([5, 2, 0, -1]))
    assert torch.equal(
        sampler.previous_timestep(torch.tensor([7, 2, 0])),
        torch.tensor([5, 0, -1]),
    )


def test_ddim_estimate_origin_matches_diffusion_rearrangement() -> None:
    sampler = DDIM_Sampler(num_timesteps=4, train_timesteps=8, schedule="linear")
    x_t = torch.full((2, 1, 2, 2), 0.5)
    noise_pred = torch.full_like(x_t, 0.1)
    t = torch.tensor([0, 7])

    x_0 = sampler.estimate_origin(x_t, t, noise_pred)

    expected = (
        x_t
        - sampler.alphas_one_minus_cumprod_sqrt[t].view(2, 1, 1, 1)
        * noise_pred
    ) / sampler.alphas_cumprod[t].view(2, 1, 1, 1).sqrt()
    assert torch.allclose(x_0, expected)


def test_ddim_deterministic_step_uses_exact_scheduled_predecessor() -> None:
    sampler = DDIM_Sampler(
        num_timesteps=4,
        train_timesteps=8,
        clip_sample=False,
        schedule="linear",
    )
    x_t = torch.full((2, 1, 3, 3), 0.5)
    noise_pred = torch.full_like(x_t, 0.1)
    t = torch.tensor([7, 2])
    t_prev = torch.tensor([5, 0])

    previous_sample = sampler.step(x_t, t, noise_pred, eta=0)

    x_0 = sampler.estimate_origin(x_t, t, noise_pred)
    alpha_bar_prev = sampler.alphas_cumprod[t_prev].view(2, 1, 1, 1)
    expected = alpha_bar_prev.sqrt() * x_0 + (1 - alpha_bar_prev).sqrt() * noise_pred
    assert torch.allclose(previous_sample, expected)


def test_ddim_final_step_returns_clipped_origin() -> None:
    sampler = DDIM_Sampler(
        num_timesteps=4,
        train_timesteps=8,
        clip_sample=True,
    )
    x_t = torch.full((1, 1, 2, 2), 10.0)
    noise_pred = torch.zeros_like(x_t)

    previous_sample = sampler.step(x_t, torch.tensor([0]), noise_pred)

    assert torch.equal(previous_sample, torch.ones_like(previous_sample))


def test_ddim_generator_reproduces_eta_noise() -> None:
    sampler = DDIM_Sampler(
        num_timesteps=4,
        train_timesteps=8,
        clip_sample=True,
        eta=0.5,
    )
    x_t = torch.full((1, 1, 3, 3), 0.5)
    noise_pred = torch.full_like(x_t, 0.1)
    t = torch.tensor([5])

    first = sampler.step(
        x_t, t, noise_pred, generator=torch.Generator().manual_seed(123)
    )
    second = sampler.step(
        x_t, t, noise_pred, generator=torch.Generator().manual_seed(123)
    )
    different = sampler.step(
        x_t, t, noise_pred, generator=torch.Generator().manual_seed(456)
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert torch.isfinite(first).all()


def test_ddim_rejects_timestep_outside_sampling_schedule() -> None:
    sampler = DDIM_Sampler(num_timesteps=4, train_timesteps=8)
    x_t = torch.zeros(1, 1, 2, 2)

    with pytest.raises(ValueError, match="not in the sampling schedule"):
        sampler.step(x_t, torch.tensor([3]), torch.zeros_like(x_t))
