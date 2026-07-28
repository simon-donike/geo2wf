from __future__ import annotations

import torch

from src.DenoisingDiffusionProcess.samplers.DDIM import DDIM_Sampler
from src.DenoisingDiffusionProcess.samplers.DDPM import DDPM_Sampler


def test_ddpm_posterior_params_match_closed_form() -> None:
    sampler = DDPM_Sampler(num_timesteps=5, schedule="linear")
    x_t = torch.full((2, 1, 2, 2), 0.75)
    noise_pred = torch.full_like(x_t, 0.25)
    t = torch.tensor([1, 4])

    mean, std = sampler.posterior_params(x_t, t, noise_pred)

    expected_mean = sampler.alphas_sqrt_recip[t].view(2, 1, 1, 1) * (
        x_t
        - sampler.betas[t].view(2, 1, 1, 1)
        * noise_pred
        / sampler.alphas_one_minus_cumprod_sqrt[t].view(2, 1, 1, 1)
    )
    expected_std = sampler.betas_sqrt[t].view(2, 1, 1, 1)
    assert torch.allclose(mean, expected_mean)
    assert torch.allclose(std, expected_std)


def test_ddpm_step_is_deterministic_at_timestep_zero() -> None:
    sampler = DDPM_Sampler(num_timesteps=5, schedule="linear")
    x_t = torch.full((1, 1, 2, 2), 0.75)
    noise_pred = torch.full_like(x_t, 0.25)
    t = torch.tensor([0])

    mean, _ = sampler.posterior_params(x_t, t, noise_pred)

    assert torch.allclose(sampler.step(x_t, t, noise_pred), mean)


def test_ddim_estimate_origin_matches_diffusion_rearrangement() -> None:
    sampler = DDIM_Sampler(num_timesteps=4, train_timesteps=8, schedule="linear")
    x_t = torch.full((2, 1, 2, 2), 0.5)
    z_t = torch.full_like(x_t, 0.1)
    t = torch.tensor([0, 6])

    x_0 = sampler.estimate_origin(x_t, t, z_t)

    expected = (
        x_t
        - sampler.alphas_one_minus_cumprod_sqrt[t].view(2, 1, 1, 1) * z_t
    ) / sampler.alphas_cumprod[t].view(2, 1, 1, 1).sqrt()
    assert torch.allclose(x_0, expected)


def test_ddim_step_returns_previous_sample_shape_without_noise() -> None:
    sampler = DDIM_Sampler(
        num_timesteps=4,
        train_timesteps=8,
        clip_sample=False,
        schedule="linear",
    )
    x_t = torch.full((2, 1, 3, 3), 0.5)
    z_t = torch.full_like(x_t, 0.1)
    t = torch.tensor([0, 3])

    prev_sample = sampler.step(x_t, t, z_t, eta=0)

    assert prev_sample.shape == x_t.shape
    assert torch.isfinite(prev_sample).all()


def test_ddim_step_with_eta_adds_finite_noise() -> None:
    sampler = DDIM_Sampler(
        num_timesteps=4,
        train_timesteps=8,
        clip_sample=True,
        schedule="linear",
    )
    x_t = torch.full((1, 1, 3, 3), 0.5)
    z_t = torch.full_like(x_t, 0.1)

    prev_sample = sampler.step(x_t, torch.tensor([2]), z_t, eta=0.5)

    assert prev_sample.shape == x_t.shape
    assert torch.isfinite(prev_sample).all()
