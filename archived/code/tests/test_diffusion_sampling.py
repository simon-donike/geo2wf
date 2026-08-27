from __future__ import annotations

import pytest
import torch
from torch import nn

from src.DenoisingDiffusionProcess.DenoisingDiffusionProcess import (
    DenoisingDiffusionConditionalProcess,
    DenoisingDiffusionProcess,
    _run_reverse_process,
)
from src.DenoisingDiffusionProcess.samplers.DDIM import DDIM_Sampler
from src.DenoisingDiffusionProcess.samplers.DDPM import DDPM_Sampler


class RecordingDenoiser(nn.Module):
    def __init__(self, output_channels: int = 1) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.output_channels = output_channels
        self.timesteps: list[torch.Tensor] = []

    def forward(self, value: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        self.timesteps.append(timestep.detach().cpu().clone())
        return torch.zeros_like(value[:, : self.output_channels])


def _conditional_process(schedule: str = "linear") -> DenoisingDiffusionConditionalProcess:
    return DenoisingDiffusionConditionalProcess(
        generated_channels=1,
        condition_channels=1,
        num_timesteps=8,
        schedule=schedule,
        model_dim=8,
        model_dim_mults=(1, 2),
        model_channels=2,
        model_out_dim=1,
    )


def _unconditional_process(schedule: str = "linear") -> DenoisingDiffusionProcess:
    return DenoisingDiffusionProcess(
        generated_channels=1,
        num_timesteps=8,
        schedule=schedule,
        model_dim=8,
        model_dim_mults=(1, 2),
        model_channels=1,
        model_out_dim=1,
    )


def test_default_ddpm_inherits_forward_process_schedule() -> None:
    process = _unconditional_process(schedule="cosine")

    assert process.sampler.schedule == "cosine"
    assert process.sampler.train_timesteps == process.forward_process.num_timesteps
    assert torch.equal(process.sampler.betas, process.forward_process.betas)


def test_conditional_process_dispatches_custom_ddim_at_actual_train_times() -> None:
    process = _conditional_process()
    denoiser = RecordingDenoiser()
    process.model = denoiser
    sampler = DDIM_Sampler(
        num_timesteps=4,
        train_timesteps=8,
        schedule="linear",
    )
    condition = torch.zeros(2, 1, 4, 4)

    output = process(
        condition,
        sampler=sampler,
        initial_noise=torch.zeros(2, 1, 4, 4),
    )

    observed = [int(timestep[0]) for timestep in denoiser.timesteps]
    assert observed == sampler.timesteps.tolist() == [7, 5, 2, 0]
    assert output.shape == (2, 1, 4, 4)
    assert torch.isfinite(output).all()


def test_fixed_initial_noise_makes_eta_zero_ddim_deterministic() -> None:
    process = _conditional_process()
    process.model = RecordingDenoiser()
    sampler = DDIM_Sampler(4, 8, schedule="linear", eta=0)
    condition = torch.zeros(1, 1, 4, 4)
    initial_noise = torch.randn(1, 1, 4, 4)

    first = process(condition, sampler=sampler, initial_noise=initial_noise)
    second = process(condition, sampler=sampler, initial_noise=initial_noise)

    assert torch.equal(first, second)


def test_generator_controls_initial_and_reverse_ddpm_noise() -> None:
    process = _unconditional_process()
    process.model = RecordingDenoiser()

    first = process(
        shape=(4, 4),
        generator=torch.Generator().manual_seed(123),
    )
    second = process(
        shape=(4, 4),
        generator=torch.Generator().manual_seed(123),
    )
    different = process(
        shape=(4, 4),
        generator=torch.Generator().manual_seed(456),
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_process_rejects_sampler_schedule_mismatch() -> None:
    process = _conditional_process(schedule="linear")
    process.model = RecordingDenoiser()
    condition = torch.zeros(1, 1, 4, 4)
    sampler = DDIM_Sampler(4, 8, schedule="cosine")

    with pytest.raises(ValueError, match="Sampler schedule does not match"):
        process(condition, sampler=sampler)


def test_process_rejects_sampler_training_length_mismatch() -> None:
    process = _conditional_process(schedule="linear")
    process.model = RecordingDenoiser()
    condition = torch.zeros(1, 1, 4, 4)
    sampler = DDIM_Sampler(4, 10, schedule="linear")

    with pytest.raises(ValueError, match="train_timesteps does not match"):
        process(condition, sampler=sampler)


def test_process_validates_initial_noise_shape() -> None:
    process = _conditional_process()
    process.model = RecordingDenoiser()
    condition = torch.zeros(1, 1, 4, 4)

    with pytest.raises(ValueError, match="initial_noise has shape"):
        process(condition, initial_noise=torch.zeros(1, 1, 3, 4))


def test_derived_sampler_buffers_do_not_break_old_state_dicts() -> None:
    ddpm_keys = set(DDPM_Sampler(8).state_dict())
    ddim_keys = set(DDIM_Sampler(4, 8).state_dict())

    assert "timesteps" not in ddpm_keys
    assert "alphas_cumprod_prev" not in ddpm_keys
    assert "posterior_variance" not in ddpm_keys
    assert "posterior_std" not in ddpm_keys
    assert "posterior_mean_coef_x0" not in ddpm_keys
    assert "posterior_mean_coef_xt" not in ddpm_keys
    assert "timesteps" not in ddim_keys
    assert "previous_timesteps" not in ddim_keys
    assert "_previous_timestep_lookup" not in ddim_keys
    assert "final_alpha_cumprod" not in ddim_keys


class ConditionEchoDenoiser(nn.Module):
    def forward(self, value: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return value[:, 1:2]


class CaptureSampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("timesteps", torch.tensor([0]), persistent=False)

    def forward(self, x_t, t, noise_pred, generator=None):
        del x_t, t, generator
        return noise_pred


def test_classifier_free_guidance_combines_conditional_noise_predictions() -> None:
    sample = _run_reverse_process(
        ConditionEchoDenoiser(),
        torch.zeros(1, 1, 2, 2),
        CaptureSampler(),
        condition=torch.ones(1, 1, 2, 2),
        unconditional_condition=torch.zeros(1, 1, 2, 2),
        guidance_scale=1.75,
    )

    assert torch.allclose(sample, torch.full_like(sample, 1.75))


def test_min_snr_epsilon_weight_caps_easy_high_snr_steps() -> None:
    snr = torch.tensor([100.0, 5.0, 1.0])

    weight = DenoisingDiffusionConditionalProcess.min_snr_weight(snr, 5.0)

    assert torch.allclose(weight, torch.tensor([0.05, 1.0, 1.0]))
