"""DDIM reverse-process sampler."""

from __future__ import annotations

import torch
from torch import nn

from ..beta_schedules import get_beta_schedule
from .DDPM import _extract, _randn_like


class DDIM_Sampler(nn.Module):
    """DDIM sampler whose public timesteps are exact training timesteps.

    The sampler owns an endpoint-inclusive schedule and maps every scheduled
    training timestep to its exact predecessor. This keeps model time embeddings
    and reverse-process coefficients on the same clock.
    """

    def __init__(
        self,
        num_timesteps: int = 100,
        train_timesteps: int = 1000,
        clip_sample: bool = True,
        schedule: str = "linear",
        eta: float = 0.0,
    ) -> None:
        super().__init__()
        if train_timesteps < 1:
            raise ValueError("train_timesteps must be at least 1")
        if num_timesteps < 1 or num_timesteps > train_timesteps:
            raise ValueError(
                "num_timesteps must be between 1 and train_timesteps"
            )
        if eta < 0:
            raise ValueError("eta must be non-negative")

        self.num_timesteps = int(num_timesteps)
        self.train_timesteps = int(train_timesteps)
        self.clip_sample = bool(clip_sample)
        self.schedule = schedule
        self.eta = float(eta)

        betas = get_beta_schedule(self.schedule, self.train_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        timesteps = torch.linspace(
            self.train_timesteps - 1,
            0,
            steps=self.num_timesteps,
            dtype=torch.float64,
        ).round().long()
        previous_timesteps = torch.cat(
            [timesteps[1:], torch.full((1,), -1, dtype=torch.long)]
        )
        previous_lookup = torch.full(
            (self.train_timesteps,), -2, dtype=torch.long
        )
        previous_lookup[timesteps] = previous_timesteps

        self.register_buffer("timesteps", timesteps, persistent=False)
        self.register_buffer(
            "previous_timesteps", previous_timesteps, persistent=False
        )
        self.register_buffer(
            "_previous_timestep_lookup", previous_lookup, persistent=False
        )
        self.register_buffer(
            "final_alpha_cumprod", torch.tensor(1.0), persistent=False
        )
        self.register_buffer("betas", betas)
        self.register_buffer("betas_sqrt", betas.sqrt())
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_sqrt", alphas_cumprod.sqrt())
        self.register_buffer(
            "alphas_one_minus_cumprod_sqrt", (1 - alphas_cumprod).sqrt()
        )
        self.register_buffer("alphas_sqrt", alphas.sqrt())
        self.register_buffer("alphas_sqrt_recip", alphas.rsqrt())

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.step(*args, **kwargs)

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
        eta: float | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Predict the previous scheduled sample from an epsilon prediction."""
        self._validate_timesteps(t)
        eta_value = self.eta if eta is None else float(eta)
        if eta_value < 0:
            raise ValueError("eta must be non-negative")

        t_prev = self.previous_timestep(t)
        alpha_bar_prev = self._previous_alpha_cumprod(t_prev, x_t)

        x_0_pred = self.estimate_origin(x_t, t, noise_pred)
        if self.clip_sample:
            x_0_pred = x_0_pred.clamp(-1, 1)

        std_dev = eta_value * self.estimate_std(t, t_prev).view(
            x_t.shape[0], *((1,) * (x_t.ndim - 1))
        )
        direction_scale = (
            1 - alpha_bar_prev - std_dev.square()
        ).clamp_min(0).sqrt()
        previous_sample = (
            alpha_bar_prev.sqrt() * x_0_pred + direction_scale * noise_pred
        )

        if eta_value > 0 and bool((std_dev > 0).any()):
            previous_sample = previous_sample + std_dev * _randn_like(
                x_t, generator=generator
            )
        return previous_sample

    def previous_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """Return the exact predecessor for each scheduled training timestep."""
        self._validate_timesteps(t)
        return self._previous_timestep_lookup[t]

    def estimate_std(
        self,
        t: torch.Tensor,
        t_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Return DDIM posterior std before multiplication by ``eta``."""
        self._validate_training_range(t)
        if not bool(((t_prev >= -1) & (t_prev < self.train_timesteps)).all()):
            raise ValueError(
                f"previous timesteps must be in [-1, {self.train_timesteps - 1}]"
            )
        alpha_bar_t = self.alphas_cumprod[t]
        safe_prev = t_prev.clamp_min(0)
        alpha_bar_prev = torch.where(
            t_prev >= 0,
            self.alphas_cumprod[safe_prev],
            self.final_alpha_cumprod.expand_as(alpha_bar_t),
        )
        variance = (
            (1 - alpha_bar_prev)
            / (1 - alpha_bar_t).clamp_min(1e-20)
            * (1 - alpha_bar_t / alpha_bar_prev.clamp_min(1e-20))
        )
        return variance.clamp_min(0).sqrt()

    def estimate_origin(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the clean-sample prediction from an epsilon prediction."""
        self._validate_training_range(t)
        alpha_bar_sqrt = _extract(self.alphas_cumprod_sqrt, t, x_t)
        one_minus_alpha_bar_sqrt = _extract(
            self.alphas_one_minus_cumprod_sqrt, t, x_t
        )
        return (x_t - one_minus_alpha_bar_sqrt * noise_pred) / alpha_bar_sqrt

    def _previous_alpha_cumprod(
        self,
        t_prev: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        safe_prev = t_prev.clamp_min(0)
        values = torch.where(
            t_prev >= 0,
            self.alphas_cumprod[safe_prev],
            self.final_alpha_cumprod.expand_as(t_prev),
        )
        return values.view(
            reference.shape[0], *((1,) * (reference.ndim - 1))
        )

    def _validate_timesteps(self, t: torch.Tensor) -> None:
        self._validate_training_range(t)
        if not bool((self._previous_timestep_lookup[t] != -2).all()):
            allowed = self.timesteps.detach().cpu().tolist()
            raise ValueError(
                f"DDIM timestep is not in the sampling schedule: {allowed}"
            )

    def _validate_training_range(self, t: torch.Tensor) -> None:
        if t.ndim != 1:
            raise ValueError("t must be a one-dimensional batch of timesteps")
        if not bool(((t >= 0) & (t < self.train_timesteps)).all()):
            raise ValueError(
                f"DDIM timesteps must be in [0, {self.train_timesteps - 1}]"
            )
