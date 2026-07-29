"""DDPM reverse-process sampler."""

from __future__ import annotations

import torch
from torch import nn

from ..beta_schedules import get_beta_schedule


def _extract(values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Extract one scalar per batch item and broadcast it over image dimensions."""
    return values[timesteps].view(reference.shape[0], *((1,) * (reference.ndim - 1)))


def _randn_like(
    reference: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create like-shaped Gaussian noise with explicit generator support."""
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


class DDPM_Sampler(nn.Module):
    """Ancestral DDPM sampler using exact posterior coefficients.

    ``timesteps`` contains the actual training timesteps supplied to both the
    denoiser and sampler. Clean-sample predictions are clipped at every reverse
    step by default so low-SNR epsilon errors cannot drive the chain arbitrarily
    far outside its training range.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: str = "linear",
        clip_sample: bool = True,
    ) -> None:
        super().__init__()
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be at least 1")

        self.num_timesteps = int(num_timesteps)
        self.train_timesteps = int(num_timesteps)
        self.schedule = schedule
        self.clip_sample = bool(clip_sample)

        betas = get_beta_schedule(self.schedule, self.train_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]]
        )
        posterior_variance = (
            betas
            * (1 - alphas_cumprod_prev)
            / (1 - alphas_cumprod).clamp_min(1e-20)
        )

        self.register_buffer(
            "timesteps",
            torch.arange(self.train_timesteps - 1, -1, -1),
            persistent=False,
        )
        self.register_buffer("betas", betas)
        self.register_buffer("betas_sqrt", betas.sqrt())
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "alphas_cumprod_prev", alphas_cumprod_prev, persistent=False
        )
        self.register_buffer("alphas_cumprod_sqrt", alphas_cumprod.sqrt())
        self.register_buffer(
            "alphas_one_minus_cumprod_sqrt", (1 - alphas_cumprod).sqrt()
        )
        self.register_buffer("alphas_sqrt", alphas.sqrt())
        self.register_buffer("alphas_sqrt_recip", alphas.rsqrt())
        self.register_buffer(
            "posterior_variance", posterior_variance, persistent=False
        )
        self.register_buffer(
            "posterior_std",
            posterior_variance.clamp_min(0).sqrt(),
            persistent=False,
        )
        self.register_buffer(
            "posterior_mean_coef_x0",
            betas
            * alphas_cumprod_prev.sqrt()
            / (1 - alphas_cumprod).clamp_min(1e-20),
            persistent=False,
        )
        self.register_buffer(
            "posterior_mean_coef_xt",
            (1 - alphas_cumprod_prev)
            * alphas.sqrt()
            / (1 - alphas_cumprod).clamp_min(1e-20),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.step(*args, **kwargs)

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample ``x_(t-1)`` from an epsilon prediction at training time ``t``."""
        mean_pred, std_pred = self.posterior_params(x_t, t, noise_pred)
        nonzero_mask = (t > 0).to(dtype=x_t.dtype).view(
            x_t.shape[0], *((1,) * (x_t.ndim - 1))
        )
        if not bool(nonzero_mask.any()):
            return mean_pred
        return mean_pred + nonzero_mask * std_pred * _randn_like(
            x_t, generator=generator
        )

    def estimate_origin(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the clean-sample prediction from an epsilon prediction."""
        self._validate_timesteps(t)
        alpha_bar_sqrt = _extract(self.alphas_cumprod_sqrt, t, x_t)
        one_minus_alpha_bar_sqrt = _extract(
            self.alphas_one_minus_cumprod_sqrt, t, x_t
        )
        return (x_t - one_minus_alpha_bar_sqrt * noise_pred) / alpha_bar_sqrt

    def posterior_params(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact DDPM posterior mean and standard deviation."""
        self._validate_timesteps(t)
        x_0_pred = self.estimate_origin(x_t, t, noise_pred)
        if self.clip_sample:
            x_0_pred = x_0_pred.clamp(-1, 1)

        mean = (
            _extract(self.posterior_mean_coef_x0, t, x_t) * x_0_pred
            + _extract(self.posterior_mean_coef_xt, t, x_t) * x_t
        )
        std = _extract(self.posterior_std, t, x_t)
        return mean, std

    def _validate_timesteps(self, t: torch.Tensor) -> None:
        if t.ndim != 1:
            raise ValueError("t must be a one-dimensional batch of timesteps")
        if not bool(((t >= 0) & (t < self.train_timesteps)).all()):
            raise ValueError(
                f"DDPM timesteps must be in [0, {self.train_timesteps - 1}]"
            )
