"""Construction helpers for residual diffusion baselines."""

from __future__ import annotations

from typing import Any

from .module import ERA5ResidualDiffusion, load_frozen_deterministic_baseline


def build_residual_diffusion(
    *,
    baseline_source: str = "era5",
    baseline_checkpoint_path: str | None = None,
    **kwargs: Any,
) -> ERA5ResidualDiffusion:
    """Build residual diffusion while keeping checkpoint I/O outside the module."""

    baseline_model = None
    if baseline_source == "deterministic":
        if not baseline_checkpoint_path:
            raise ValueError("deterministic baseline requires baseline_checkpoint_path")
        baseline_model = load_frozen_deterministic_baseline(baseline_checkpoint_path)
    return ERA5ResidualDiffusion(
        baseline_source=baseline_source,
        baseline_model=baseline_model,
        **kwargs,
    )
