"""Reusable mask-aware objective primitives."""

from __future__ import annotations

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)
