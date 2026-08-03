"""Framework-independent evaluation over the public prediction contract."""

from __future__ import annotations

import torch

from geo2wf.models.base import PredictionBatch


def evaluate_prediction(
    prediction: PredictionBatch,
    target_physical: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return shared physical-unit pixel metrics."""

    valid = target_mask.bool() & torch.isfinite(target_physical)
    central = prediction.central_physical
    valid = valid & torch.isfinite(central)
    count = valid.sum().clamp_min(1)
    error = central - target_physical
    return {
        "mae_ms": error.abs().masked_fill(~valid, 0).sum() / count,
        "rmse_ms": (error.square().masked_fill(~valid, 0).sum() / count).sqrt(),
        "bias_ms": error.masked_fill(~valid, 0).sum() / count,
    }
