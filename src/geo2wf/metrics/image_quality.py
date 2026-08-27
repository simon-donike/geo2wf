"""Masked image-quality metrics for physical wind-speed reconstructions."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity
import torch


WIND_SPEED_MIN_MS = 0.2
WIND_SPEED_MAX_MS = 80.0
WIND_SPEED_DATA_RANGE_MS = WIND_SPEED_MAX_MS - WIND_SPEED_MIN_MS
SSIM_WINDOW_SIZE = 7


def psnr_db_from_mse(
    mse: float, *, data_range: float = WIND_SPEED_DATA_RANGE_MS
) -> float:
    """Return PSNR in dB for a pooled physical-space mean squared error."""

    if not math.isfinite(mse) or mse < 0.0:
        raise ValueError("mse must be finite and non-negative")
    if not math.isfinite(data_range) or data_range <= 0.0:
        raise ValueError("data_range must be finite and positive")
    return 20.0 * math.log10(data_range / math.sqrt(max(mse, 1.0e-12)))


def masked_ssim_sum_count(
    prediction_ms: torch.Tensor,
    target_ms: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    data_min_ms: float = WIND_SPEED_MIN_MS,
    data_max_ms: float = WIND_SPEED_MAX_MS,
    window_size: int = SSIM_WINDOW_SIZE,
) -> tuple[float, int]:
    """Return the sum and count of scene-mean SSIM scores.

    Prediction and target are clipped to the fixed physical target export range.
    Missing pixels receive the same neutral fill in both images, and SSIM windows
    touching that fill are discarded.  Each scene with at least one complete
    window contributes equally to the returned sum.
    """

    if prediction_ms.ndim != 4 or prediction_ms.shape[1] != 1:
        raise ValueError("prediction_ms must have shape [batch, 1, height, width]")
    if target_ms.shape != prediction_ms.shape:
        raise ValueError("target_ms must match prediction_ms")
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    if valid_mask.shape != prediction_ms.shape:
        raise ValueError("valid_mask must match prediction_ms")
    if not math.isfinite(data_min_ms) or not math.isfinite(data_max_ms):
        raise ValueError("data bounds must be finite")
    data_range = data_max_ms - data_min_ms
    if data_range <= 0.0:
        raise ValueError("data_max_ms must be greater than data_min_ms")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer of at least three")

    prediction = prediction_ms.detach().float().cpu().numpy()
    target = target_ms.detach().float().cpu().numpy()
    valid = valid_mask.detach().bool().cpu().numpy()
    finite = np.isfinite(prediction) & np.isfinite(target)
    valid &= finite
    structure = np.ones((window_size, window_size), dtype=bool)
    scores: list[float] = []
    for index in range(prediction.shape[0]):
        spatial_valid = valid[index, 0]
        complete_window = binary_erosion(
            spatial_valid,
            structure=structure,
            border_value=0,
        )
        if not complete_window.any():
            continue
        predicted_image = np.where(
            spatial_valid,
            np.clip(prediction[index, 0], data_min_ms, data_max_ms),
            data_min_ms,
        )
        target_image = np.where(
            spatial_valid,
            np.clip(target[index, 0], data_min_ms, data_max_ms),
            data_min_ms,
        )
        _, ssim_image = structural_similarity(
            target_image,
            predicted_image,
            data_range=data_range,
            win_size=window_size,
            full=True,
        )
        scores.append(float(np.asarray(ssim_image)[complete_window].mean()))
    return float(sum(scores)), len(scores)


__all__ = [
    "SSIM_WINDOW_SIZE",
    "WIND_SPEED_DATA_RANGE_MS",
    "WIND_SPEED_MAX_MS",
    "WIND_SPEED_MIN_MS",
    "masked_ssim_sum_count",
    "psnr_db_from_mse",
]
