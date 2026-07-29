from __future__ import annotations

from typing import Any

import torch


def log_wandb_reconstruction(
    module: Any,
    batch: Any,
    prediction_batch: torch.Tensor,
    *,
    wandb_key: str,
    condition_batch: torch.Tensor | None = None,
    target_batch: torch.Tensor | None = None,
    baseline_batch: torch.Tensor | None = None,
    physical_wind_output: bool = False,
) -> None:
    """Log a shared GEO/prediction/target reconstruction figure to W&B."""
    trainer = getattr(module, "_trainer", None)
    if trainer is None or not trainer.is_global_zero:
        return
    logger = module.logger
    if logger is None:
        return

    try:
        import matplotlib.pyplot as plt
        import wandb
        from utils.plotting import plot_validation_reconstruction_batch
    except ImportError:
        return

    if isinstance(batch, dict):
        condition_batch = (
            batch["condition"] if condition_batch is None else condition_batch
        )
        target_batch = batch["target"] if target_batch is None else target_batch
    elif condition_batch is None or target_batch is None:
        raise ValueError(
            "condition_batch and target_batch are required for non-dict batches"
        )

    sample_count = min(int(prediction_batch.shape[0]), 5)
    samples = []
    for index in range(sample_count):
        sample = {
            "condition": condition_batch[index],
            "prediction": prediction_batch[index],
            "target": target_batch[index],
            "physical_wind_output": physical_wind_output,
        }
        if baseline_batch is not None:
            sample["baseline"] = baseline_batch[index]
        if isinstance(batch, dict):
            meta = batch.get("meta", {})
            label = " · ".join(
                value
                for value in (
                    _batch_value(meta.get("storm_id"), index),
                    _batch_value(batch.get("sample_id"), index),
                )
                if value
            )
            sample.update(
                {
                    "condition_mask": _batch_item(batch.get("condition_mask"), index),
                    "target_mask": _batch_item(batch.get("target_mask"), index),
                    "era5_wind_speed_physical": _batch_item(
                        batch.get("era5_wind_speed_physical"), index
                    ),
                    "era5_wind_speed_mask": _batch_item(
                        batch.get("era5_wind_speed_mask"), index
                    ),
                    "baseline_mask": _batch_item(
                        batch.get("_residual_diffusion_baseline_mask"), index
                    ),
                    "condition_channels": _channel_names(
                        meta.get("condition_channels"), index
                    ),
                    "condition_bounds": _batch_item(
                        batch.get("condition_bounds"), index
                    ),
                    "target_bounds": _batch_item(batch.get("target_bounds"), index),
                    "center": _finite_pair(batch.get("center"), index),
                    "sample_label": label,
                }
            )
        samples.append(sample)

    fig = plot_validation_reconstruction_batch(samples)
    # Keep validation media lightweight: cap the longest rendered edge and use
    # JPEG instead of W&B's lossless PNG default.
    max_edge_pixels = 1600
    width_inches, height_inches = fig.get_size_inches()
    max_dpi = max_edge_pixels / max(width_inches, height_inches)
    fig.set_dpi(min(float(fig.dpi), max_dpi))
    try:
        logger.experiment.log(
            {wandb_key: wandb.Image(fig, file_type="jpg")},
            step=module.global_step,
        )
    finally:
        plt.close(fig)


def _batch_item(value: Any, index: int) -> Any:
    if value is None:
        return None
    item = value[index]
    if torch.is_tensor(item):
        item = item.detach().cpu()
    return item


def _batch_value(value: Any, index: int) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return str(value[index]) if index < len(value) else ""
    return str(value)


def _channel_names(value: Any, index: int) -> list[str] | None:
    if not isinstance(value, (list, tuple)):
        return None
    # Default collation transposes each sample's channel list by band.
    return [
        str(item[index] if isinstance(item, (list, tuple)) else item) for item in value
    ]


def _finite_pair(value: Any, index: int) -> tuple[float, float] | None:
    if value is None:
        return None
    pair = value[index].detach().double().cpu()
    if pair.numel() != 2 or not torch.isfinite(pair).all():
        return None
    return float(pair[0]), float(pair[1])
