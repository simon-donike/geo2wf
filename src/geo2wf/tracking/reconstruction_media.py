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
    intensity_prediction_batch: torch.Tensor | None = None,
    intensity_target_batch: torch.Tensor | None = None,
    physical_wind_output: bool = False,
    physical_output_units: str | None = None,
) -> None:
    """Log a shared GEO/prediction/target reconstruction figure to W&B."""
    trainer = getattr(module, "_trainer", None)
    if trainer is None or not trainer.is_global_zero:
        return
    experiment = _wandb_experiment(module, trainer)
    if experiment is None:
        return

    try:
        import matplotlib.pyplot as plt
        import wandb
    except ImportError:
        return

    fig = build_reconstruction_figure(
        batch,
        prediction_batch,
        condition_batch=condition_batch,
        target_batch=target_batch,
        baseline_batch=baseline_batch,
        intensity_prediction_batch=intensity_prediction_batch,
        intensity_target_batch=intensity_target_batch,
        physical_wind_output=physical_wind_output,
        physical_output_units=physical_output_units,
    )
    # Keep validation media lightweight: cap the longest rendered edge and use
    # JPEG instead of W&B's lossless PNG default.
    max_edge_pixels = 1600
    width_inches, height_inches = fig.get_size_inches()
    max_dpi = max_edge_pixels / max(width_inches, height_inches)
    fig.set_dpi(min(float(fig.dpi), max_dpi))
    try:
        experiment.log(
            {wandb_key: wandb.Image(fig, file_type="jpg")},
            step=module.global_step,
        )
    finally:
        plt.close(fig)


def build_reconstruction_figure(
    batch: Any,
    prediction_batch: torch.Tensor,
    *,
    condition_batch: torch.Tensor | None = None,
    target_batch: torch.Tensor | None = None,
    baseline_batch: torch.Tensor | None = None,
    intensity_prediction_batch: torch.Tensor | None = None,
    intensity_target_batch: torch.Tensor | None = None,
    physical_wind_output: bool = False,
    physical_output_units: str | None = None,
    max_samples: int = 5,
):
    """Build the same reconstruction figure used by W&B for local artifacts."""
    if max_samples < 1:
        raise ValueError("max_samples must be positive")

    from geo2wf.visualization.wind_fields import plot_validation_reconstruction_batch

    if isinstance(batch, dict):
        condition_batch = (
            batch["condition"] if condition_batch is None else condition_batch
        )
        target_batch = batch["target"] if target_batch is None else target_batch
    elif condition_batch is None or target_batch is None:
        raise ValueError(
            "condition_batch and target_batch are required for non-dict batches"
        )

    sample_count = min(int(prediction_batch.shape[0]), int(max_samples), 5)
    samples = []
    for index in range(sample_count):
        sample = {
            "condition": condition_batch[index],
            "prediction": prediction_batch[index],
            "target": target_batch[index],
            "physical_wind_output": physical_wind_output,
            "physical_output_units": (
                "m s-1" if physical_wind_output else physical_output_units
            ),
        }
        if baseline_batch is not None:
            sample["baseline"] = baseline_batch[index]
        if intensity_prediction_batch is not None:
            sample["intensity_prediction_ms"] = _batch_scalar(
                intensity_prediction_batch, index
            )
        if intensity_target_batch is not None:
            sample["intensity_target_ms"] = _batch_scalar(intensity_target_batch, index)
        if isinstance(batch, dict):
            meta = batch.get("meta", {})
            if isinstance(meta, list):
                meta = meta[index] if index < len(meta) else {}
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
                    "pmw_physical": _batch_item(batch.get("pmw_physical"), index),
                    "pmw_mask": _batch_item(batch.get("pmw_mask"), index),
                    "pmw_bounds": _batch_item(batch.get("pmw_bounds"), index),
                    "pmw_sensor": _batch_value(meta.get("pmw_sensor"), index),
                    "pmw_dt_minutes": _batch_value(meta.get("pmw_dt_minutes"), index),
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
                    "intensity_target_label": {
                        "ibtracs": "IBTrACS max wind",
                        "sar_robust_peak": "SAR robust peak",
                    }.get(
                        _batch_value(batch.get("intensity_target_source"), index),
                        "Scalar intensity",
                    ),
                }
            )
        samples.append(sample)

    return plot_validation_reconstruction_batch(samples)


def _wandb_experiment(module: Any, trainer: Any) -> Any | None:
    """Return the W&B experiment when Lightning has multiple loggers."""
    candidates = getattr(trainer, "loggers", None)
    if not candidates:
        candidates = [getattr(module, "logger", None)]
    for logger in candidates:
        experiment = getattr(logger, "experiment", None)
        if callable(getattr(experiment, "log", None)):
            return experiment
    return None


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
    if torch.is_tensor(value):
        item = value[index] if value.ndim else value
        item = item.detach().cpu()
        return str(item.item()) if item.numel() == 1 else str(item)
    if isinstance(value, (list, tuple)):
        return str(value[index]) if index < len(value) else ""
    return str(value)


def _batch_scalar(value: Any, index: int) -> float:
    item = value[index]
    if torch.is_tensor(item):
        item = item.detach().cpu()
        if item.numel() != 1:
            raise ValueError("Expected one scalar intensity value per sample")
        return float(item.item())
    return float(item)


def _channel_names(value: Any, index: int) -> list[str] | None:
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        return [str(item) for item in value]
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
