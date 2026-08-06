"""Full-storm W&B diagnostics for scalar intensity forecasts."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .reconstruction_media import _wandb_experiment


def log_wandb_full_storm_forecasts(
    module: Any,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    prefix: str = "val",
) -> None:
    """Log every validation fix and a time-series panel for every storm."""
    trainer = getattr(module, "_trainer", None)
    if (
        not rows
        or trainer is None
        or not trainer.is_global_zero
        or getattr(trainer, "sanity_checking", False)
    ):
        return
    experiment = _wandb_experiment(module, trainer)
    if experiment is None:
        return
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import pandas as pd
        import wandb
    except ImportError:
        return

    storm_ids = sorted({str(row["storm_id"]) for row in rows})
    columns = min(4, max(1, len(storm_ids)))
    panel_rows = math.ceil(len(storm_ids) / columns)
    figure, axes = plt.subplots(
        panel_rows,
        columns,
        figsize=(5.0 * columns, 3.2 * panel_rows),
        squeeze=False,
    )
    for axis, storm_id in zip(axes.flat, storm_ids):
        storm_rows = sorted(
            (row for row in rows if str(row["storm_id"]) == storm_id),
            key=lambda row: str(row["init_timestamp"]),
        )
        timestamps = [pd.Timestamp(row["init_timestamp"]) for row in storm_rows]
        series = (
            ("IBTrACS truth", "target_ms", "black", "-", "o"),
            ("Forecast", "prediction_ms", "#0072B2", "--", "o"),
            ("Persistence", "persistence_ms", "#777777", ":", None),
            ("Recent trend", "trend_ms", "#009E73", "-.", None),
        )
        for label, key, color, linestyle, marker in series:
            axis.plot(
                timestamps,
                [float(row[key]) for row in storm_rows],
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.5,
            )
        axis.set_title(storm_id)
        axis.set_ylabel("Maximum wind (m/s)")
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d\n%H:%M", tz=mdates.UTC)
        )
    for axis in axes.flat[len(storm_ids) :]:
        axis.set_visible(False)
    axes.flat[0].legend(loc="best", fontsize="small")
    figure.suptitle("Full matched-validation storm forecasts")
    figure.tight_layout()

    prediction_columns = [
        "storm_id",
        "sample_id",
        "init_timestamp_utc",
        "prediction_ms",
        "target_ms",
        "anchor_ms",
        "predicted_delta_ms",
        "error_ms",
        "persistence_ms",
        "persistence_error_ms",
        "recent_trend_ms",
        "recent_trend_error_ms",
    ]
    prediction_data = [
        [
            str(row["storm_id"]),
            str(row["sample_id"]),
            str(row["init_timestamp"]),
            float(row["prediction_ms"]),
            float(row["target_ms"]),
            float(row["persistence_ms"]),
            float(row["predicted_delta_ms"]),
            float(row["prediction_ms"] - row["target_ms"]),
            float(row["persistence_ms"]),
            float(row["persistence_ms"] - row["target_ms"]),
            float(row["trend_ms"]),
            float(row["trend_ms"] - row["target_ms"]),
        ]
        for row in rows
    ]
    metric_columns = [
        "storm_id",
        "samples",
        "mae_ms",
        "rmse_ms",
        "bias_ms",
        "persistence_mae_ms",
        "recent_trend_mae_ms",
    ]
    metric_data = [
        [
            storm_id,
            values["samples"],
            values["mae_ms"],
            values["rmse_ms"],
            values["bias_ms"],
            values["persistence_mae_ms"],
            values["trend_mae_ms"],
        ]
        for storm_id, values in sorted(summary["per_storm"].items())
    ]
    payload = {
        f"{prefix}/full_storm_forecasts": wandb.Image(figure),
        f"{prefix}/full_storm_predictions": wandb.Table(
            columns=prediction_columns, data=prediction_data
        ),
        f"{prefix}/per_storm_metrics": wandb.Table(
            columns=metric_columns, data=metric_data
        ),
    }
    try:
        experiment.log(payload, step=getattr(module, "global_step", 0))
    finally:
        plt.close(figure)


__all__ = ["log_wandb_full_storm_forecasts"]
