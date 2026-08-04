"""Weights & Biases media for scalar intensity-correction validation."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .reconstruction_media import _wandb_experiment


def select_intensity_plot_storms(
    rows: Sequence[Mapping[str, Any]],
    count: int = 3,
    preferred_storm_ids: Sequence[str] | None = None,
) -> list[str]:
    """Select stable, information-rich storms for longitudinal validation plots."""

    if count < 1:
        raise ValueError("validation plot storm count must be at least one")
    observation_counts = Counter(str(row["storm_id"]) for row in rows)
    ranked = sorted(
        observation_counts, key=lambda storm: (-observation_counts[storm], storm)
    )
    selected: list[str] = []
    for storm_id in preferred_storm_ids or ():
        normalized = str(storm_id)
        if normalized in observation_counts and normalized not in selected:
            selected.append(normalized)
        if len(selected) == count:
            return selected
    for storm_id in ranked:
        if storm_id not in selected:
            selected.append(storm_id)
        if len(selected) == count:
            break
    return selected


def log_wandb_intensity_evaluation(
    module: Any,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    prefix: str,
    storm_count: int = 3,
    preferred_storm_ids: Sequence[str] | None = None,
) -> None:
    """Log storm tracks, prediction samples, and summary tables to W&B.

    This function only visualizes independently predicted observations. It does
    not feed a time series back into the model.
    """

    trainer = getattr(module, "_trainer", None)
    if (
        trainer is None
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

    selected_storms = select_intensity_plot_storms(
        rows, count=storm_count, preferred_storm_ids=preferred_storm_ids
    )
    if not selected_storms:
        return

    figure, axes = plt.subplots(
        len(selected_storms),
        1,
        figsize=(12, max(3.2 * len(selected_storms), 4.0)),
        squeeze=False,
        sharex=False,
    )
    prediction_table_rows: list[list[Any]] = []
    for axis, storm_id in zip(axes[:, 0], selected_storms):
        storm_rows = sorted(
            (row for row in rows if str(row["storm_id"]) == storm_id),
            key=lambda row: str(row["observation_timestamp"]),
        )
        timestamps = pd.to_datetime(
            [row["observation_timestamp"] for row in storm_rows], utc=True
        ).to_pydatetime()
        target = np.asarray([row["target_ms"] for row in storm_rows], dtype=float)
        prediction = np.asarray(
            [row["prediction_ms"] for row in storm_rows], dtype=float
        )
        raw = np.asarray([row["raw_unet_ms"] for row in storm_rows], dtype=float)
        axis.plot(
            timestamps,
            target,
            color="black",
            marker="o",
            linewidth=2.0,
            label="IBTrACS USA_WIND",
        )
        axis.plot(
            timestamps,
            prediction,
            color="#0072B2",
            marker="o",
            linewidth=2.0,
            label="Corrected prediction",
        )
        axis.plot(
            timestamps,
            raw,
            color="#D55E00",
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
            label="Raw U-Net maximum",
        )
        mae = float(np.mean(np.abs(prediction - target)))
        axis.set_title(f"{storm_id} · {len(storm_rows)} fixes · MAE {mae:.2f} m/s")
        axis.set_ylabel("Maximum wind (m/s)")
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=mdates.UTC)
        )
        for row in storm_rows:
            prediction_table_rows.append(
                [
                    str(row.get("sample_id", "")),
                    storm_id,
                    str(row["observation_timestamp"]),
                    float(row["target_ms"]),
                    float(row["prediction_ms"]),
                    float(row["raw_unet_ms"]),
                    float(
                        row.get(
                            "correction_ms",
                            float(row["prediction_ms"]) - float(row["raw_unet_ms"]),
                        )
                    ),
                    float(row["prediction_ms"]) - float(row["target_ms"]),
                    int(row["target_category"]),
                    int(row["prediction_category"]),
                ]
            )
    axes[0, 0].legend(loc="best", ncol=3)
    axes[-1, 0].set_xlabel("Observation time (UTC)")
    figure.suptitle(
        "Single-timestep intensity predictions across validation storm fixes",
        fontsize=14,
    )
    figure.tight_layout()

    storm_table_rows = []
    for storm_id, metrics in sorted(summary["per_storm"].items()):
        storm_table_rows.append(
            [
                storm_id,
                int(metrics["samples"]),
                float(metrics["mae_ms"]),
                float(metrics["rmse_ms"]),
                float(metrics["bias_ms"]),
                float(metrics["raw_unet_mae_ms"]),
            ]
        )

    category = summary["category"]
    confusion_rows = [
        [observed, *counts]
        for observed, counts in zip(category["labels"], category["confusion_matrix"])
    ]
    payload = {
        f"{prefix}/three_storm_intensity_comparison": wandb.Image(figure),
        f"{prefix}/three_storm_predictions": wandb.Table(
            columns=[
                "sample_id",
                "storm_id",
                "timestamp_utc",
                "ibtracs_wind_ms",
                "predicted_wind_ms",
                "raw_unet_wind_ms",
                "correction_ms",
                "prediction_error_ms",
                "ibtracs_category",
                "predicted_category",
            ],
            data=prediction_table_rows,
        ),
        f"{prefix}/per_storm_metrics": wandb.Table(
            columns=[
                "storm_id",
                "samples",
                "mae_ms",
                "rmse_ms",
                "bias_ms",
                "raw_unet_mae_ms",
            ],
            data=storm_table_rows,
        ),
        f"{prefix}/category_confusion_matrix": wandb.Table(
            columns=["observed\\predicted", *category["labels"]],
            data=confusion_rows,
        ),
    }
    try:
        experiment.log(payload, step=module.global_step)
    finally:
        plt.close(figure)


__all__ = ["log_wandb_intensity_evaluation", "select_intensity_plot_storms"]
