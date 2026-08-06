"""Weights & Biases media for recursive scalar intensity forecasts."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .reconstruction_media import _wandb_experiment


def recursive_ri_rows(
    module: Any, cases: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        anchor = torch.tensor([float(case["anchor_wind_ms"])], device=module.device)
        minus_6 = torch.tensor([float(case["wind_minus_6h_ms"])], device=module.device)
        minus_12 = torch.tensor(
            [float(case["wind_minus_12h_ms"])], device=module.device
        )
        plus_6, plus_12 = module.predict_two_steps(anchor, minus_6, minus_12)
        for horizon, prediction, target in (
            (6, plus_6.item(), float(case["target_plus_6h_wind_ms"])),
            (12, plus_12.item(), float(case["target_plus_12h_wind_ms"])),
        ):
            rows.append(
                {
                    "storm_id": str(case["storm_id"]),
                    "sample_id": str(case["sample_id"]),
                    "init_timestamp": str(case["init_timestamp"]),
                    "ri_onset_timestamp": str(case["ri_onset_timestamp"]),
                    "horizon_hours": horizon,
                    "prediction_ms": float(prediction),
                    "target_ms": target,
                    "anchor_ms": float(case["anchor_wind_ms"]),
                    "error_ms": float(prediction - target),
                    "persistence_error_ms": float(case["anchor_wind_ms"] - target),
                }
            )
    return rows


@torch.no_grad()
def log_wandb_ri_forecasts(
    module: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    prefix: str = "val",
) -> None:
    trainer = getattr(module, "_trainer", None)
    if (
        not cases
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

    rows = recursive_ri_rows(module, cases)
    figure, axes = plt.subplots(
        len(cases), 1, figsize=(12, max(3.4 * len(cases), 4.0)), squeeze=False
    )
    for axis, case in zip(axes[:, 0], cases):
        init = pd.Timestamp(case["init_timestamp"])
        onset = pd.Timestamp(case["ri_onset_timestamp"])
        history_times = [init - timedelta(hours=12), init - timedelta(hours=6), init]
        history = [
            float(case["wind_minus_12h_ms"]),
            float(case["wind_minus_6h_ms"]),
            float(case["current_ibtracs_wind_ms"]),
        ]
        storm_rows = [row for row in rows if row["storm_id"] == case["storm_id"]]
        forecast_times = [init] + [
            init + timedelta(hours=int(row["horizon_hours"])) for row in storm_rows
        ]
        forecasts = [float(case["anchor_wind_ms"])] + [
            row["prediction_ms"] for row in storm_rows
        ]
        truth_times = history_times[:2] + forecast_times
        truth = history[:2] + [
            float(case["current_ibtracs_wind_ms"]),
            float(case["target_plus_6h_wind_ms"]),
            float(case["target_plus_12h_wind_ms"]),
        ]
        axis.plot(
            truth_times,
            truth,
            color="black",
            marker="o",
            linewidth=2,
            label="IBTrACS truth/history",
        )
        axis.plot(
            forecast_times,
            forecasts,
            color="#0072B2",
            marker="o",
            linestyle="--",
            linewidth=2,
            label="Recursive forecast",
        )
        axis.plot(
            forecast_times,
            np.full(len(forecast_times), float(case["anchor_wind_ms"])),
            color="#777777",
            linestyle=":",
            linewidth=1.5,
            label="Persistence",
        )
        axis.scatter(
            [init],
            [float(case["anchor_wind_ms"])],
            marker="s",
            s=65,
            color="#D55E00",
            zorder=5,
            label="UNet+MLP anchor",
        )
        axis.axvline(init, color="#555555", linewidth=1, alpha=0.7)
        plot_start = init - timedelta(hours=12)
        plot_end = init + timedelta(hours=12)
        shade_start = max(onset, plot_start)
        if shade_start < plot_end:
            axis.axvspan(
                shade_start,
                plot_end,
                color="#E69F00",
                alpha=0.14,
                label="RI period",
            )
        axis.set_title(f"{case['storm_id']} · initialization {init:%Y-%m-%d %H:%M UTC}")
        axis.set_ylabel("Maximum wind (m/s)")
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d\n%H:%M", tz=mdates.UTC)
        )
    axes[0, 0].legend(loc="best", ncol=3)
    axes[-1, 0].set_xlabel("UTC")
    figure.suptitle("Recursive +6 h / +12 h forecasts around rapid intensification")
    figure.tight_layout()

    table_data = [
        [
            row["storm_id"],
            row["sample_id"],
            row["init_timestamp"],
            row["ri_onset_timestamp"],
            row["horizon_hours"],
            row["prediction_ms"],
            row["target_ms"],
            row["anchor_ms"],
            row["error_ms"],
            row["persistence_error_ms"],
        ]
        for row in rows
    ]
    payload = {
        f"{prefix}/ri_two_step_forecast": wandb.Image(figure),
        f"{prefix}/ri_two_step_forecasts": wandb.Table(
            columns=[
                "storm_id",
                "sample_id",
                "init_timestamp_utc",
                "ri_onset_timestamp_utc",
                "horizon_hours",
                "prediction_ms",
                "target_ms",
                "anchor_ms",
                "error_ms",
                "persistence_error_ms",
            ],
            data=table_data,
        ),
    }
    try:
        experiment.log(payload, step=getattr(module, "global_step", 0))
    finally:
        plt.close(figure)


__all__ = ["log_wandb_ri_forecasts", "recursive_ri_rows"]
