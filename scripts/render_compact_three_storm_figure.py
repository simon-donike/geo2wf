#!/usr/bin/env python3
"""Render a space-saving, paper-facing three-storm nowcast figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
STORMS = (
    ("AL082025", "Humberto"),
    ("EP112025", "Kiko"),
    ("EP182023", "Otis"),
)
REGIME = "without_era5"
MODEL_KEY = "latent_sar_max_wind_radii"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "docs/assets/data/final-results/current-three-storm-predictions.csv",
    )
    parser.add_argument("--smoothing-hours", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "docs/assets/images/final-results/current-three-storm-compact-nowcasts",
        help="Output path without an extension; both PDF and PNG are written.",
    )
    return parser.parse_args()


def _series(
    frame: pd.DataFrame,
    storm_id: str,
    key: str,
    smoothing_hours: int,
) -> pd.DataFrame:
    if key == "ibtracs_best_track":
        mask = (frame["model_key"] == MODEL_KEY) & (
            frame["conditioning"] == "without_era5"
        )
        value_column = "target_ms"
    elif key == "era5_max_wind":
        mask = frame["model_key"] == "era5_max_wind"
        value_column = "prediction_ms"
    elif key == "model_without_era5":
        mask = (frame["model_key"] == MODEL_KEY) & (
            frame["conditioning"] == "without_era5"
        )
        value_column = "prediction_ms"
    elif key == "unet_without_era5":
        mask = (frame["model_key"] == "unet_raw") & (
            frame["conditioning"] == "without_era5"
        )
        value_column = "prediction_ms"
    else:
        raise ValueError(f"unknown series key {key!r}")

    selected = frame.loc[
        (frame["storm_id"] == storm_id) & mask,
        ["observation_timestamp", value_column, "is_rapid_intensification"],
    ].copy()
    if selected.empty:
        raise ValueError(f"missing {key!r} values for {storm_id}")
    selected = selected.rename(columns={value_column: "maximum_wind_ms"})
    selected = selected.sort_values("observation_timestamp")
    hourly = (
        selected.set_index("observation_timestamp")["maximum_wind_ms"]
        .resample("1h")
        .mean()
    )
    smoothed = hourly.rolling(
        smoothing_hours, center=True, min_periods=1
    ).mean()
    output = smoothed.dropna().rename("maximum_wind_ms").reset_index()
    return output


def _ri_spans(truth: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    flags = truth.set_index("observation_timestamp")["is_rapid_intensification"]
    flags = flags.astype(bool).resample("1h").max().fillna(False)
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start: pd.Timestamp | None = None
    for timestamp, active in flags.items():
        if active and start is None:
            start = timestamp
        elif not active and start is not None:
            spans.append((start, timestamp))
            start = None
    if start is not None:
        spans.append((start, flags.index[-1] + pd.Timedelta(hours=1)))
    return spans


def main() -> None:
    args = parse_args()
    if args.smoothing_hours < 1:
        raise ValueError("smoothing-hours must be positive")
    frame = pd.read_csv(args.input)
    frame["observation_timestamp"] = pd.to_datetime(
        frame["observation_timestamp"], utc=True
    )

    colors = {
        "ibtracs_best_track": "#111111",
        "model_without_era5": "#0072B2",
        "unet_without_era5": "#D55E00",
        "era5_max_wind": "#777777",
    }
    labels = {
        "ibtracs_best_track": "IBTrACS",
        "model_without_era5": "Latent MLP",
        "unet_without_era5": "U-Net",
        "era5_max_wind": "ERA5",
    }
    styles = {
        "ibtracs_best_track": {"linewidth": 1.65, "zorder": 5},
        "model_without_era5": {"linewidth": 1.35, "zorder": 4},
        "unet_without_era5": {"linewidth": 1.25, "zorder": 4},
        "era5_max_wind": {
            "linewidth": 1.05,
            "linestyle": (0, (4, 2)),
            "zorder": 3,
        },
    }

    with plt.rc_context(
        {
            "font.size": 8.8,
            "axes.titlesize": 9.6,
            "axes.labelsize": 9.3,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.4,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(
            1,
            len(STORMS),
            figsize=(7.05, 1.95),
            sharey=True,
            squeeze=False,
        )
        legend: dict[str, object] = {}
        ri_handle = None
        for panel, (storm_id, storm_name) in enumerate(STORMS):
            axis = axes[0, panel]
            storm_rows = frame.loc[
                (frame["storm_id"] == storm_id)
                & (frame["model_key"] == MODEL_KEY)
                & (frame["conditioning"] == REGIME)
            ].copy()
            for start, end in _ri_spans(storm_rows):
                ri_handle = axis.axvspan(
                    start,
                    end,
                    color="#F4A3A8",
                    alpha=0.34,
                    linewidth=0,
                    zorder=0,
                )
            for key in (
                "ibtracs_best_track",
                "model_without_era5",
                "unet_without_era5",
                "era5_max_wind",
            ):
                values = _series(frame, storm_id, key, args.smoothing_hours)
                line = axis.plot(
                    values["observation_timestamp"],
                    values["maximum_wind_ms"],
                    color=colors[key],
                    label=labels[key],
                    **styles[key],
                )[0]
                legend.setdefault(key, line)

            axis.set_title(f"({chr(97 + panel)}) {storm_name}", pad=2.5)
            axis.set_ylim(0.0, 80.0)
            axis.set_yticks((0, 20, 40, 60, 80))
            axis.grid(color="#D0D0D0", alpha=0.42, linewidth=0.45)
            axis.xaxis.set_major_locator(
                mdates.AutoDateLocator(minticks=2, maxticks=4)
            )
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            axis.tick_params(axis="both", length=2.5, pad=1.8)
        axes[0, 0].set_ylabel(r"$V_{\max}$ (m s$^{-1}$)", labelpad=2.5)

        series_order = (
            "ibtracs_best_track",
            "model_without_era5",
            "unet_without_era5",
            "era5_max_wind",
        )
        handles = [legend[key] for key in series_order]
        legend_labels = [labels[key] for key in series_order]
        if ri_handle is not None:
            handles.append(ri_handle)
            legend_labels.append("RI")
        figure.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=5,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.15,
        )
        figure.text(
            0.5,
            0.035,
            "Latent MLP and U-Net are both evaluated without ERA5 inputs.",
            ha="center",
            va="bottom",
            fontsize=8.4,
        )
        figure.subplots_adjust(
            left=0.065, right=0.995, bottom=0.25, top=0.79, wspace=0.10
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            args.output.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=0.015,
            facecolor="white",
        )
        figure.savefig(
            args.output.with_suffix(".png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.015,
            facecolor="white",
        )
        plt.close(figure)


if __name__ == "__main__":
    main()
