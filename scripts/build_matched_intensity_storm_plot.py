#!/usr/bin/env python3
"""Build full matched-storm intensity trajectories for the results documentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    ROOT
    / "logs"
    / "intensity-target-matrix"
    / "20260824-resume-s42"
    / "run"
    / "matched-validation.json"
)
DEFAULT_IMAGE = (
    ROOT
    / "docs"
    / "assets"
    / "images"
    / "intensity-comparison"
    / "matched-ri-full-storm-trajectories.png"
)
DEFAULT_DATA = (
    ROOT
    / "docs"
    / "assets"
    / "data"
    / "intensity-comparison"
    / "matched-ri-full-storm-trajectories.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--era5", choices=("with_era5", "without_era5"), default="with_era5"
    )
    parser.add_argument(
        "--trained-target",
        choices=("ibtracs", "sar_robust_peak"),
        default="ibtracs",
    )
    parser.add_argument(
        "--model-key",
        choices=("unet_correction", "joint_unet_mlp"),
        default="unet_correction",
    )
    parser.add_argument("--storms", type=int, default=3)
    args = parser.parse_args()
    if args.storms <= 0:
        parser.error("--storms must be positive")
    return args


def _merged_ri_windows(
    timestamps: Iterable[pd.Timestamp],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows = sorted(
        (timestamp - pd.Timedelta(hours=24), timestamp) for timestamp in timestamps
    )
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in windows:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _selected_rows(args: argparse.Namespace) -> pd.DataFrame:
    payload = json.loads(args.result.expanduser().read_text(encoding="utf-8"))
    rows = pd.DataFrame(payload["prediction_rows"])
    selected = rows.loc[
        (rows["era5"] == args.era5)
        & (rows["trained_target"] == args.trained_target)
        & (rows["model_key"] == args.model_key)
    ].copy()
    if selected.empty:
        raise ValueError("the requested prediction series is absent from the report")
    if selected["sample_id"].duplicated().any():
        raise ValueError("selected prediction series contains duplicate sample IDs")
    selected["observation_timestamp"] = pd.to_datetime(
        selected["observation_timestamp"], utc=True
    )
    ranking = (
        selected.groupby("storm_id", as_index=False)
        .agg(
            ri_samples=("is_rapid_intensification", "sum"),
            samples=("sample_id", "size"),
        )
        .sort_values(
            ["ri_samples", "samples", "storm_id"],
            ascending=[False, False, True],
        )
    )
    ranking = ranking.loc[ranking["ri_samples"] > 0].head(args.storms)
    if len(ranking) < args.storms:
        raise ValueError(f"only {len(ranking)} storms contain RI samples")
    selected = selected.loc[selected["storm_id"].isin(ranking["storm_id"])].copy()
    order = {storm_id: index for index, storm_id in enumerate(ranking["storm_id"])}
    selected["storm_order"] = selected["storm_id"].map(order)
    return selected.sort_values(["storm_order", "observation_timestamp"])


def build(args: argparse.Namespace) -> pd.DataFrame:
    selected = _selected_rows(args)
    storms = selected["storm_id"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(storms),
        1,
        figsize=(13, 3.4 * len(storms)),
        squeeze=False,
    )
    figure.subplots_adjust(top=0.88, bottom=0.07, hspace=0.42)
    colors = {
        "ibtracs": "#0072B2",
        "sar": "#D55E00",
        "prediction": "#009E73",
        "ri": "#F0E442",
    }
    for axis, storm_id in zip(axes[:, 0], storms):
        storm = selected.loc[selected["storm_id"] == storm_id].copy()
        times = storm["observation_timestamp"]
        ri_times = storm.loc[
            storm["is_rapid_intensification"].astype(bool),
            "observation_timestamp",
        ]
        for start, end in _merged_ri_windows(ri_times):
            axis.axvspan(start, end, color=colors["ri"], alpha=0.23, linewidth=0)
        axis.plot(
            times,
            storm["ibtracs_target_ms"],
            color=colors["ibtracs"],
            marker="o",
            linewidth=2,
            markersize=5,
            label="Interpolated IBTrACS",
        )
        axis.plot(
            times,
            storm["sar_max_wind_ms"],
            color=colors["sar"],
            marker="^",
            linewidth=1.7,
            markersize=5,
            label="Observed SAR maximum",
        )
        axis.plot(
            times,
            storm["prediction_ms"],
            color=colors["prediction"],
            marker="s",
            linewidth=2,
            markersize=4.5,
            label="Predicted intensity",
        )
        axis.set_title(
            f"{storm_id} · {len(storm)} matched SAR acquisitions · "
            f"{int(storm['is_rapid_intensification'].sum())} RI observations",
            loc="left",
            fontweight="bold",
        )
        axis.set_ylabel("Wind speed (m/s)")
        axis.grid(True, axis="both", alpha=0.22)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
        axis.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(axis.xaxis.get_major_locator())
        )
    axes[-1, 0].set_xlabel("Observation time (UTC)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ri_handle = plt.Rectangle((0, 0), 1, 1, color=colors["ri"], alpha=0.23)
    figure.legend(
        [*handles, ri_handle],
        [*labels, "Preceding 24 h for an RI-classified observation"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Matched full-storm intensity trajectories during rapid intensification",
        fontsize=16,
        fontweight="bold",
        y=0.992,
    )
    image = args.output_image.expanduser().resolve()
    data = args.output_data.expanduser().resolve()
    image.parent.mkdir(parents=True, exist_ok=True)
    data.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(image, dpi=180, bbox_inches="tight")
    plt.close(figure)
    selected.drop(columns=["storm_order"]).to_csv(data, index=False)
    print(f"Wrote {image}")
    print(f"Wrote {data}")
    return selected


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
