#!/usr/bin/env python3
"""Build the publishable ERA5 intensity comparison report and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUNS = {
    "With ERA5": {
        "root": ROOT / "logs/intensity-comparisons/20260820T144011Z-with-era5",
        "storm": ROOT
        / "logs/intensity-comparisons/three-storm-inference/with-era5.csv",
        "stages": {
            "Raw U-Net": (
                "unet-runs/20260820-164014_modular/metrics/metrics.csv",
                "val/eye_structure_score",
                61,
                "4rqrc3oh",
            ),
            "Separate correction": (
                "correction-runs/20260820-171845_modular/metrics/metrics.csv",
                "val/storm_macro_mae_ms",
                27,
                "ymivzoau",
            ),
            "Joint U-Net + MLP": (
                "joint-runs/20260820-164014_modular/metrics/metrics.csv",
                "val/loss",
                139,
                "rj7951rk",
            ),
        },
    },
    "Without ERA5": {
        "root": ROOT / "logs/intensity-comparisons/20260820T155344Z-without-era5",
        "storm": ROOT
        / "logs/intensity-comparisons/three-storm-inference/without-era5.csv",
        "stages": {
            "Raw U-Net": (
                "unet-runs/20260820-175347_modular/metrics/metrics.csv",
                "val/eye_structure_score",
                77,
                "ldd7fp28",
            ),
            "Separate correction": (
                "correction-runs/20260820-183313_modular/metrics/metrics.csv",
                "val/storm_macro_mae_ms",
                50,
                "frrrn6dl",
            ),
            "Joint U-Net + MLP": (
                "joint-runs/20260820-175347_modular/metrics/metrics.csv",
                "val/loss",
                70,
                "oyiqs6go",
            ),
        },
    },
}
MODEL_ORDER = [
    "U-Net raw field maximum",
    "U-Net + correction",
    "Joint U-Net + MLP",
]
STORM_ORDER = ["AL082025", "EP112025", "EP182023"]
STORM_NAMES = {
    "AL082025": "Humberto (AL082025)",
    "EP112025": "Kiko (EP112025)",
    "EP182023": "Otis (EP182023)",
}
PREDICTION_COLUMNS = {
    "U-Net raw field maximum": "unet_raw_max_ms",
    "U-Net + correction": "unet_correction_ms",
    "Joint U-Net + MLP": "joint_unet_mlp_ms",
}
COLORS = {
    "U-Net raw field maximum": "#E69F00",
    "U-Net + correction": "#0072B2",
    "Joint U-Net + MLP": "#009E73",
    "IBTrACS": "#111111",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/experiments/intensity-comparison-results.md",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=ROOT / "docs/assets/images/intensity-comparison",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "docs/assets/data/intensity-comparison",
    )
    return parser.parse_args()


def _require_inputs() -> None:
    missing = []
    for values in RUNS.values():
        root = Path(values["root"])
        missing.extend(
            path
            for path in [root / "val-comparison.csv", Path(values["storm"])]
            if not path.is_file()
        )
        for relative, _, _, _ in values["stages"].values():
            path = root / relative
            if not path.is_file():
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            "report inputs are missing: " + ", ".join(map(str, missing))
        )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _validation_frames() -> dict[str, pd.DataFrame]:
    return {
        regime: pd.read_csv(Path(values["root"]) / "val-comparison.csv")
        for regime, values in RUNS.items()
    }


def _plot_validation_mae(frames: Mapping[str, pd.DataFrame], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    regime_colors = ["#4477AA", "#CC6677"]
    for index, (regime, frame) in enumerate(frames.items()):
        selected = frame.set_index("model").loc[MODEL_ORDER]
        center = selected["intensity_mae_ms"].to_numpy(float)
        lower = center - selected["intensity_mae_95ci_low_ms"].to_numpy(float)
        upper = selected["intensity_mae_95ci_high_ms"].to_numpy(float) - center
        positions = x + (index - 0.5) * width
        bars = axis.bar(
            positions,
            center,
            width,
            label=regime,
            color=regime_colors[index],
            yerr=np.vstack([lower, upper]),
            capsize=4,
            error_kw={"linewidth": 1.2},
        )
        axis.bar_label(bars, fmt="%.2f", padding=5, fontsize=9)
    axis.set_xticks(
        x, ["Raw field\nmaximum", "Separate\ncorrection", "Joint\nU-Net + MLP"]
    )
    axis.set_ylabel("Validation intensity MAE (m/s)")
    axis.set_title("Matched 232-sample validation cohort")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    axis.set_ylim(0, 11.5)
    figure.tight_layout()
    _save_figure(figure, path)


def _validation_series(path: Path, metric: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame.loc[
        pd.to_numeric(frame.get("epoch"), errors="coerce").notna()
        & pd.to_numeric(frame.get(metric), errors="coerce").notna(),
        ["epoch", metric],
    ].copy()
    selected["epoch"] = pd.to_numeric(selected["epoch"])
    selected[metric] = pd.to_numeric(selected[metric])
    return selected.groupby("epoch", as_index=False)[metric].last().sort_values("epoch")


def _plot_training_curves(path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=False)
    for row, (regime, values) in enumerate(RUNS.items()):
        root = Path(values["root"])
        for column, (stage, stage_values) in enumerate(values["stages"].items()):
            relative, metric, best_epoch, _ = stage_values
            series = _validation_series(root / relative, metric)
            axis = axes[row, column]
            axis.plot(series["epoch"], series[metric], color="#0072B2", linewidth=1.5)
            axis.axvline(
                best_epoch,
                color="#D55E00",
                linestyle="--",
                linewidth=1.2,
                label=f"selected epoch {best_epoch}",
            )
            axis.set_title(f"{regime}: {stage}", fontsize=10)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(metric.removeprefix("val/").replace("_", " "))
            axis.grid(alpha=0.22)
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Validation monitors used for checkpoint selection (early-stopping patience = 50)",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(figure, path)


def _storm_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for regime, values in RUNS.items():
        frame = pd.read_csv(Path(values["storm"]))
        frame["observation_timestamp"] = pd.to_datetime(
            frame["observation_timestamp"], utc=True
        )
        frames[regime] = frame.sort_values(["storm_id", "observation_timestamp"])
    reference = frames["With ERA5"]
    candidate = frames["Without ERA5"]
    keys = [
        "observation_id",
        "storm_id",
        "observation_timestamp",
        "target_ms",
        "inference_valid",
    ]
    if (
        not reference[keys]
        .reset_index(drop=True)
        .equals(candidate[keys].reset_index(drop=True))
    ):
        raise ValueError("ERA5 and no-ERA5 dense inference cohorts differ")
    return frames


def _hourly(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = ["target_ms", *PREDICTION_COLUMNS.values()]
    return (
        frame.set_index("observation_timestamp")[numeric]
        .resample("1h")
        .mean()
        .dropna(how="all")
    )


def _plot_storm_trajectories(frames: Mapping[str, pd.DataFrame], path: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15, 11), squeeze=False)
    for row, storm_id in enumerate(STORM_ORDER):
        for column, (regime, frame) in enumerate(frames.items()):
            axis = axes[row, column]
            storm = _hourly(frame.loc[frame["storm_id"] == storm_id])
            axis.plot(
                storm.index,
                storm["target_ms"],
                color=COLORS["IBTrACS"],
                linewidth=2.6,
                label="IBTrACS ground truth",
                zorder=5,
            )
            for model, prediction_column in PREDICTION_COLUMNS.items():
                axis.plot(
                    storm.index,
                    storm[prediction_column],
                    color=COLORS[model],
                    linewidth=1.5,
                    label=model,
                    alpha=0.92,
                )
            axis.set_title(f"{STORM_NAMES[storm_id]} — {regime}")
            axis.set_ylabel("Maximum wind (m/s)")
            axis.grid(alpha=0.22)
            locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("Dense storm intensity trajectories (hourly means for readability)")
    figure.tight_layout(rect=(0, 0.055, 1, 0.965))
    _save_figure(figure, path)


def _dense_metrics(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for regime, frame in frames.items():
        for model, column in PREDICTION_COLUMNS.items():
            valid = np.isfinite(frame[column].to_numpy(float)) & np.isfinite(
                frame["target_ms"].to_numpy(float)
            )
            evaluated = frame.loc[valid]
            error = evaluated[column].to_numpy(float) - evaluated["target_ms"].to_numpy(
                float
            )
            row: dict[str, Any] = {
                "Conditioning": regime,
                "Model": model,
                "Attempted": len(frame),
                "Samples": len(evaluated),
                "MAE (m/s)": np.mean(np.abs(error)),
                "RMSE (m/s)": np.sqrt(np.mean(np.square(error))),
                "Bias (m/s)": np.mean(error),
            }
            for storm_id in STORM_ORDER:
                storm = evaluated.loc[evaluated["storm_id"] == storm_id]
                row[f"{STORM_NAMES[storm_id]} MAE (m/s)"] = np.mean(
                    np.abs(
                        storm[column].to_numpy(float)
                        - storm["target_ms"].to_numpy(float)
                    )
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_storm_mae(metrics: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    width = 0.24
    x = np.arange(len(STORM_ORDER))
    for axis, regime in zip(axes, RUNS):
        selected = metrics.loc[metrics["Conditioning"] == regime].set_index("Model")
        for index, model in enumerate(MODEL_ORDER):
            values = [
                selected.loc[model, f"{STORM_NAMES[storm]} MAE (m/s)"]
                for storm in STORM_ORDER
            ]
            positions = x + (index - 1) * width
            bars = axis.bar(
                positions,
                values,
                width,
                color=COLORS[model],
                label=model,
            )
            axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        axis.set_xticks(x, ["Humberto", "Kiko", "Otis"])
        axis.set_title(regime)
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Dense case-study MAE (m/s)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Error by validation storm (all 10-minute GEO observations)")
    figure.tight_layout(rect=(0, 0.1, 1, 0.95))
    _save_figure(figure, path)


def _format_number(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _validation_table(frame: pd.DataFrame) -> list[str]:
    header = (
        "| Model | n | MAE (95% CI), m/s | Δ MAE vs raw, m/s | RMSE, m/s | "
        "Bias, m/s | Storm-macro MAE, m/s | Exact category | Macro F1 | "
        "Within one | Field MAE / RMSE / bias, m/s |"
    )
    lines = [
        header,
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        row = frame.loc[frame["model"] == model].iloc[0]
        ci = (
            f"{_format_number(row['intensity_mae_ms'])} "
            f"({_format_number(row['intensity_mae_95ci_low_ms'])}–"
            f"{_format_number(row['intensity_mae_95ci_high_ms'])})"
        )
        delta = _format_number(row["intensity_mae_delta_vs_unet_raw_max_ms"])
        if model != MODEL_ORDER[0]:
            delta += (
                f" ({_format_number(row['intensity_mae_delta_95ci_low_ms'])}–"
                f"{_format_number(row['intensity_mae_delta_95ci_high_ms'])})"
            )
        field = " / ".join(
            _format_number(row[column])
            for column in ("field_mae_ms", "field_rmse_ms", "field_bias_ms")
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    str(int(row["samples"])),
                    ci,
                    delta,
                    _format_number(row["intensity_rmse_ms"]),
                    _format_number(row["intensity_bias_ms"]),
                    _format_number(row["storm_macro_mae_ms"]),
                    _format_number(row["category_accuracy"]),
                    _format_number(row["category_macro_f1"]),
                    _format_number(row["within_one_category_accuracy"]),
                    field,
                ]
            )
            + " |"
        )
    return lines


def _dense_table(metrics: pd.DataFrame) -> list[str]:
    lines = [
        "| Conditioning | Model | valid n / attempted | MAE | RMSE | Bias | Humberto MAE | Kiko MAE | Otis MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Conditioning"]),
                    str(row["Model"]),
                    f"{int(row['Samples'])} / {int(row['Attempted'])}",
                    _format_number(row["MAE (m/s)"]),
                    _format_number(row["RMSE (m/s)"]),
                    _format_number(row["Bias (m/s)"]),
                    _format_number(row["Humberto (AL082025) MAE (m/s)"]),
                    _format_number(row["Kiko (EP112025) MAE (m/s)"]),
                    _format_number(row["Otis (EP182023) MAE (m/s)"]),
                ]
            )
            + " |"
        )
    return lines


def _publish_data(
    validation: Mapping[str, pd.DataFrame],
    storms: Mapping[str, pd.DataFrame],
    dense_metrics: pd.DataFrame,
    data_root: Path,
) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    validation_rows = []
    for regime, frame in validation.items():
        copy = frame.copy()
        copy.insert(0, "conditioning", regime)
        validation_rows.append(copy)
    pd.concat(validation_rows, ignore_index=True).to_csv(
        data_root / "validation-results.csv", index=False
    )

    with_era5 = storms["With ERA5"].copy()
    without_era5 = storms["Without ERA5"].copy()
    shared = [
        "observation_id",
        "storm_id",
        "storm_name",
        "source_split",
        "observation_timestamp",
        "target_ms",
        "target_category",
        "inference_valid",
        "inference_issue",
    ]
    combined = with_era5[shared + list(PREDICTION_COLUMNS.values())].rename(
        columns={value: f"with_era5_{value}" for value in PREDICTION_COLUMNS.values()}
    )
    no_era_predictions = without_era5[
        ["observation_id", *PREDICTION_COLUMNS.values()]
    ].rename(
        columns={
            value: f"without_era5_{value}" for value in PREDICTION_COLUMNS.values()
        }
    )
    combined = combined.merge(
        no_era_predictions, on="observation_id", validate="one_to_one"
    )
    combined.to_csv(data_root / "three-storm-inference.csv", index=False)
    dense_metrics.to_csv(data_root / "three-storm-metrics.csv", index=False)
    summary = {
        "samples": len(combined),
        "storms": STORM_ORDER,
        "source_split": "val",
        "metrics": dense_metrics.to_dict(orient="records"),
    }
    (data_root / "three-storm-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def _copy_wandb_media(image_root: Path) -> None:
    sources = {
        "wandb-correction-with-era5-best.png": ROOT
        / "logs/intensity-comparisons/20260820T144011Z-with-era5/correction-runs/20260820-171845_modular/wandb/run-20260820_171845-ymivzoau/files/media/images/val/three_storm_intensity_comparison_1492_faec1ed028410c889886.png",
        "wandb-correction-without-era5-best.png": ROOT
        / "logs/intensity-comparisons/20260820T155344Z-without-era5/correction-runs/20260820-183313_modular/wandb/run-20260820_183313-frrrn6dl/files/media/images/val/three_storm_intensity_comparison_2711_88341a5824285eb715f7.png",
        "wandb-joint-with-era5-best.jpg": ROOT
        / "logs/intensity-comparisons/20260820T144011Z-with-era5/joint-runs/20260820-164014_modular/wandb/run-20260820_164017-rj7951rk/files/media/images/images/val_reconstruction_29564_21676dd0146422d0a031.jpg",
        "wandb-joint-without-era5-best.jpg": ROOT
        / "logs/intensity-comparisons/20260820T155344Z-without-era5/joint-runs/20260820-175347_modular/wandb/run-20260820_175350-oyiqs6go/files/media/images/images/val_reconstruction_15005_94f7853d591db892be96.jpg",
    }
    image_root.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, image_root / name)


def _report_markdown(
    validation: Mapping[str, pd.DataFrame],
    storm_frames: Mapping[str, pd.DataFrame],
    dense: pd.DataFrame,
) -> str:
    era = validation["With ERA5"].set_index("model")
    no_era = validation["Without ERA5"].set_index("model")
    joint_era = era.loc["Joint U-Net + MLP", "intensity_mae_ms"]
    correction_era = era.loc["U-Net + correction", "intensity_mae_ms"]
    raw_era = era.loc["U-Net raw field maximum", "intensity_mae_ms"]
    joint_no = no_era.loc["Joint U-Net + MLP", "intensity_mae_ms"]
    dense_attempted = int(dense["Attempted"].max())
    dense_valid = int(dense["Samples"].min())
    invalid = dense_attempted - dense_valid
    invalid_label = "scan" if invalid == 1 else "scans"
    storm_valid = (
        storm_frames["With ERA5"]
        .groupby("storm_id")["inference_valid"]
        .sum()
        .astype(int)
        .to_dict()
    )
    dense_best = {
        regime: rows.loc[rows["MAE (m/s)"].idxmin()]
        for regime, rows in dense.groupby("Conditioning")
    }
    lines = [
        "# Intensity reconstruction benchmark",
        "",
        "This report compares the raw U-Net field maximum, a separately trained "
        "U-Net plus correction network, and the jointly trained U-Net+MLP. Every "
        "model is evaluated once with ERA5 conditioning and once without it.",
        "",
        '!!! warning "Validation results, not a final test claim"',
        "    The matched benchmark and the Humberto, Kiko, and Otis trajectories "
        "are all from the `val` split. None of these storm IDs occurs in training, "
        "but validation metrics were used for early stopping and checkpoint "
        "selection. These are model-selection diagnostics, not an "
        "unbiased held-out test estimate.",
        "",
        "## Main result",
        "",
        f"On the matched **232-observation, 34-storm** validation cohort, the joint "
        f"model has the lowest intensity MAE: **{joint_era:.3f} m/s with ERA5** and "
        f"**{joint_no:.3f} m/s without ERA5**. With ERA5, the separate correction "
        f"reaches **{correction_era:.3f} m/s**, versus **{raw_era:.3f} m/s** for the "
        "raw field maximum. The storm-bootstrap intervals for both learned scalar "
        "heads' improvement over the raw maximum exclude zero in both regimes.",
        "",
        "![Validation intensity MAE comparison](../assets/images/intensity-comparison/validation-intensity-mae.png)",
        "",
        "[Download validation results as CSV](../assets/data/intensity-comparison/validation-results.csv){ .md-button }",
        "",
        "## Matched validation tables",
        "",
        "### With ERA5",
        "",
        *_validation_table(validation["With ERA5"]),
        "",
        "### Without ERA5",
        "",
        *_validation_table(validation["Without ERA5"]),
        "",
        "The two tables use identical sample IDs. ERA5 therefore changes only the "
        "conditioning available to the models, not the evaluation cohort.",
        "",
        "## What each metric measures",
        "",
        "Let the scalar error be `prediction − IBTrACS target` for one observation.",
        "",
        "| Metric | Calculation and interpretation |",
        "|---|---|",
        "| **Intensity MAE** | Mean absolute scalar error. It describes the typical magnitude of an intensity miss in m/s; lower is better. The range is a 95% paired cluster-bootstrap interval over storms. |",
        "| **Δ MAE vs raw** | Candidate MAE minus raw-U-Net MAE on the same storm resample. Negative favors the candidate. An interval below zero means the improvement is consistent across the storm bootstrap. |",
        "| **Intensity RMSE** | Square root of mean squared scalar error. Large misses receive extra weight; lower is better. |",
        "| **Intensity bias** | Mean signed scalar error. Negative is systematic underprediction, positive is overprediction, and zero is ideal. Positive and negative errors can cancel. |",
        "| **Storm-macro MAE** | MAE is computed within each storm and then averaged with equal weight per storm. It prevents storms with many images from dominating. |",
        "| **Exact category accuracy** | Fraction assigned exactly the correct TD, TS, or Saffir–Simpson category; higher is better. |",
        "| **Category macro F1** | Per-category harmonic mean of precision and recall, averaged equally across represented categories; higher is better. |",
        "| **Within one** | Fraction no more than one category away from the target; higher is better. |",
        "| **Field MAE / RMSE / bias** | Pixel-pooled U-Net-minus-SAR errors over the common finite valid mask. These diagnose the reconstructed wind field, not scalar intensity. They do not apply to the separate correction head, which emits only a scalar. |",
        "",
        "IBTrACS `USA_WIND` is converted from knots with `1 kt = 0.514444 m/s` "
        "and linearly interpolated to the image timestamp only when the enclosing "
        "fixes are at most three hours apart. Categories use the unrounded "
        "thresholds: TD `<34 kt`, TS `34–<64`, C1 `64–<83`, C2 `83–<96`, C3 "
        "`96–<113`, C4 `113–<137`, and C5 `≥137 kt`.",
        "",
        "The 95% intervals use 2,000 paired cluster-bootstrap repetitions over "
        "storm IDs (seed 42). Every resample evaluates all models on the same "
        "storms and retains every observation from each sampled storm.",
        "",
        "## Training, W&B, and early stopping",
        "",
        "All six runs logged metrics to Weights & Biases. Training allowed up to "
        "1,000 epochs but stopped after **50 validation epochs without improvement**. "
        "The vertical dashed lines below mark the checkpoint selected by each "
        "stage-specific validation monitor.",
        "",
        "![Validation monitor histories](../assets/images/intensity-comparison/training-validation-curves.png)",
        "",
        "| Conditioning | Raw U-Net | Separate correction | Joint U-Net + MLP |",
        "|---|---|---|---|",
        "| With ERA5 | epoch 61 · `4rqrc3oh` | epoch 27 · `ymivzoau` | epoch 139 · `rj7951rk` |",
        "| Without ERA5 | epoch 77 · `ldd7fp28` | epoch 50 · `frrrn6dl` | epoch 70 · `oyiqs6go` |",
        "",
        "### W&B validation media at the selected checkpoints",
        "",
        "The correction images are the W&B three-storm diagnostic nearest each "
        "selected checkpoint. They show the automatically selected validation "
        "storms (not the dedicated three-storm dense analysis below). The joint "
        "images show validation GEO input, predicted and SAR target fields, valid "
        "footprints, ERA5 where applicable, and scalar-intensity error.",
        "",
        '=== "Correction · with ERA5"',
        "",
        "    ![W&B correction validation media with ERA5](../assets/images/intensity-comparison/wandb-correction-with-era5-best.png)",
        "",
        '=== "Correction · without ERA5"',
        "",
        "    ![W&B correction validation media without ERA5](../assets/images/intensity-comparison/wandb-correction-without-era5-best.png)",
        "",
        '=== "Joint · with ERA5"',
        "",
        "    ![W&B joint validation reconstruction with ERA5](../assets/images/intensity-comparison/wandb-joint-with-era5-best.jpg)",
        "",
        '=== "Joint · without ERA5"',
        "",
        "    ![W&B joint validation reconstruction without ERA5](../assets/images/intensity-comparison/wandb-joint-without-era5-best.jpg)",
        "",
        "## Humberto, Kiko, and Otis: dense full-storm inference",
        "",
        "The inference manifest contributes every listed 10-minute GEO image: "
        "**1,006 Humberto observations (`AL082025`)**, **1,578 Kiko observations "
        "(`EP112025`)**, and **684 Otis observations (`EP182023`)**, for **3,268** "
        "timestamps. All 3,268 have a valid three-hour-or-narrower IBTrACS bracket. "
        "Both conditioning regimes use exactly the same observation IDs, centers, "
        "timestamps, and ground truth.",
        "",
        f"Inference was attempted for all **{dense_attempted:,}** scans. "
        f"**{dense_valid:,}** have a non-empty valid footprint after the model's "
        f"center crop and are scored; **{invalid:,} {invalid_label}** are retained in the "
        "download with `inference_valid = false` and excluded identically from "
        "both regimes and every model metric.",
        "",
        f"Across the dense common cohort, **{dense_best['With ERA5']['Model']}** "
        f"has the lowest aggregate MAE with ERA5 "
        f"(**{dense_best['With ERA5']['MAE (m/s)']:.3f} m/s**), while "
        f"**{dense_best['Without ERA5']['Model']}** is lowest without ERA5 "
        f"(**{dense_best['Without ERA5']['MAE (m/s)']:.3f} m/s**). Per-storm "
        "behavior differs; the trajectory and storm-level table report that "
        "variation.",
        "",
        "The plotted curves are hourly means to keep the dense 10-minute "
        "series readable. The table scores every valid individual observation, "
        "while the download also retains any explicitly flagged unusable scan.",
        "",
        "![Predicted and ground-truth full-storm intensity trajectories](../assets/images/intensity-comparison/three-storm-intensity-trajectories.png)",
        "",
        "![Per-storm dense inference MAE](../assets/images/intensity-comparison/three-storm-mae.png)",
        "",
        "All values below are m/s except the sample count.",
        "",
        *_dense_table(dense),
        "",
        "[Download all six dense prediction series](../assets/data/intensity-comparison/three-storm-inference.csv){ .md-button } "
        "[Download dense metrics](../assets/data/intensity-comparison/three-storm-metrics.csv){ .md-button } "
        "[Download JSON summary](../assets/data/intensity-comparison/three-storm-summary.json){ .md-button }",
        "",
        "### Split audit",
        "",
        "| Storm | Source split | Paired train samples | Paired validation samples | Paired test samples | Dense valid / attempted |",
        "|---|---|---:|---:|---:|---:|",
        f"| Humberto (`AL082025`) | `val` | 0 | 18 | 0 | {storm_valid['AL082025']:,} / 1,006 |",
        f"| Kiko (`EP112025`) | `val` | 0 | 23 | 0 | {storm_valid['EP112025']:,} / 1,578 |",
        f"| Otis (`EP182023`) | `val` | 0 | 2 | 0 | {storm_valid['EP182023']:,} / 684 |",
        "",
        "The split audit confirms that **none of the three storms is in "
        "training**. They are validation storms, including Otis; none is in the "
        "test split. Because validation guided early stopping, the dense plots are "
        "diagnostic case studies rather than independent test cases.",
        "",
        "## Reproduce the dense inference",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_intensity_comparison_storm_inference.py --era5 with --device cuda",
        "CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_intensity_comparison_storm_inference.py --era5 without --device cuda",
        ".venv/bin/python scripts/build_intensity_comparison_web_report.py",
        "```",
        "",
        "Each inference JSON records SHA-256 hashes for the raw U-Net, correction, "
        "and joint checkpoints. The correction run additionally verifies that its "
        "frozen-field cache was generated by the exact selected raw U-Net checkpoint.",
        "",
        "## Limitations",
        "",
        "- Validation-guided selection makes all reported results model-selection diagnostics.",
        "- Dense 10-minute observations are strongly temporally correlated; 3,268 rows are not 3,268 independent storms or trials.",
        "- IBTrACS is a best-track estimate interpolated in time, not a direct measurement at each satellite scan.",
        "- The raw scalar is the largest valid pixel in a reconstructed field, while the learned heads estimate IBTrACS maximum wind directly; these are related but not identical physical quantities.",
        "- Final performance estimation requires a locked, storm-disjoint test set after architecture and checkpoint selection.",
        "",
    ]
    return "\n".join(lines)


def build(args: argparse.Namespace) -> None:
    _require_inputs()
    validation = _validation_frames()
    storms = _storm_frames()
    dense = _dense_metrics(storms)
    args.image_root.mkdir(parents=True, exist_ok=True)
    _plot_validation_mae(validation, args.image_root / "validation-intensity-mae.png")
    _plot_training_curves(args.image_root / "training-validation-curves.png")
    _plot_storm_trajectories(
        storms, args.image_root / "three-storm-intensity-trajectories.png"
    )
    _plot_storm_mae(dense, args.image_root / "three-storm-mae.png")
    _copy_wandb_media(args.image_root)
    _publish_data(validation, storms, dense, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        _report_markdown(validation, storms, dense), encoding="utf-8"
    )
    print(f"Wrote {args.output}")


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
