#!/usr/bin/env python3
"""Publish the completed experiment matrix and dense storm nowcasts to docs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
STORMS = ("AL082025", "EP112025", "EP182023")
STORM_NAMES = {
    "AL082025": "Humberto (2025)",
    "EP112025": "Kiko (2025)",
    "EP182023": "Otis (2023)",
}
REGIMES = (("with_era5", "With ERA5"), ("without_era5", "Without ERA5"))
ERA5_CROP_DEGREES = 192 * 0.027
RI_LABEL = "RI phase (≥30 kt in preceding 24 h)"


EXPERIMENTS: dict[str, dict[str, str]] = {
    "correction_image_radii": {
        "label": "U-Net + standard MLP · image radii",
        "era5": "With",
        "sar": "Yes",
        "radii": "2D image",
    },
    "correction_mlp_radii": {
        "label": "U-Net + standard MLP · MLP radii",
        "era5": "With",
        "sar": "Yes",
        "radii": "MLP head",
    },
    "latent_sar_era5_max_wind": {
        "label": "Latent MLP · SAR · wind only",
        "era5": "With",
        "sar": "Yes",
        "radii": "None",
    },
    "latent_sar_era5_max_wind_radii": {
        "label": "Latent MLP · SAR · wind + radii",
        "era5": "With",
        "sar": "Yes",
        "radii": "MLP head",
    },
    "latent_no_sar_era5_max_wind": {
        "label": "Latent MLP · no SAR · wind only",
        "era5": "With",
        "sar": "No",
        "radii": "None",
    },
    "latent_no_sar_era5_max_wind_radii": {
        "label": "Latent MLP · no SAR · wind + radii",
        "era5": "With",
        "sar": "No",
        "radii": "MLP head",
    },
    "latent_sar_no_era5_max_wind": {
        "label": "Latent MLP · SAR · wind only",
        "era5": "Without",
        "sar": "Yes",
        "radii": "None",
    },
    "latent_sar_no_era5_max_wind_radii": {
        "label": "Latent MLP · SAR · wind + radii",
        "era5": "Without",
        "sar": "Yes",
        "radii": "MLP head",
    },
    "latent_no_sar_no_era5_max_wind": {
        "label": "Latent MLP · no SAR · wind only",
        "era5": "Without",
        "sar": "No",
        "radii": "None",
    },
    "latent_no_sar_no_era5_max_wind_radii": {
        "label": "Latent MLP · no SAR · wind + radii",
        "era5": "Without",
        "sar": "No",
        "radii": "MLP head",
    },
}


SERIES: dict[str, dict[str, str]] = {
    "unet_raw": {
        "label": "Raw field U-Net",
        "column": "unet_raw_max_ms",
        "family": "core",
        "color": "#D55E00",
    },
    "unet_correction": {
        "label": "U-Net + standard MLP",
        "column": "unet_correction_ms",
        "family": "core",
        "color": "#0072B2",
    },
    "joint_unet_mlp": {
        "label": "Joint U-Net + latent MLP",
        "column": "joint_unet_mlp_ms",
        "family": "core",
        "color": "#009E73",
    },
    "correction_image_radii": {
        "label": "Standard MLP · image radii",
        "column": "correction_image_radii_max_wind_ms",
        "family": "radii",
        "color": "#AA4499",
    },
    "correction_mlp_radii": {
        "label": "Standard MLP · MLP radii",
        "column": "correction_mlp_radii_max_wind_ms",
        "family": "radii",
        "color": "#44AA99",
    },
    "latent_sar_max_wind": {
        "label": "Latent MLP · SAR · wind only",
        "column_template": "latent_sar_{era5}_max_wind_max_wind_ms",
        "family": "latent",
        "color": "#CC79A7",
    },
    "latent_sar_max_wind_radii": {
        "label": "Latent MLP · SAR · wind + radii",
        "column_template": "latent_sar_{era5}_max_wind_radii_max_wind_ms",
        "family": "latent,radii",
        "color": "#E69F00",
    },
    "latent_no_sar_max_wind": {
        "label": "Latent MLP · no SAR · wind only",
        "column_template": "latent_no_sar_{era5}_max_wind_max_wind_ms",
        "family": "latent",
        "color": "#56B4E9",
    },
    "latent_no_sar_max_wind_radii": {
        "label": "Latent MLP · no SAR · wind + radii",
        "column_template": "latent_no_sar_{era5}_max_wind_radii_max_wind_ms",
        "family": "latent,radii",
        "color": "#332288",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    evaluation = ROOT / "logs" / "current-experiment-evaluation"
    parser.add_argument(
        "--validation-csv", type=Path, default=evaluation / "validation-metrics.csv"
    )
    parser.add_argument(
        "--validation-json", type=Path, default=evaluation / "validation-results.json"
    )
    parser.add_argument(
        "--with-era5-csv", type=Path, default=evaluation / "three-storm/with-era5.csv"
    )
    parser.add_argument(
        "--without-era5-csv",
        type=Path,
        default=evaluation / "three-storm/without-era5.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "inference/inf_data/index-files/observation_manifest_v6.csv",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "inference/inf_data")
    parser.add_argument("--docs-page", type=Path, default=ROOT / "docs/results.md")
    parser.add_argument(
        "--data-output", type=Path, default=ROOT / "docs/assets/data/final-results"
    )
    parser.add_argument(
        "--image-output", type=Path, default=ROOT / "docs/assets/images/final-results"
    )
    parser.add_argument("--smoothing-hours", type=int, default=3)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _regime_series(frame: pd.DataFrame, regime: str) -> list[dict[str, str]]:
    era5_token = "era5" if regime == "with_era5" else "no_era5"
    result = []
    for key, raw in SERIES.items():
        item = dict(raw)
        column = item.get("column") or item["column_template"].format(era5=era5_token)
        if column not in frame.columns:
            continue
        item.update({"key": key, "column": column, "regime": regime})
        result.append(item)
    return result


def load_storm_frames(
    with_era5_path: Path, without_era5_path: Path
) -> dict[str, pd.DataFrame]:
    frames = {
        "with_era5": pd.read_csv(with_era5_path),
        "without_era5": pd.read_csv(without_era5_path),
    }
    required = {
        "observation_id",
        "storm_id",
        "observation_timestamp",
        "target_ms",
        "inference_valid",
        "is_rapid_intensification",
    }
    for regime, frame in frames.items():
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{regime} storm CSV is missing {sorted(missing)}")
        frame["observation_timestamp"] = pd.to_datetime(
            frame["observation_timestamp"], utc=True
        )
        frame["observation_id"] = frame["observation_id"].astype(str)
        frame["storm_id"] = frame["storm_id"].astype(str)
        frames[regime] = frame.sort_values(
            ["storm_id", "observation_timestamp", "observation_id"]
        ).reset_index(drop=True)
    reference = frames["with_era5"]
    candidate = frames["without_era5"]
    if not reference["observation_id"].equals(candidate["observation_id"]):
        raise ValueError("ERA5/no-ERA5 dense storm cohorts differ")
    if not np.allclose(reference["target_ms"], candidate["target_ms"]):
        raise ValueError("ERA5/no-ERA5 dense storm targets differ")
    found = tuple(reference["storm_id"].drop_duplicates())
    if set(found) != set(STORMS):
        raise ValueError(f"expected storms {STORMS}, found {found}")
    return frames


_CENTER_PATTERN = re.compile(r"_\[([+-]\d+(?:\.\d+)?)deg_([+-]\d+(?:\.\d+)?)deg\]")


def _input_center(row: Any) -> tuple[float, float]:
    match = _CENTER_PATTERN.search(Path(str(row.input_path)).name)
    if match:
        return float(match.group(1)), float(match.group(2))
    return float(row.center_lat), float(row.center_lon)


def add_era5_maximum_wind(
    frame: pd.DataFrame, manifest_path: Path, data_root: Path
) -> pd.DataFrame:
    """Diagnose raw ERA5 maximum wind over the model's storm-centred crop."""

    result = frame.copy()
    manifest = pd.read_csv(manifest_path)
    storm_ids = set(result["storm_id"].astype(str))
    records = manifest.loc[
        (manifest["source_type"].astype(str).str.lower() == "era5")
        & manifest["storm_id"].astype(str).isin(storm_ids)
    ]
    if records["storm_id"].nunique() != len(storm_ids):
        raise ValueError("manifest does not provide one ERA5 source for every storm")
    maxima = pd.Series(index=result.index, dtype=float)
    valid_times = pd.Series(index=result.index, dtype="object")
    half_extent = ERA5_CROP_DEGREES / 2.0
    for storm_id, storm in result.groupby("storm_id", sort=False):
        matches = records.loc[records["storm_id"].astype(str) == storm_id]
        if len(matches) != 1:
            raise ValueError(f"expected one ERA5 source for {storm_id}")
        source_path = data_root / str(matches.iloc[0]["path"])
        with xr.open_dataset(
            source_path,
            group="rectilinear",
            engine="h5netcdf",
            decode_times=True,
        ) as source:
            times = pd.DatetimeIndex(source["time"].values)
            times = (
                times.tz_localize("UTC")
                if times.tz is None
                else times.tz_convert("UTC")
            )
            u = source["u_wind_10m"].load()
            v = source["v_wind_10m"].load()
            latitudes = source["latitude"].load()
            longitudes = source["longitude"].load()
            for row in storm.itertuples():
                time_index = int(
                    np.argmin(np.abs(times - pd.Timestamp(row.observation_timestamp)))
                )
                center_lat, center_lon = _input_center(row)
                lat = np.asarray(latitudes.isel(time=time_index), dtype=float)
                lon = np.asarray(longitudes.isel(time=time_index), dtype=float)
                lon = ((lon + 180.0) % 360.0) - 180.0
                lat_mask = np.abs(lat - center_lat) <= half_extent
                lon_delta = ((lon - center_lon + 180.0) % 360.0) - 180.0
                lon_mask = np.abs(lon_delta) <= half_extent
                if not lat_mask.any() or not lon_mask.any():
                    raise ValueError(f"ERA5 crop misses {row.observation_id}")
                wind = np.hypot(
                    np.asarray(u.isel(time=time_index), dtype=float),
                    np.asarray(v.isel(time=time_index), dtype=float),
                )
                values = wind[np.ix_(lat_mask, lon_mask)]
                if not np.isfinite(values).any():
                    raise ValueError(f"ERA5 crop is invalid for {row.observation_id}")
                maxima.at[row.Index] = float(np.nanmax(values))
                valid_times.at[row.Index] = times[time_index].isoformat()
    result["era5_max_wind_ms"] = maxima
    result["era5_valid_timestamp"] = valid_times
    return result


def storm_prediction_rows(
    frames: Mapping[str, pd.DataFrame],
    series: Mapping[str, Sequence[Mapping[str, str]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime, _ in REGIMES:
        frame = frames[regime]
        valid = frame["inference_valid"].astype(bool)
        for item in series[regime]:
            selected = frame.loc[valid & np.isfinite(frame[item["column"]])]
            for row in selected.itertuples(index=False):
                rows.append(
                    {
                        "observation_id": row.observation_id,
                        "storm_id": row.storm_id,
                        "observation_timestamp": row.observation_timestamp.isoformat(),
                        "model_key": item["key"],
                        "model": item["label"],
                        "conditioning": regime,
                        "prediction_ms": float(getattr(row, item["column"])),
                        "target_ms": float(row.target_ms),
                        "is_rapid_intensification": bool(row.is_rapid_intensification),
                    }
                )
    reference = frames["with_era5"]
    selected = reference.loc[
        reference["inference_valid"].astype(bool)
        & np.isfinite(reference["era5_max_wind_ms"])
    ]
    for row in selected.itertuples(index=False):
        rows.append(
            {
                "observation_id": row.observation_id,
                "storm_id": row.storm_id,
                "observation_timestamp": row.observation_timestamp.isoformat(),
                "model_key": "era5_max_wind",
                "model": "ERA5 10 m maximum",
                "conditioning": "reference",
                "prediction_ms": float(row.era5_max_wind_ms),
                "target_ms": float(row.target_ms),
                "is_rapid_intensification": bool(row.is_rapid_intensification),
            }
        )
    output = pd.DataFrame(rows)
    output["error_ms"] = output["prediction_ms"] - output["target_ms"]
    return output


def storm_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(
        ["model_key", "model", "conditioning"], sort=False
    ):
        subsets = (("all_three_storms", group),)
        ri = group.loc[group["is_rapid_intensification"].astype(bool)]
        if not ri.empty:
            subsets += (("ri_three_storms", ri),)
        for subset, selected in subsets:
            error = selected["error_ms"].to_numpy(float)
            rows.append(
                {
                    "model_key": keys[0],
                    "model": keys[1],
                    "conditioning": keys[2],
                    "subset": subset,
                    "samples": len(selected),
                    "mae_ms": float(np.mean(np.abs(error))),
                    "rmse_ms": float(np.sqrt(np.mean(np.square(error)))),
                    "bias_ms": float(np.mean(error)),
                }
            )
    return pd.DataFrame(rows)


def _hourly_smoothed(
    frame: pd.DataFrame, storm_id: str, column: str, smoothing_hours: int
) -> pd.Series:
    values = (
        frame.loc[frame["storm_id"] == storm_id]
        .set_index("observation_timestamp")[column]
        .resample("1h")
        .mean()
    )
    return values.rolling(smoothing_hours, center=True, min_periods=1).mean().dropna()


def _ri_spans(
    frame: pd.DataFrame, storm_id: str
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    hourly = (
        frame.loc[frame["storm_id"] == storm_id]
        .set_index("observation_timestamp")["is_rapid_intensification"]
        .astype(bool)
        .resample("1h")
        .max()
        .fillna(False)
    )
    spans = []
    start = None
    for timestamp, active in hourly.items():
        if active and start is None:
            start = timestamp
        elif not active and start is not None:
            spans.append((start, timestamp))
            start = None
    if start is not None:
        spans.append((start, hourly.index[-1] + pd.Timedelta(hours=1)))
    return spans


def figure_data_rows(
    frames: Mapping[str, pd.DataFrame],
    series: Mapping[str, Sequence[Mapping[str, str]]],
    family: str,
    smoothing_hours: int,
) -> pd.DataFrame:
    """Return the exact hourly, smoothed values rendered in one figure family."""

    reference = frames["with_era5"]
    rows: list[dict[str, Any]] = []
    for storm_id in reference["storm_id"].drop_duplicates():
        ri_spans = _ri_spans(reference, storm_id)
        for regime, _ in REGIMES:
            sources = [
                ("ibtracs_best_track", "IBTrACS best track", reference, "target_ms"),
                (
                    "era5_max_wind",
                    "ERA5 10 m maximum",
                    reference,
                    "era5_max_wind_ms",
                ),
            ]
            sources.extend(
                (item["key"], item["label"], frames[regime], item["column"])
                for item in series[regime]
                if family in item["family"].split(",")
            )
            for series_key, label, source, column in sources:
                values = _hourly_smoothed(source, storm_id, column, smoothing_hours)
                for timestamp, value in values.items():
                    rows.append(
                        {
                            "storm_id": storm_id,
                            "storm": STORM_NAMES.get(storm_id, storm_id),
                            "conditioning": regime,
                            "observation_timestamp": timestamp.isoformat(),
                            "series_key": series_key,
                            "series": label,
                            "maximum_wind_ms": float(value),
                            "is_rapid_intensification": any(
                                start <= timestamp < end for start, end in ri_spans
                            ),
                            "aggregation": "hourly mean",
                            "smoothing_hours": smoothing_hours,
                        }
                    )
    return pd.DataFrame(rows)


def write_family_figure(
    frames: Mapping[str, pd.DataFrame],
    series: Mapping[str, Sequence[Mapping[str, str]]],
    family: str,
    smoothing_hours: int,
    png_path: Path,
    pdf_path: Path,
) -> None:
    titles = {
        "core": "Core maximum-wind nowcasts",
        "latent": "Latent-MLP experiment matrix",
        "radii": "Radii-supervised maximum-wind nowcasts",
    }
    with plt.rc_context(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.0,
            "lines.solid_capstyle": "round",
            "figure.dpi": 120,
        }
    ):
        figure, axes = plt.subplots(
            len(STORMS), 2, figsize=(14.0, 9.2), sharey="row", squeeze=False
        )
        reference = frames["with_era5"]
        legend: dict[str, Any] = {}
        panel = 0
        for row_index, storm_id in enumerate(STORMS):
            for column_index, (regime, regime_label) in enumerate(REGIMES):
                axis = axes[row_index, column_index]
                panel += 1
                for span_index, (start, end) in enumerate(
                    _ri_spans(reference, storm_id)
                ):
                    patch = axis.axvspan(
                        start,
                        end,
                        color="#F4A3A8",
                        alpha=0.22,
                        linewidth=0,
                        label=RI_LABEL if span_index == 0 else None,
                        zorder=0,
                    )
                    if span_index == 0:
                        legend.setdefault(RI_LABEL, patch)
                truth = _hourly_smoothed(
                    reference, storm_id, "target_ms", smoothing_hours
                )
                line = axis.plot(
                    truth.index,
                    truth,
                    color="#111111",
                    linewidth=2.4,
                    label="IBTrACS best track",
                    zorder=9,
                )[0]
                legend.setdefault("IBTrACS best track", line)
                era5 = _hourly_smoothed(
                    reference, storm_id, "era5_max_wind_ms", smoothing_hours
                )
                line = axis.plot(
                    era5.index,
                    era5,
                    color="#777777",
                    linewidth=1.8,
                    linestyle=(0, (5, 2)),
                    label="ERA5 10 m maximum",
                    zorder=7,
                )[0]
                legend.setdefault("ERA5 10 m maximum", line)
                for item in series[regime]:
                    if family not in item["family"].split(","):
                        continue
                    values = _hourly_smoothed(
                        frames[regime], storm_id, item["column"], smoothing_hours
                    )
                    line = axis.plot(
                        values.index,
                        values,
                        color=item["color"],
                        linewidth=1.55,
                        alpha=0.96,
                        label=item["label"],
                        zorder=5,
                    )[0]
                    legend.setdefault(item["label"], line)
                axis.text(
                    0.012,
                    0.965,
                    f"({chr(96 + panel)})",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontweight="bold",
                )
                if row_index == 0:
                    axis.set_title(regime_label)
                if column_index == 0:
                    axis.set_ylabel(
                        f"{STORM_NAMES[storm_id]}\nMaximum wind (m s$^{{-1}}$)"
                    )
                axis.set_ylim(bottom=0.0)
                axis.grid(color="#D0D0D0", alpha=0.45, linewidth=0.6)
                locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
                axis.xaxis.set_major_locator(locator)
                axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        for axis in axes[-1, :]:
            axis.set_xlabel("Observation time (UTC)")
        figure.suptitle(
            f"{titles[family]} across complete validation-storm lifecycles\n"
            f"{smoothing_hours}-h centred rolling mean; scores use unsmoothed native observations",
            fontsize=14,
            y=0.995,
        )
        figure.legend(
            list(legend.values()),
            list(legend.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=3,
            frameon=False,
        )
        figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.955), h_pad=1.3, w_pad=1.0)
        for path, options in ((png_path, {"dpi": 300}), (pdf_path, {})):
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, bbox_inches="tight", facecolor="white", **options)
        plt.close(figure)


def _metric_value(
    metrics: pd.DataFrame,
    experiment: str,
    subset: str,
    output: str,
    target: str,
    metric: str,
) -> float | None:
    selected = metrics.loc[
        (metrics["experiment"] == experiment)
        & (metrics["subset"] == subset)
        & (metrics["output"] == output)
        & (metrics["target"] == target)
        & (metrics["metric"] == metric)
        & metrics["available"].astype(bool)
    ]
    if selected.empty:
        return None
    if len(selected) != 1:
        raise ValueError("validation metric key is not unique")
    return float(selected.iloc[0]["value"])


def _number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _validation_wind_table(metrics: pd.DataFrame) -> str:
    lines = [
        "| Experiment | ERA5 | SAR | Radii supervision | All MAE | All RMSE | RI MAE | RI RMSE |",
        "|---|:---:|:---:|---|---:|---:|---:|---:|",
    ]
    for experiment, metadata in EXPERIMENTS.items():
        values = [
            _metric_value(
                metrics, experiment, subset, "scalar_head", "maximum_wind", metric
            )
            for subset in ("all_validation", "ri_validation")
            for metric in ("mae", "rmse")
        ]
        lines.append(
            f"| {metadata['label']} | {metadata['era5']} | {metadata['sar']} | "
            f"{metadata['radii']} | "
            + " | ".join(_number(value) for value in values)
            + " |"
        )
    return "\n".join(lines)


def _validation_image_table(metrics: pd.DataFrame) -> str:
    lines = [
        "| Experiment | ERA5 | All L1 | All PSNR | All SSIM | RI L1 | RI PSNR | RI SSIM |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment, metadata in EXPERIMENTS.items():
        values = [
            _metric_value(
                metrics,
                experiment,
                subset,
                "image_reconstruction",
                "wind_field",
                metric,
            )
            for subset in ("all_validation", "ri_validation")
            for metric in ("l1", "psnr", "ssim")
        ]
        if all(value is None for value in values):
            continue
        lines.append(
            f"| {metadata['label']} | {metadata['era5']} | "
            + " | ".join(_number(value) for value in values)
            + " |"
        )
    return "\n".join(lines)


def _validation_radius_table(metrics: pd.DataFrame) -> str:
    lines = [
        "| Experiment | ERA5 | Radius source | RMW MAE | R34 MAE | R50 MAE | R64 MAE |",
        "|---|:---:|---|---:|---:|---:|---:|",
    ]
    targets = {
        "scalar_radius_head": (
            "rmw",
            "r34_equivalent",
            "r50_equivalent",
            "r64_equivalent",
        ),
        "image_derived_radius": ("rmw", "r34", "r50", "r64"),
    }
    source_labels = {
        "scalar_radius_head": "Direct MLP head",
        "image_derived_radius": "Diagnosed from 2D image",
    }
    for experiment, metadata in EXPERIMENTS.items():
        for output, radius_targets in targets.items():
            values = [
                _metric_value(
                    metrics, experiment, "all_validation", output, target, "mae"
                )
                for target in radius_targets
            ]
            if all(value is None for value in values):
                continue
            lines.append(
                f"| {metadata['label']} | {metadata['era5']} | "
                f"{source_labels[output]} | "
                + " | ".join(_number(value, 2) for value in values)
                + " |"
            )
    return "\n".join(lines)


def _storm_table(metrics: pd.DataFrame) -> str:
    overall = metrics.loc[metrics["subset"] == "all_three_storms"]
    ri = metrics.loc[metrics["subset"] == "ri_three_storms"].set_index(
        ["model_key", "conditioning"]
    )
    lines = [
        "| Model | Conditioning | All n | MAE | RMSE | Bias | RI n | RI MAE | RI RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        ri_row = ri.loc[(row.model_key, row.conditioning)]
        conditioning = {
            "with_era5": "With ERA5",
            "without_era5": "Without ERA5",
            "reference": "Reanalysis reference",
        }[row.conditioning]
        lines.append(
            f"| {row.model} | {conditioning} | {row.samples} | {row.mae_ms:.3f} | "
            f"{row.rmse_ms:.3f} | {row.bias_ms:.3f} | {ri_row.samples} | "
            f"{ri_row.mae_ms:.3f} | {ri_row.rmse_ms:.3f} |"
        )
    return "\n".join(lines)


def _relative(path: Path, docs_page: Path) -> str:
    return path.resolve().relative_to(docs_page.parent.resolve()).as_posix()


def _download_csv(docs_page: Path, download: Path) -> str:
    return (
        f"[Download CSV]({_relative(download, docs_page)})"
        "{ .md-button .result-download download }"
    )


def write_docs(
    path: Path,
    validation_metrics: pd.DataFrame,
    storm_metrics: pd.DataFrame,
    data_paths: Mapping[str, Path],
    figures: Mapping[str, tuple[Path, Path]],
    smoothing_hours: int,
) -> None:
    figure_sections = []
    captions = {
        "core": "Core field and scalar architectures",
        "latent": "SAR/no-SAR and ERA5/no-ERA5 latent-MLP matrix",
        "radii": "Radii-supervised correction and latent experiments",
    }
    for family in ("core", "latent", "radii"):
        png, _ = figures[family]
        figure_sections.extend(
            [
                f"### {captions[family]}",
                "",
                f"![{captions[family]}]({_relative(png, path)})",
                "",
                _download_csv(path, data_paths[f"figure_{family}"]),
                "",
            ]
        )
    content = "\n".join(
        [
            "# Current results",
            "",
            '!!! success "Canonical experiment outputs"',
            "    These tables are generated directly from the completed experiment",
            "    matrix and its selected checkpoints. They are the paper-facing source",
            "    of truth for current validation, RI validation, and the three complete",
            "    validation-storm case studies.",
            "",
            "## Current validation matrix",
            "",
            "Maximum-wind targets are interpolated IBTrACS USA_WIND values. RI is",
            "defined as an increase of at least 30 kt during the preceding 24 hours.",
            "The configured validation cohorts are used exactly as trained; ERA5-required",
            "runs exclude observations without valid matched ERA5, so cross-regime rankings",
            "are descriptive rather than strictly paired.",
            "",
            "### Maximum wind",
            "",
            "All wind errors are in m s⁻¹.",
            "",
            _validation_wind_table(validation_metrics),
            "",
            _download_csv(path, data_paths["validation_wind"]),
            "",
            "### Wind-field reconstruction",
            "",
            "L1 is pooled valid-pixel MAE in m s⁻¹. PSNR uses the fixed 79.8 m s⁻¹",
            "physical range; SSIM is the scene mean over complete valid 7×7 windows.",
            "Encoder-only no-SAR models have no image output and are therefore omitted.",
            "",
            _validation_image_table(validation_metrics),
            "",
            _download_csv(path, data_paths["validation_image"]),
            "",
            "#### Held-out reconstruction examples",
            "",
            "Four validation observations from the selected SAR + ERA5 latent-MLP model",
            "with wind-and-radii supervision are shown below. Predictions are complete 2D",
            "wind-speed fields; SAR targets are only observed inside the orange footprint,",
            "so the unobserved part of each prediction is a conditional reconstruction.",
            "The red cross marks the interpolated IBTrACS storm centre.",
            "",
            "[![Held-out 2D wind-field reconstructions]"
            "(assets/images/final-results/current-validation-windfields-batch-01.jpg)]"
            "(assets/images/final-results/current-validation-windfields-batch-01.jpg)",
            "",
            "### Wind radii",
            "",
            "The compact table reports all-validation MAE in km. The downloadable",
            "canonical table additionally contains RMSE, bias, RI-only values, and",
            "explicit not-applicable reasons for every experiment/metric combination.",
            "ERA5 identifies whether ERA5 fields were supplied as conditioning inputs;",
            "rows with the same experiment label but different ERA5 values are separate",
            "input-ablation runs, not repeated measurements.",
            "",
            _validation_radius_table(validation_metrics),
            "",
            _download_csv(path, data_paths["validation_radii"]),
            "",
            "## Complete-storm nowcasts",
            "",
            "Humberto 2025, Kiko 2025, and Otis 2023 are dense validation-storm",
            "case studies. Each value is an independent instantaneous nowcast from one",
            "GEO observation. Figures show hourly means followed by a centred",
            f"{smoothing_hours}-hour rolling mean; every metric below uses all valid,",
            "unsmoothed native observations. Shaded intervals are RI phases.",
            "",
            "The ERA5 line is the maximum native 10 m wind speed within the same",
            f"{ERA5_CROP_DEGREES:.3f}° storm-centred crop used by the models at the",
            "nearest ERA5 analysis time. It is an external reanalysis reference in the",
            "without-ERA5 panels, not an input to those models.",
            "",
            *figure_sections,
            "### Dense three-storm metrics",
            "",
            "Wind errors are in m s⁻¹. The two invalid GEO observations are excluded",
            "consistently from every model series and the ERA5 reference.",
            "",
            _storm_table(storm_metrics),
            "",
            _download_csv(path, data_paths["storm_metrics"]),
            "",
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def publish(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoothing_hours < 1:
        raise ValueError("smoothing-hours must be positive")
    validation = pd.read_csv(args.validation_csv)
    expected = set(EXPERIMENTS)
    found = set(validation["experiment"].astype(str))
    if found != expected:
        raise ValueError(
            f"validation experiment set differs: missing={sorted(expected-found)}, "
            f"extra={sorted(found-expected)}"
        )
    frames = load_storm_frames(args.with_era5_csv, args.without_era5_csv)
    frames["with_era5"] = add_era5_maximum_wind(
        frames["with_era5"], args.manifest, args.data_root
    )
    era5_by_id = frames["with_era5"].set_index("observation_id")
    frames["without_era5"]["era5_max_wind_ms"] = frames["without_era5"][
        "observation_id"
    ].map(era5_by_id["era5_max_wind_ms"])
    frames["without_era5"]["era5_valid_timestamp"] = frames["without_era5"][
        "observation_id"
    ].map(era5_by_id["era5_valid_timestamp"])
    series = {regime: _regime_series(frames[regime], regime) for regime, _ in REGIMES}
    predictions = storm_prediction_rows(frames, series)
    storm_metrics = storm_metric_rows(predictions)

    data_paths = {
        "validation_csv": args.data_output / "current-validation-metrics.csv",
        "validation_json": args.data_output / "current-validation-results.json",
        "validation_wind": args.data_output / "current-validation-maximum-wind.csv",
        "validation_image": args.data_output
        / "current-validation-image-reconstruction.csv",
        "validation_radii": args.data_output / "current-validation-radii.csv",
        "storm_predictions": args.data_output / "current-three-storm-predictions.csv",
        "storm_metrics": args.data_output / "current-three-storm-metrics.csv",
        "publication_json": args.data_output / "current-results.json",
        "with_era5": args.data_output / "current-three-storm-with-era5.csv",
        "without_era5": args.data_output / "current-three-storm-without-era5.csv",
        **{
            f"figure_{family}": args.data_output
            / f"current-three-storm-{family}-nowcasts.csv"
            for family in ("core", "latent", "radii")
        },
    }
    args.data_output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.validation_csv, data_paths["validation_csv"])
    shutil.copyfile(args.validation_json, data_paths["validation_json"])
    _atomic_csv(
        validation.loc[
            (validation["output"] == "scalar_head")
            & (validation["target"] == "maximum_wind")
            & validation["metric"].isin(("mae", "rmse"))
            & validation["available"].astype(bool)
        ],
        data_paths["validation_wind"],
    )
    _atomic_csv(
        validation.loc[
            (validation["output"] == "image_reconstruction")
            & validation["metric"].isin(("l1", "psnr", "ssim"))
            & validation["available"].astype(bool)
        ],
        data_paths["validation_image"],
    )
    _atomic_csv(
        validation.loc[
            validation["output"].isin(("scalar_radius_head", "image_derived_radius"))
            & (validation["subset"] == "all_validation")
            & (validation["metric"] == "mae")
            & validation["available"].astype(bool)
        ],
        data_paths["validation_radii"],
    )
    _atomic_csv(predictions, data_paths["storm_predictions"])
    _atomic_csv(storm_metrics, data_paths["storm_metrics"])
    _atomic_csv(frames["with_era5"], data_paths["with_era5"])
    _atomic_csv(frames["without_era5"], data_paths["without_era5"])

    figures = {
        family: (
            args.image_output / f"current-three-storm-{family}-nowcasts.png",
            args.image_output / f"current-three-storm-{family}-nowcasts.pdf",
        )
        for family in ("core", "latent", "radii")
    }
    for family, (png_path, pdf_path) in figures.items():
        _atomic_csv(
            figure_data_rows(
                frames,
                series,
                family,
                args.smoothing_hours,
            ),
            data_paths[f"figure_{family}"],
        )
        write_family_figure(
            frames,
            series,
            family,
            args.smoothing_hours,
            png_path,
            pdf_path,
        )

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "canonical_current_validation_and_dense_storm_case_studies",
        "inputs": {
            "validation_csv": {
                "path": str(args.validation_csv),
                "sha256": _sha256(args.validation_csv),
            },
            "validation_json": {
                "path": str(args.validation_json),
                "sha256": _sha256(args.validation_json),
            },
            "with_era5_csv": {
                "path": str(args.with_era5_csv),
                "sha256": _sha256(args.with_era5_csv),
            },
            "without_era5_csv": {
                "path": str(args.without_era5_csv),
                "sha256": _sha256(args.without_era5_csv),
            },
        },
        "storm_figure_processing": {
            "aggregation": "hourly mean",
            "smoothing": f"centred {args.smoothing_hours}-hour rolling mean",
            "metrics_use_smoothed_values": False,
            "ri_definition": "IBTrACS maximum wind increase >=30 kt in preceding 24 h",
            "era5_reference": {
                "variable": "sqrt(u_wind_10m^2 + v_wind_10m^2)",
                "statistic": "maximum",
                "crop_degrees": ERA5_CROP_DEGREES,
                "time_matching": "nearest ERA5 analysis",
            },
        },
        "storms": list(STORMS),
        "series": series,
        "storm_metrics": storm_metrics.to_dict(orient="records"),
    }
    _atomic_json(payload, data_paths["publication_json"])
    write_docs(
        args.docs_page,
        validation,
        storm_metrics,
        data_paths,
        figures,
        args.smoothing_hours,
    )
    return payload


def main() -> None:
    payload = publish(parse_args())
    print(
        f"Published {len(payload['storm_metrics'])} dense-storm metric rows and "
        f"{len(EXPERIMENTS)} validation experiments"
    )


if __name__ == "__main__":
    main()
