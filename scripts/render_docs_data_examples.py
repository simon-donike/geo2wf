"""Render documentation figures from one exported GEO–ERA5–SAR sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/geo2wf_docs_matplotlib")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.colors import ListedColormap

from geo2wf.data.features import (
    append_era5_derived_channels as _append_era5_derived_channels,
    normalized_distance_to_center as _normalized_distance_to_center,
    solar_time_features as _solar_time_features,
)


DEFAULT_DATA_ROOT = ROOT / "data" / "geotiff" / "geo_sar_10bands_era5"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "assets" / "images"
DEFAULT_SAMPLE_ID = "WP232024_sar_geo_20241030095303_bb2c52ca"

INK = "#17242b"
MUTED = "#5d6b73"
MASK = ListedColormap(["#f1f3f4", "#147d83"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_manifest_row(data_root: Path, split: str, sample_id: str) -> pd.Series:
    manifest = pd.read_csv(data_root / split / "manifest.csv")
    rows = manifest.loc[manifest["sample_id"] == sample_id]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one manifest row for {sample_id!r}, found {len(rows)}"
        )
    return rows.iloc[0]


def read_raster(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], rasterio.coords.BoundingBox]:
    with rasterio.open(path) as dataset:
        array = dataset.read().astype(np.float32)
        mask = dataset.dataset_mask() > 0
        channels = [
            description or f"band_{index}"
            for index, description in enumerate(dataset.descriptions, start=1)
        ]
        return array, mask, channels, dataset.bounds


def resolve(data_root: Path, row: pd.Series, column: str) -> Path:
    path = Path(str(row[column]))
    return path if path.is_absolute() else data_root / path


def bounds_tensor(bounds: rasterio.coords.BoundingBox) -> torch.Tensor:
    return torch.tensor(
        [bounds.left, bounds.right, bounds.bottom, bounds.top],
        dtype=torch.float64,
    )


def percentile_limits(array: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = array[np.isfinite(array) & mask]
    if not valid.size:
        return 0.0, 1.0
    low, high = np.nanpercentile(valid, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def style_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#d5dadd")
        spine.set_linewidth(0.8)


def add_center(ax: plt.Axes, row: pd.Series, shape: tuple[int, int], bounds) -> None:
    height, width = shape
    column = (
        (float(row["ibtracs_center_lon"]) - bounds.left)
        / (bounds.right - bounds.left)
        * width
    )
    line = (
        (bounds.top - float(row["ibtracs_center_lat"]))
        / (bounds.top - bounds.bottom)
        * height
    )
    ax.plot(
        column, line, marker="+", color="#d3483e", markersize=9, markeredgewidth=1.8
    )


def save_figure(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 88, "method": 6},
    )
    plt.close(fig)
    print(f"Wrote {output_path}")


def render_geo(
    array: np.ndarray,
    mask: np.ndarray,
    channels: list[str],
    bounds,
    row: pd.Series,
    output_path: Path,
    dpi: int,
) -> None:
    selected = ["B08", "B09", "B13", "B14"]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.15), constrained_layout=True)
    for ax, channel in zip(axes, selected):
        index = channels.index(channel)
        image = np.where(mask, array[index], np.nan)
        low, high = percentile_limits(image, mask)
        artist = ax.imshow(image, cmap="Greys", vmin=low, vmax=high)
        add_center(ax, row, image.shape, bounds)
        ax.set_title(channel, color=INK, fontsize=10, fontweight="bold")
        ax.set_xlabel(f"{low:.0f}–{high:.0f} K", color=MUTED, fontsize=8)
        style_axis(ax)
        fig.colorbar(artist, ax=ax, fraction=0.044, pad=0.025)
    fig.suptitle(
        "GEO input · four of ten infrared / water-vapor bands",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, output_path, dpi)


def render_era5(
    array: np.ndarray,
    mask: np.ndarray,
    channels: list[str],
    bounds,
    row: pd.Series,
    output_path: Path,
    dpi: int,
) -> None:
    tensor, derived_channels = _append_era5_derived_channels(
        torch.from_numpy(array),
        channels.copy(),
        bounds_tensor(bounds),
    )
    fields = tensor.numpy()
    specs = [
        ("era5_precipitable_water", "Precipitable water", "kg m⁻²", 1.0, "Blues"),
        ("era5_sst", "Sea-surface temperature", "K", 1.0, "magma"),
        ("era5_pressure_msl", "Mean sea-level pressure", "hPa", 0.01, "viridis"),
        ("era5_wind_speed_10m", "10 m wind speed", "m s⁻¹", 1.0, "turbo"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.15), constrained_layout=True)
    for ax, (channel, title, unit, scale, cmap) in zip(axes, specs):
        index = derived_channels.index(channel)
        image = np.where(mask, fields[index] * scale, np.nan)
        low, high = percentile_limits(image, mask)
        artist = ax.imshow(image, cmap=cmap, vmin=low, vmax=high)
        add_center(ax, row, image.shape, bounds)
        ax.set_title(title, color=INK, fontsize=9.2, fontweight="bold")
        ax.set_xlabel(unit, color=MUTED, fontsize=8)
        style_axis(ax)
        fig.colorbar(artist, ax=ax, fraction=0.044, pad=0.025)
    fig.suptitle(
        "ERA5 context · four of nine source and derived fields",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, output_path, dpi)


def render_derived(
    shape: tuple[int, int],
    bounds,
    row: pd.Series,
    output_path: Path,
    dpi: int,
) -> None:
    bounds_value = bounds_tensor(bounds)
    center = torch.tensor(
        [float(row["ibtracs_center_lat"]), float(row["ibtracs_center_lon"])],
        dtype=torch.float64,
    )
    distance = _normalized_distance_to_center(bounds_value, shape, center).numpy()[0]
    solar = _solar_time_features(
        bounds_value,
        shape,
        pd.Timestamp(row["condition_timestamp"]),
    ).numpy()
    specs = [
        (distance, "Distance to center", "normalized 0–1", "viridis"),
        (solar[0], "Local solar time · sin", "shifted to 0–1", "twilight"),
        (solar[1], "Local solar time · cos", "shifted to 0–1", "twilight"),
        (solar[2], "Solar zenith angle", "angle ÷ π", "cividis"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.15), constrained_layout=True)
    for ax, (image, title, unit, cmap) in zip(axes, specs):
        artist = ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        add_center(ax, row, image.shape, bounds)
        ax.set_title(title, color=INK, fontsize=9.2, fontweight="bold")
        ax.set_xlabel(unit, color=MUTED, fontsize=8)
        style_axis(ax)
        fig.colorbar(artist, ax=ax, fraction=0.044, pad=0.025)
    fig.suptitle(
        "Derived context · generated by data/dataset.py for every sample",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, output_path, dpi)


def render_target(
    era5: np.ndarray,
    era5_mask: np.ndarray,
    era5_channels: list[str],
    sar: np.ndarray,
    sar_mask: np.ndarray,
    bounds,
    row: pd.Series,
    output_path: Path,
    dpi: int,
) -> None:
    era5_tensor, channels = _append_era5_derived_channels(
        torch.from_numpy(era5),
        era5_channels.copy(),
        bounds_tensor(bounds),
    )
    era5_wind = era5_tensor[channels.index("era5_wind_speed_10m")].numpy()
    target = sar[0]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.2), constrained_layout=True)
    panels = [
        (
            np.where(era5_mask, era5_wind, np.nan),
            "ERA5 wind",
            "dense physical anchor",
            "turbo",
            0.0,
            60.0,
        ),
        (
            np.where(sar_mask, target, np.nan),
            "SAR wind target",
            "training supervision · m s⁻¹",
            "turbo",
            0.0,
            60.0,
        ),
        (
            sar_mask.astype(float),
            "SAR target mask",
            "1 = observed, 0 = missing",
            MASK,
            0.0,
            1.0,
        ),
    ]
    for ax, (image, title, label, cmap, low, high) in zip(axes, panels):
        artist = ax.imshow(image, cmap=cmap, vmin=low, vmax=high)
        add_center(ax, row, image.shape, bounds)
        ax.set_title(title, color=INK, fontsize=10, fontweight="bold")
        ax.set_xlabel(label, color=MUTED, fontsize=8)
        style_axis(ax)
        if title != "SAR target mask":
            fig.colorbar(artist, ax=ax, fraction=0.044, pad=0.025)
    fig.suptitle(
        "Physical anchor and supervision · same 256 × 256 grid",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, output_path, dpi)


def main() -> None:
    args = parse_args()
    row = read_manifest_row(args.data_root, args.split, args.sample_id)
    geo, geo_mask, geo_channels, bounds = read_raster(
        resolve(args.data_root, row, "condition_path")
    )
    era5, era5_mask, era5_channels, era5_bounds = read_raster(
        resolve(args.data_root, row, "context_path")
    )
    sar, sar_mask, _, sar_bounds = read_raster(
        resolve(args.data_root, row, "target_path")
    )
    if bounds != era5_bounds or bounds != sar_bounds:
        raise ValueError("Example rasters do not share one grid")

    metadata_path = args.output_dir / "data-example-metadata.json"
    metadata = {
        "sample_id": str(row["sample_id"]),
        "storm_id": str(row["storm_id"]),
        "condition_sensor": str(row["condition_sensor"]),
        "target_sensor": str(row["target_sensor"]),
        "condition_timestamp": str(row["condition_timestamp"]),
        "target_timestamp": str(row["target_timestamp"]),
        "dt_minutes": float(row["dt_minutes"]),
        "grid_size": int(row["grid_size"]),
        "grid_resolution": float(row["grid_resolution"]),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    render_geo(
        geo,
        geo_mask,
        geo_channels,
        bounds,
        row,
        args.output_dir / "data-example-geo.webp",
        args.dpi,
    )
    render_era5(
        era5,
        era5_mask,
        era5_channels,
        bounds,
        row,
        args.output_dir / "data-example-era5.webp",
        args.dpi,
    )
    render_derived(
        geo.shape[-2:],
        bounds,
        row,
        args.output_dir / "data-example-derived.webp",
        args.dpi,
    )
    render_target(
        era5,
        era5_mask,
        era5_channels,
        sar,
        sar_mask,
        bounds,
        row,
        args.output_dir / "data-example-target.webp",
        args.dpi,
    )
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
