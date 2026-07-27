from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dif_img_rec_matplotlib")

import numpy as np
import pandas as pd
import rasterio
from global_land_mask import globe
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


def plot_random_geo_sar_pairs(
    root: str | Path,
    split: str = "train",
    n: int = 5,
    *,
    seed: int | None = None,
    geo_band: int = 1,
    sar_band: int = 1,
    center_columns: tuple[str, str] = ("center_lat", "center_lon"),
    output_path: str | Path | None = None,
    dpi: int = 180,
) -> Figure:
    """Plot random exported GEO/SAR pairs with storm centers and footprints.

    Parameters
    ----------
    root:
        Export root containing split directories and manifests, for example
        ``data/geotiff/geo_sar``.
    split:
        Dataset split to sample from.
    n:
        Number of random pairs to plot.
    seed:
        Optional random seed for reproducible sample selection.
    geo_band, sar_band:
        One-based GeoTIFF band indices to display.
    center_columns:
        Manifest latitude/longitude columns used for the storm-center marker.
    output_path:
        Optional file path where the figure should be saved.
    dpi:
        Save/display resolution.
    """
    root = Path(root).expanduser()
    manifest_file = root / split / "manifest.csv"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_file}")

    manifest = pd.read_csv(manifest_file, keep_default_na=False)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_file}")

    rows = manifest.sample(
        n=min(n, len(manifest)),
        random_state=seed,
    ).reset_index(drop=True)

    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(12, 3.2 * len(rows)),
        constrained_layout=True,
    )
    axes_array = np.asarray(axes, dtype=object).reshape(len(rows), 3)

    for row_index, row in rows.iterrows():
        geo_path = _resolve_pair_path(root, row, "condition_path", "geo_path")
        sar_path = _resolve_pair_path(root, row, "target_path", "sar_path")
        center = _row_center(row, center_columns)

        geo = _read_raster_view(geo_path, geo_band)
        sar = _read_raster_view(sar_path, sar_band)

        _plot_image(
            axes_array[row_index, 0],
            geo,
            center,
            title=_panel_title(row, "GEO", geo["band_name"]),
            cmap="gray_r",
        )
        _plot_image(
            axes_array[row_index, 1],
            sar,
            center,
            title=_panel_title(row, "SAR", sar["band_name"]),
            cmap="viridis",
        )
        _plot_footprint_map(
            axes_array[row_index, 2],
            geo,
            sar,
            center,
            title=_panel_title(row, "Footprints", None),
        )

    if output_path is not None:
        fig.savefig(Path(output_path).expanduser(), dpi=dpi, bbox_inches="tight")
    return fig


def _resolve_pair_path(
    root: Path,
    row: pd.Series,
    primary_column: str,
    fallback_column: str,
) -> Path:
    column = primary_column if _has_value(row, primary_column) else fallback_column
    if not _has_value(row, column):
        raise KeyError(f"Manifest row is missing {primary_column!r}/{fallback_column!r}")
    path = Path(str(row[column])).expanduser()
    return path if path.is_absolute() else root / path


def _read_raster_view(path: Path, band: int) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        if band < 1 or band > dataset.count:
            raise ValueError(f"{path} has {dataset.count} bands, cannot read band {band}")
        data = dataset.read(band).astype(float)
        mask = dataset.dataset_mask() > 0
        data[~mask] = np.nan
        bounds = dataset.bounds
        tags = dataset.tags()
        band_name = dataset.descriptions[band - 1] or f"band {band}"

    return {
        "path": path,
        "data": data,
        "bounds": bounds,
        "tags": tags,
        "band_name": band_name,
        "extent": (bounds.left, bounds.right, bounds.bottom, bounds.top),
    }


def _plot_image(
    ax: Axes,
    raster: dict[str, Any],
    center: tuple[float, float] | None,
    *,
    title: str,
    cmap: str,
) -> None:
    ax.imshow(raster["data"], extent=raster["extent"], origin="upper", cmap=cmap)
    _plot_center(ax, center)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")


def _plot_footprint_map(
    ax: Axes,
    geo: dict[str, Any],
    sar: dict[str, Any],
    center: tuple[float, float] | None,
    *,
    title: str,
) -> None:
    lon_min, lon_max, lat_min, lat_max = _combined_bounds(geo["bounds"], sar["bounds"])
    lon_pad = max((lon_max - lon_min) * 0.12, 0.25)
    lat_pad = max((lat_max - lat_min) * 0.12, 0.25)
    lon_min -= lon_pad
    lon_max += lon_pad
    lat_min -= lat_pad
    lat_max += lat_pad

    lon = np.linspace(lon_min, lon_max, 220)
    lat = np.linspace(lat_min, lat_max, 220)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    land = globe.is_land(lat_grid, lon_grid)
    ax.imshow(
        land.astype(int),
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin="lower",
        cmap="BrBG",
        alpha=0.35,
        vmin=0,
        vmax=1,
    )
    _add_bounds(ax, geo["bounds"], "tab:blue", "GEO")
    _add_bounds(ax, sar["bounds"], "tab:orange", "SAR")
    _plot_center(ax, center)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal", adjustable="box")


def _add_bounds(ax: Axes, bounds: Any, color: str, label: str) -> None:
    width = bounds.right - bounds.left
    height = bounds.top - bounds.bottom
    patch = Rectangle(
        (bounds.left, bounds.bottom),
        width,
        height,
        fill=False,
        edgecolor=color,
        linewidth=1.8,
        label=label,
    )
    ax.add_patch(patch)


def _plot_center(ax: Axes, center: tuple[float, float] | None) -> None:
    if center is None:
        return
    lat, lon = center
    ax.scatter(
        lon,
        lat,
        marker="x",
        s=80,
        c="red",
        linewidths=2.0,
        label="storm center",
        zorder=5,
    )


def _combined_bounds(*bounds: Any) -> tuple[float, float, float, float]:
    lon_min = min(bound.left for bound in bounds)
    lon_max = max(bound.right for bound in bounds)
    lat_min = min(bound.bottom for bound in bounds)
    lat_max = max(bound.top for bound in bounds)
    return lon_min, lon_max, lat_min, lat_max


def _row_center(
    row: pd.Series,
    center_columns: tuple[str, str],
) -> tuple[float, float] | None:
    lat_column, lon_column = center_columns
    if not (_has_value(row, lat_column) and _has_value(row, lon_column)):
        return None
    lat = float(row[lat_column])
    lon = float(row[lon_column])
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return None
    return lat, lon


def _panel_title(row: pd.Series, label: str, band_name: str | None) -> str:
    sample_id = str(row["sample_id"]) if _has_value(row, "sample_id") else ""
    storm_id = str(row["storm_id"]) if _has_value(row, "storm_id") else ""
    parts = [part for part in (storm_id, sample_id) if part]
    prefix = " | ".join(parts)
    suffix = f" ({band_name})" if band_name else ""
    return f"{prefix}\n{label}{suffix}" if prefix else f"{label}{suffix}"


def _has_value(row: pd.Series, column: str) -> bool:
    return column in row.index and str(row[column]).strip() not in {"", "nan", "None"}
