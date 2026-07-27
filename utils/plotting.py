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
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


def plot_random_geo_sar_pairs(
    root: str | Path,
    split: str = "train",
    n: int = 5,
    *,
    seed: int | None = None,
    geo_rgb_bands: tuple[str, str, str] = ("B13", "B14", "B08"),
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
    geo_rgb_bands:
        Band descriptions to use as red, green, and blue for the GEO false-color
        panel. Defaults to two IR bands as red/green and water vapor as blue.
        If a requested ``Bxx`` band is not present, the matching ``Cxx`` band is
        tried automatically, and vice versa.
    sar_band:
        One-based GeoTIFF band index to display for the SAR target.
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

        geo = _read_geo_false_color_view(geo_path, geo_rgb_bands)
        sar = _read_raster_view(sar_path, sar_band)

        _plot_image(
            axes_array[row_index, 0],
            geo,
            center,
            title=_panel_title(row, "GEO false color", geo["band_name"]),
            cmap=None,
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
        mask &= np.isfinite(data)
        data[~mask] = np.nan
        bounds = dataset.bounds
        tags = dataset.tags()
        band_name = dataset.descriptions[band - 1] or f"band {band}"

    return {
        "path": path,
        "data": data,
        "mask": mask,
        "bounds": bounds,
        "tags": tags,
        "band_name": band_name,
        "extent": (bounds.left, bounds.right, bounds.bottom, bounds.top),
    }


def _read_geo_false_color_view(
    path: Path,
    rgb_bands: tuple[str, str, str],
) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        descriptions = [
            description or f"band {index}"
            for index, description in enumerate(dataset.descriptions, start=1)
        ]
        band_indices = [_find_band_index(descriptions, band) for band in rgb_bands]
        channels = [dataset.read(index).astype(float) for index in band_indices]
        mask = dataset.dataset_mask() > 0
        mask &= np.logical_and.reduce([np.isfinite(channel) for channel in channels])
        mask &= ~np.logical_and.reduce([np.isclose(channel, 0.0) for channel in channels])
        bounds = dataset.bounds
        tags = dataset.tags()

    rgb = np.dstack([_normalize_channel(channel, mask) for channel in channels])
    rgb[~mask] = 0.0
    band_name = "/".join(
        descriptions[index - 1] for index in band_indices
    )
    return {
        "path": path,
        "data": rgb,
        "mask": mask,
        "bounds": bounds,
        "tags": tags,
        "band_name": band_name,
        "extent": (bounds.left, bounds.right, bounds.bottom, bounds.top),
    }


def _find_band_index(descriptions: list[str], requested: str) -> int:
    candidates = [requested]
    if requested.startswith("B"):
        candidates.append(f"C{requested[1:]}")
    elif requested.startswith("C"):
        candidates.append(f"B{requested[1:]}")

    normalized = {
        _normalize_band_name(description): index
        for index, description in enumerate(descriptions, start=1)
    }
    for candidate in candidates:
        index = normalized.get(_normalize_band_name(candidate))
        if index is not None:
            return index
    raise ValueError(
        f"Could not find GEO band {requested!r}; available bands are {descriptions}"
    )


def _normalize_band_name(name: str) -> str:
    normalized = name.upper()
    for prefix in ("CMI_", "ABI_", "AHI_"):
        normalized = normalized.removeprefix(prefix)
    return normalized


def _normalize_channel(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = channel[np.isfinite(channel) & mask]
    if valid.size == 0:
        return np.zeros_like(channel, dtype=float)
    low = float(np.nanmin(valid))
    high = float(np.nanmax(valid))
    scale = max(high - low, 1e-6)
    return np.clip((channel - low) / scale, 0.0, 1.0)


def _plot_image(
    ax: Axes,
    raster: dict[str, Any],
    center: tuple[float, float] | None,
    *,
    title: str,
    cmap: str | None,
) -> None:
    image = raster["data"]
    ax.imshow(image, extent=raster["extent"], origin="upper", cmap=cmap)
    _overlay_nodata(ax, raster.get("mask"), raster["extent"])
    _plot_center(ax, center)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")



def plot_validation_reconstruction(
    condition: Any, prediction: Any, target: Any, *,
    condition_mask: Any | None = None, target_mask: Any | None = None,
    condition_channels: list[str] | tuple[str, ...] | None = None,
    condition_bounds: Any | None = None, target_bounds: Any | None = None,
    center: tuple[float, float] | None = None, sample_label: str = "",
) -> Figure:
    """Build a stretched, georeferenced validation figure for W&B."""
    condition_array = _as_chw_numpy(condition)
    prediction_array = _as_chw_numpy(prediction)
    target_array = _as_chw_numpy(target)
    condition_valid = _as_2d_mask(condition_mask, condition_array.shape[1:])
    target_valid = _as_2d_mask(target_mask, target_array.shape[1:])
    condition_image, condition_cmap, band_name = _condition_plot_view(
        condition_array, condition_valid, condition_channels
    )
    prediction_image, prediction_cmap = _output_plot_view(prediction_array, target_valid)
    target_image, target_cmap = _output_plot_view(target_array, target_valid)
    condition_extent = _bounds_extent(condition_bounds, condition_array.shape[1:])
    target_extent = _bounds_extent(target_bounds, target_array.shape[1:])
    has_map = condition_bounds is not None and target_bounds is not None
    panel_count = 4 if has_map else 3
    fig, axes = plt.subplots(1, panel_count, figsize=(4.1 * panel_count, 4), constrained_layout=True)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    panels = (
        (condition_image, condition_cmap, condition_valid, condition_extent, f"Condition ({band_name})"),
        (prediction_image, prediction_cmap, target_valid, target_extent, "Prediction"),
        (target_image, target_cmap, target_valid, target_extent, "Target"),
    )
    for ax, (image, cmap, valid, extent, title) in zip(axes, panels):
        ax.imshow(image, extent=extent, origin="upper", cmap=cmap)
        _overlay_nodata(ax, valid, extent)
        _plot_center(ax, center)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
    if has_map:
        geo = {"bounds": _coerce_bounds(condition_bounds)}
        output = {"bounds": _coerce_bounds(target_bounds)}
        _plot_footprint_map(axes[3], geo, output, center, title="Footprints")
    if sample_label:
        fig.suptitle(sample_label, fontsize=11)
    return fig


def _condition_plot_view(array, mask, channels):
    names = [str(name) for name in channels] if channels else []
    if array.shape[0] >= 3:
        indices = list(range(3))
        if names:
            try:
                indices = [_find_band_index(names, band) - 1 for band in ("B13", "B14", "B08")]
            except ValueError:
                pass
        image = np.dstack([_normalize_channel(array[index], mask) for index in indices])
        band_name = "/".join(names[index] if index < len(names) else f"band {index + 1}" for index in indices)
        return image, None, band_name
    return _normalize_channel(array[0], mask), "viridis", names[0] if names else "band 1"


def _output_plot_view(array, mask):
    if array.shape[0] == 1:
        return _normalize_channel(array[0], mask), "viridis"
    if array.shape[0] >= 3:
        return np.dstack([_normalize_channel(array[index], mask) for index in range(3)]), None
    return _normalize_channel(array[0], mask), "viridis"


def _overlay_nodata(ax, valid_mask, extent):
    """Shade invalid pixels gray and trace their boundary in orange."""
    if valid_mask is None:
        return
    invalid = ~np.asarray(valid_mask, dtype=bool)
    if not invalid.any():
        return
    overlay = np.ma.masked_where(~invalid, invalid.astype(float))
    ax.imshow(overlay, extent=extent, origin="upper", cmap=ListedColormap(["#8f8f8f"]), alpha=0.68, vmin=0, vmax=1, interpolation="nearest", zorder=3)
    height, width = invalid.shape
    x = np.linspace(extent[0], extent[1], width)
    y = np.linspace(extent[3], extent[2], height)
    ax.contour(x, y, invalid.astype(float), levels=[0.5], colors=["#ff8c00"], linewidths=0.65, zorder=4)


def _as_chw_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=float)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected CHW or HW image, got shape {array.shape}")
    return array


def _as_2d_mask(value, shape):
    if value is None:
        return np.ones(shape, dtype=bool)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    mask = np.asarray(value).astype(bool).squeeze()
    if mask.shape != shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {shape}")
    return mask


def _bounds_extent(bounds, shape):
    if bounds is None:
        height, width = shape
        return 0.0, float(width), 0.0, float(height)
    bound = _coerce_bounds(bounds)
    return bound.left, bound.right, bound.bottom, bound.top


def _coerce_bounds(bounds):
    if hasattr(bounds, "left"):
        return bounds
    values = np.asarray(bounds, dtype=float).reshape(-1)
    if values.size != 4:
        raise ValueError("Bounds must contain left, right, bottom, top")
    from rasterio.coords import BoundingBox
    return BoundingBox(*values)

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
    storm_id = str(row["storm_id"]) if _has_value(row, "storm_id") else ""
    suffix = f" ({band_name})" if band_name else ""
    return f"{storm_id}\n{label}{suffix}" if storm_id else f"{label}{suffix}"


def _has_value(row: pd.Series, column: str) -> bool:
    return column in row.index and str(row[column]).strip() not in {"", "nan", "None"}
