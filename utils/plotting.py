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
from matplotlib.patches import Patch

IBTRACS_CENTER_COLUMNS = ("ibtracs_center_lat", "ibtracs_center_lon")
ERA5_WIND_SPEED_10M = "era5_wind_speed_10m"
ERA5_WIND_SPEED_RANGE_M_S = (0.0, 85.0)


def plot_random_geo_sar_pairs(
    root: str | Path,
    split: str = "train",
    n: int = 5,
    *,
    seed: int | None = None,
    geo_rgb_bands: tuple[str, str, str] = ("B13", "B14", "B08"),
    sar_band: int = 1,
    center_columns: tuple[str, str] = IBTRACS_CENTER_COLUMNS,
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
        Defaults to the IBTrACS storm center.
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
        _plot_valid_area_map(
            axes_array[row_index, 2],
            geo,
            sar,
            center,
            title=_panel_title(row, "Valid areas", None),
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


def _normalize_channel(
    channel: np.ndarray,
    mask: np.ndarray,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    if value_range is None:
        valid = channel[np.isfinite(channel) & mask]
        if valid.size == 0:
            return np.zeros_like(channel, dtype=float)
        low = float(np.nanmin(valid))
        high = float(np.nanmax(valid))
    else:
        low, high = value_range
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
    condition: Any, prediction: Any, target: Any, **kwargs: Any,
) -> Figure:
    """Build one stretched, georeferenced validation row."""
    return plot_validation_reconstruction_batch([
        {"condition": condition, "prediction": prediction, "target": target, **kwargs}
    ])


def plot_validation_reconstruction_batch(samples: list[dict[str, Any]]) -> Figure:
    """Stack up to five validation samples as identically styled figure rows."""
    if not samples:
        raise ValueError("At least one validation sample is required")
    samples = samples[:5]
    has_map = all(
        sample.get("condition_bounds") is not None
        and sample.get("target_bounds") is not None
        for sample in samples
    )
    has_era5_wind = any(
        sample.get("era5_wind_speed_physical") is not None
        or _era5_wind_speed_index(sample) is not None
        for sample in samples
    )
    panel_count = 3 + int(has_map) + int(has_era5_wind)
    fig, axes = plt.subplots(
        len(samples), panel_count,
        figsize=(4.1 * panel_count, 4 * len(samples)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, sample in enumerate(samples):
        _draw_validation_row(
            axes[row],
            sample,
            has_map=has_map,
            has_era5_wind=has_era5_wind,
        )
    return fig


def _draw_validation_row(axes, sample, *, has_map, has_era5_wind):
    condition_array = _as_chw_numpy(sample["condition"])
    prediction_array = _as_chw_numpy(sample["prediction"])
    target_array = _as_chw_numpy(sample["target"])
    condition_valid = _as_2d_mask(
        sample.get("condition_mask"), condition_array.shape[1:]
    )
    target_valid = _as_2d_mask(sample.get("target_mask"), target_array.shape[1:])
    condition_image, condition_cmap, band_name = _condition_plot_view(
        condition_array, condition_valid, sample.get("condition_channels")
    )
    # Derive the display stretch exclusively from valid ground-truth pixels,
    # then apply those same per-channel limits to prediction and target.
    output_ranges = _channel_ranges(target_array, target_valid)
    prediction_valid = np.ones(prediction_array.shape[1:], dtype=bool)
    prediction_image, prediction_cmap = _output_plot_view(
        prediction_array, prediction_valid, output_ranges
    )
    target_image, target_cmap = _output_plot_view(
        target_array, target_valid, output_ranges
    )
    condition_extent = _bounds_extent(
        sample.get("condition_bounds"), condition_array.shape[1:]
    )
    target_extent = _bounds_extent(sample.get("target_bounds"), target_array.shape[1:])
    panels = (
        (condition_image, condition_cmap, condition_valid, condition_extent,
         f"Condition ({band_name})"),
        (prediction_image, prediction_cmap, None, target_extent, "Prediction"),
        (target_image, target_cmap, target_valid, target_extent, "Target"),
    )
    center = sample.get("center")
    for ax, (image, cmap, valid, extent, title) in zip(axes, panels):
        ax.imshow(image, extent=extent, origin="upper", cmap=cmap)
        if title == "Target":
            _overlay_nodata(ax, valid, extent, alpha=1.0)
        _plot_center(ax, center)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
    if has_map:
        geo = {
            "bounds": _coerce_bounds(sample["condition_bounds"]),
            "mask": condition_valid,
        }
        output = {
            "bounds": _coerce_bounds(sample["target_bounds"]),
            "mask": target_valid,
        }
        _plot_valid_area_map(axes[3], geo, output, center, title="Valid areas")
    if has_era5_wind:
        wind_ax = axes[3 + int(has_map)]
        wind_index = _era5_wind_speed_index(sample)
        physical_wind = sample.get("era5_wind_speed_physical")
        if physical_wind is not None:
            wind_speed = _as_chw_numpy(physical_wind)[0]
            wind_valid = _as_2d_mask(
                sample.get("era5_wind_speed_mask"), wind_speed.shape
            )
        elif wind_index is not None:
            low, high = ERA5_WIND_SPEED_RANGE_M_S
            wind_speed = condition_array[wind_index] * (high - low) + low
            wind_valid = condition_valid
        else:
            wind_ax.set_axis_off()
            wind_speed = None
        if wind_speed is not None:
            _plot_era5_wind_speed_map(
                wind_ax,
                wind_speed,
                wind_valid,
                sample.get("condition_bounds"),
                center,
            )
    if sample.get("sample_label"):
        axes[0].annotate(
            sample["sample_label"], xy=(0, 1.13), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=11, annotation_clip=False,
        )

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


def _era5_wind_speed_index(sample):
    if sample.get("condition_bounds") is None:
        return None
    channels = sample.get("condition_channels")
    if not channels:
        return None
    normalized_names = [str(name).strip().lower() for name in channels]
    try:
        return normalized_names.index(ERA5_WIND_SPEED_10M)
    except ValueError:
        return None


def _plot_era5_wind_speed_map(
    ax,
    wind_speed,
    valid_mask,
    bounds,
    center,
):
    """Overlay physical ERA5 10 m wind speed on a land/ocean map."""
    extent = _bounds_extent(bounds, wind_speed.shape)
    left, right, bottom, top = extent
    lon = np.linspace(left, right, 220)
    lat = np.linspace(bottom, top, 220)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    wrapped_lon_grid = (lon_grid + 180.0) % 360.0 - 180.0
    land = globe.is_land(lat_grid, wrapped_lon_grid)
    ax.imshow(
        land.astype(int),
        extent=extent,
        origin="lower",
        cmap=ListedColormap(["#cfe8f3", "#e8e3cf"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=0,
    )

    wind_speed = np.ma.masked_where(
        ~np.asarray(valid_mask, dtype=bool), wind_speed
    )
    image = ax.imshow(
        wind_speed,
        extent=extent,
        origin="upper",
        cmap="turbo",
        alpha=0.76,
        interpolation="nearest",
        zorder=2,
    )
    ax.contour(
        lon_grid,
        lat_grid,
        land.astype(float),
        levels=[0.5],
        colors=["#414b42"],
        linewidths=0.75,
        alpha=0.9,
        zorder=3,
    )
    _plot_center(ax, center)
    ax.figure.colorbar(
        image, ax=ax, shrink=0.76, pad=0.03, label=r"Wind speed (m s$^{-1}$)"
    )
    ax.set_title("ERA5 10 m wind speed", fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.grid(
        color="white", linestyle="--", linewidth=0.6, alpha=0.65, zorder=4
    )
    ax.set_aspect("equal", adjustable="box")


def _channel_ranges(array, mask):
    """Return per-channel min/max limits from valid ground-truth pixels."""
    ranges = []
    for channel in array:
        valid = channel[np.isfinite(channel) & mask]
        ranges.append(
            (float(np.nanmin(valid)), float(np.nanmax(valid)))
            if valid.size
            else (0.0, 1.0)
        )
    return ranges


def _output_plot_view(array, mask, ranges=None):
    if ranges is None:
        ranges = [None] * array.shape[0]
    if array.shape[0] == 1:
        return _normalize_channel(array[0], mask, ranges[0]), "viridis"
    if array.shape[0] >= 3:
        return np.dstack([
            _normalize_channel(array[index], mask, ranges[index])
            for index in range(3)
        ]), None
    return _normalize_channel(array[0], mask, ranges[0]), "viridis"


def _overlay_nodata(ax, valid_mask, extent, *, alpha=0.68):
    """Shade invalid pixels gray and trace their boundary in orange."""
    if valid_mask is None:
        return
    invalid = ~np.asarray(valid_mask, dtype=bool)
    if not invalid.any():
        return
    overlay = np.ma.masked_where(~invalid, invalid.astype(float))
    ax.imshow(
        overlay, extent=extent, origin="upper",
        cmap=ListedColormap(["#8f8f8f"]), alpha=alpha, vmin=0, vmax=1,
        interpolation="nearest", zorder=3,
    )
    height, width = invalid.shape
    x = np.linspace(extent[0], extent[1], width)
    y = np.linspace(extent[3], extent[2], height)
    ax.contour(
        x, y, invalid.astype(float), levels=[0.5], colors=["#ff8c00"],
        linewidths=0.65, zorder=4,
    )


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
    left, right, bottom, top = values
    return BoundingBox(left=left, bottom=bottom, right=right, top=top)

def _plot_valid_area_map(
    ax: Axes,
    geo: dict[str, Any],
    sar: dict[str, Any],
    center: tuple[float, float] | None,
    *,
    title: str,
) -> None:
    lon_min, lon_max, lat_min, lat_max = _combined_bounds(geo["bounds"], sar["bounds"])
    # Leave enough surrounding geography visible to make the logged valid areas
    # easier to place in their broader spatial context.
    lon_pad = max((lon_max - lon_min) * 0.25, 0.25)
    lat_pad = max((lat_max - lat_min) * 0.25, 0.25)
    lon_min -= lon_pad
    lon_max += lon_pad
    lat_min -= lat_pad
    lat_max += lat_pad

    lon = np.linspace(lon_min, lon_max, 220)
    lat = np.linspace(lat_min, lat_max, 220)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    wrapped_lon_grid = (lon_grid + 180.0) % 360.0 - 180.0
    land = globe.is_land(lat_grid, wrapped_lon_grid)
    ax.imshow(
        land.astype(int),
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin="lower",
        cmap=ListedColormap(["#cfe8f3", "#e8e3cf"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=0,
    )
    ax.contour(
        lon_grid, lat_grid, land.astype(float), levels=[0.5],
        colors=["#849080"], linewidths=0.65, alpha=0.9, zorder=1,
    )
    _add_valid_area(ax, geo["mask"], geo["bounds"], "tab:blue")
    _add_valid_area(ax, sar["mask"], sar["bounds"], "tab:orange")
    _plot_center(ax, center)
    handles = [
        Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.3, label="GEO valid"),
        Patch(facecolor="tab:orange", edgecolor="tab:orange", alpha=0.3, label="SAR valid"),
    ]
    if center is not None:
        handles.append(
            plt.Line2D(
                [], [], marker="x", linestyle="none", color="red",
                markersize=8, markeredgewidth=2, label="IBTrACS center",
            )
        )
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.grid(
        color="white", linestyle="--", linewidth=0.6, alpha=0.65, zorder=1,
    )
    ax.set_axisbelow(False)
    ax.set_aspect("equal", adjustable="box")


def _add_valid_area(ax: Axes, mask: Any, bounds: Any, color: str) -> None:
    """Draw the georeferenced valid-pixel region as filled polygons."""
    valid = np.asarray(mask, dtype=bool).squeeze()
    if valid.ndim != 2:
        raise ValueError(f"Valid-area mask must be 2D, got shape {valid.shape}")
    if not valid.any():
        return
    height, width = valid.shape
    pixel_width = (bounds.right - bounds.left) / width
    pixel_height = (bounds.top - bounds.bottom) / height
    x = np.linspace(
        bounds.left - pixel_width / 2, bounds.right + pixel_width / 2, width + 2,
    )
    y = np.linspace(
        bounds.top + pixel_height / 2, bounds.bottom - pixel_height / 2, height + 2,
    )
    polygon_mask = np.pad(valid, 1, constant_values=False).astype(float)
    ax.contourf(
        x, y, polygon_mask, levels=[0.5, 1.5],
        colors=[color], alpha=0.3, zorder=2,
    )
    ax.contour(
        x, y, polygon_mask, levels=[0.5],
        colors=[color], linewidths=1.5, zorder=3,
    )


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
        label="IBTrACS center",
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
