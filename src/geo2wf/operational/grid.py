"""Nearest-neighbor grid used by the checkpoint training exporter."""

import numpy as np
from scipy.spatial import cKDTree


def regrid(
    values: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(values) & np.isfinite(src_lat) & np.isfinite(src_lon)
    if not ok.any():
        return np.full(grid_lat.shape, np.nan, np.float32), np.zeros(
            grid_lat.shape, bool
        )
    coslat = np.cos(np.deg2rad(np.nanmean(grid_lat)))
    tree = cKDTree(np.column_stack([src_lat[ok], src_lon[ok] * coslat]))
    dist, idx = tree.query(
        np.column_stack([grid_lat.ravel(), grid_lon.ravel() * coslat])
    )
    spacing = _source_spacing(src_lat, src_lon, coslat)
    result = values[ok][idx].astype(np.float32)
    mask = dist <= 1.5 * spacing
    result[~mask] = np.nan
    return result.reshape(grid_lat.shape), mask.reshape(grid_lat.shape)


def _source_spacing(src_lat: np.ndarray, src_lon: np.ndarray, coslat: float) -> float:
    spacings = []
    if src_lat.shape[-1] > 1:
        spacings.append(
            np.hypot(np.diff(src_lat, axis=-1), np.diff(src_lon, axis=-1) * coslat)
        )
    if src_lat.ndim > 1 and src_lat.shape[-2] > 1:
        spacings.append(
            np.hypot(np.diff(src_lat, axis=-2), np.diff(src_lon, axis=-2) * coslat)
        )
    finite_parts = [
        spacing[np.isfinite(spacing)].ravel()
        for spacing in spacings
        if np.isfinite(spacing).any()
    ]
    if not finite_parts:
        return 0.05
    finite = np.concatenate(finite_parts)
    if finite.size == 0:
        return 0.05
    spacing = float(np.nanmedian(finite))
    return spacing if spacing > 0 else 0.05
