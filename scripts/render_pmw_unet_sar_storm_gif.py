#!/usr/bin/env python3
"""Render GEO, direct-PMW, and predicted-SAR fields on the GEO model footprint."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.render_native_storm_gif as renderer

GRID_SIZE = 256
GRID_RESOLUTION_DEGREES = 0.027
_native_geo_loader = renderer._load_geo_channels


def pmw_unet_table(root: Path, storm: str) -> pd.DataFrame:
    path = root / "dense-pmw-unet-manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    table = table[table.storm_id.astype(str) == storm].copy()
    table["parsed_time"] = pd.to_datetime(table.timestamp, utc=True)
    return table.sort_values("parsed_time").reset_index(drop=True)


def pmw_unet_field(row, root: Path, storm: str, **kwargs):
    del kwargs
    del storm
    path = root / Path(row.npz_path)
    with np.load(path) as bundle:
        field = bundle["brightness_temperature_k"].astype(np.float32)
        field[~bundle["valid_mask"].astype(bool)] = np.nan
        lat = bundle["grid_lat"].astype(np.float32)
        lon = bundle["grid_lon"].astype(np.float32)
        center = tuple(float(value) for value in bundle["source_center"])
    return field, lat, lon, center


def model_footprint_geo(record, channels):
    """Put displayed GEO channels on the direct-PMW model's full input grid."""
    native = _native_geo_loader(record, channels)
    grid_lat, grid_lon = renderer._make_grid(
        record.center[0],
        record.center[1],
        GRID_SIZE,
        GRID_RESOLUTION_DEGREES,
    )
    result = {}
    for channel in channels:
        field, valid = renderer._regrid(*native[channel], grid_lat, grid_lon)
        field = np.asarray(field, dtype=np.float32)
        field[~valid] = np.nan
        result[channel] = field, grid_lat, grid_lon
    return result


def _force_full_geo_grid() -> None:
    try:
        index = sys.argv.index("--geo-crop-size")
    except ValueError:
        sys.argv.extend(["--geo-crop-size", str(GRID_SIZE)])
    else:
        try:
            sys.argv[index + 1] = str(GRID_SIZE)
        except IndexError as error:
            raise SystemExit("--geo-crop-size requires a value") from error


def main() -> None:
    try:
        root_index = sys.argv.index("--pmw-unet-root")
        root = sys.argv[root_index + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit("--pmw-unet-root PATH is required") from error
    if "--dense-wind-root" not in sys.argv:
        raise SystemExit("--dense-wind-root PATH is required for predicted SAR")
    sys.argv[root_index : root_index + 2] = ["--dense-pmw-root", root]
    _force_full_geo_grid()
    renderer.dense_pmw_table = pmw_unet_table
    renderer.dense_pmw_field = pmw_unet_field
    renderer._load_geo_channels = model_footprint_geo
    renderer.main()


if __name__ == "__main__":
    main()
