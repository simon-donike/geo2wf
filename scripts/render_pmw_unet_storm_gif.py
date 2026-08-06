#!/usr/bin/env python3
"""Render the native storm GIF with direct-U-Net fields in the PMW panel."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.render_native_storm_gif as renderer


def pmw_unet_table(root: Path, storm: str) -> pd.DataFrame:
    path = root / "dense-pmw-unet-manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    table = table[table.storm_id.astype(str) == storm].copy()
    table["parsed_time"] = pd.to_datetime(table.timestamp, utc=True)
    return table.sort_values("parsed_time").reset_index(drop=True)


def pmw_unet_field(
    row,
    root: Path,
    storm: str,
    records_by_id=None,
    geometry_cache=None,
    geolocation_root=None,
):
    del storm, records_by_id, geometry_cache, geolocation_root
    path = root / Path(row.npz_path)
    with np.load(path) as bundle:
        field = bundle["brightness_temperature_k"].astype(np.float32)
        field[~bundle["valid_mask"].astype(bool)] = np.nan
        lat = bundle["grid_lat"].astype(np.float32)
        lon = bundle["grid_lon"].astype(np.float32)
        center = tuple(float(value) for value in bundle["source_center"])
    return field, lat, lon, center


def main() -> None:
    try:
        root_index = sys.argv.index("--pmw-unet-root")
        root = sys.argv[root_index + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit("--pmw-unet-root PATH is required") from error
    sys.argv[root_index : root_index + 2] = ["--dense-pmw-root", root]
    renderer.dense_pmw_table = pmw_unet_table
    renderer.dense_pmw_field = pmw_unet_field
    renderer.main()


if __name__ == "__main__":
    main()
