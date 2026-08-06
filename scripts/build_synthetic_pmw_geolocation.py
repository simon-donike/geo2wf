#!/usr/bin/env python3
"""Materialize explicit per-pixel geolocation for synthetic PMW tensors."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_geo_sar_geotiffs import _read_manifest  # noqa: E402
from render_native_storm_gif import (  # noqa: E402
    dense_pmw_table,
    synthetic_pmw_grid,
)

DEFAULT_DATA_ROOT = ROOT / "inference" / "inf_data"
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "index-files" / "observation_manifest_v6.csv"
DEFAULT_SYNTHETIC_ROOT = ROOT / "inference" / "synthetic_pmw"
GEOMETRY_METHOD = "template-real-swath-aeqd-rigid-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover the referenced real-swath geometry and write an explicit "
            "grid_lat/grid_lon sidecar for every synthetic PMW tensor."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--synthetic-root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to SYNTHETIC_ROOT/geolocation.",
    )
    parser.add_argument("--storm", action="append", dest="storms")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def available_storms(root: Path) -> list[str]:
    shard_dir = root / "index-files" / "shards"
    return sorted(path.stem.upper() for path in shard_dir.glob("*.csv"))


def sidecar_path(output_root: Path, storm: str, tensor_path: str) -> Path:
    return output_root / storm / f"{Path(tensor_path).stem}.npz"


def write_sidecar(
    path: Path,
    *,
    row,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    target_center: tuple[float, float],
    template_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        grid_lat=np.asarray(grid_lat, dtype=np.float32),
        grid_lon=np.asarray(grid_lon, dtype=np.float32),
        valid_mask=np.isfinite(grid_lat) & np.isfinite(grid_lon),
        observation_id=np.asarray(str(row.observation_id)),
        template_observation_id=np.asarray(template_id),
        target_center=np.asarray(target_center, dtype=np.float64),
        timestamp=np.asarray(str(row.timestamp)),
        geometry_method=np.asarray(GEOMETRY_METHOD),
        crs=np.asarray("EPSG:4326"),
    )
    os.replace(temporary, path)


def build_storm(
    storm: str,
    *,
    synthetic_root: Path,
    output_root: Path,
    records_by_id: dict,
    force: bool,
) -> dict:
    table = dense_pmw_table(synthetic_root, storm)
    cache: dict = {}
    written = 0
    reused = 0
    template_ids: set[str] = set()
    for row in tqdm(table.itertuples(index=False), total=len(table), desc=storm):
        output = sidecar_path(output_root, storm, row.path)
        if output.is_file() and not force:
            reused += 1
            continue
        grid_lat, grid_lon, center, template_id = synthetic_pmw_grid(
            row, records_by_id, cache
        )
        expected_shape = (int(row.height), int(row.width))
        if grid_lat.shape != expected_shape or grid_lon.shape != expected_shape:
            raise ValueError(
                f"{row.observation_id}: template grid {grid_lat.shape}/{grid_lon.shape} "
                f"does not match declared tensor shape {expected_shape}"
            )
        write_sidecar(
            output,
            row=row,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
            target_center=center,
            template_id=template_id,
        )
        template_ids.add(template_id)
        written += 1
    summary = {
        "storm_id": storm,
        "geometry_method": GEOMETRY_METHOD,
        "crs": "EPSG:4326",
        "sidecar_count": len(table),
        "written_count": written,
        "reused_count": reused,
        "template_count": len(template_ids) if written else None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output_root / storm / "geolocation-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    synthetic_root = args.synthetic_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else synthetic_root / "geolocation"
    )
    records = _read_manifest(args.manifest, args.data_root)
    records_by_id = {record.observation_id: record for record in records}
    storms = [value.upper() for value in args.storms] if args.storms else available_storms(synthetic_root)
    for storm in storms:
        summary = build_storm(
            storm,
            synthetic_root=synthetic_root,
            output_root=output_root,
            records_by_id=records_by_id,
            force=args.force,
        )
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
