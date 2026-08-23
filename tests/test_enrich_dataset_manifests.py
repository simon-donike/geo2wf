from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from scripts.enrich_dataset_manifests import enrich_manifests


def _write_raster(
    path: Path, values: np.ndarray, *, nodata: float | None = np.nan
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[-2],
        width=values.shape[-1],
        count=values.shape[0],
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(0.0, 4.0, 1.0, 1.0),
        nodata=nodata,
    ) as destination:
        destination.write(values.astype("float32"))


def test_enrichment_preserves_rows_and_adds_physical_metrics(tmp_path: Path) -> None:
    root = tmp_path / "export"
    target_values = np.array(
        [[[np.nan, 10.0], [20.0, 40.0]]],
        dtype=np.float32,
    )
    context_values = np.zeros((7, 2, 2), dtype=np.float32)
    context_values[5] = 3.0
    context_values[6] = 4.0
    _write_raster(root / "train" / "target.tif", target_values)
    _write_raster(root / "train" / "context.tif", context_values)
    rows = pd.DataFrame(
        [
            {
                "sample_id": "storm_a_0",
                "storm_id": "storm_a",
                "target_path": "train/target.tif",
                "context_path": "train/context.tif",
                "context_channels": json.dumps(
                    [
                        "precipitable_water",
                        "sst",
                        "pressure_msl",
                        "temperature_2m",
                        "dewpoint_2m",
                        "u_wind_10m",
                        "v_wind_10m",
                    ]
                ),
                "target_timestamp": "2025-01-01T00:00:00Z",
                "era5_timestamp": "2025-01-01T00:00:00Z",
                "center_lat": 2.5,
                "center_lon": 1.5,
                "ibtracs_center_lat": 2.5,
                "ibtracs_center_lon": 1.5,
            }
        ]
    )
    (root / "train").mkdir(parents=True, exist_ok=True)
    rows.to_csv(root / "train" / "manifest.csv", index=False)

    summary = enrich_manifests(root)
    output = pd.read_csv(root / "train" / "manifest.csv")

    assert len(output) == 1
    assert output.loc[0, "sample_id"] == "storm_a_0"
    assert output.loc[0, "sar_valid_pixels"] == 3
    assert output.loc[0, "sar_max_wind_ms"] == 40.0
    assert output.loc[0, "sar_robust_peak_ms"] == 40.0
    assert output.loc[0, "era5_max_wind_ms"] == 5.0
    assert output.loc[0, "sar_minus_era5_max_ms"] == 35.0
    assert bool(output.loc[0, "sar_has_valid_center"])
    assert output.loc[0, "metadata_schema_version"] == 1
    assert summary["manifests"]["train/manifest.csv"]["samples"] == 1
    assert (root / "manifest-metadata-summary.json").is_file()


def test_enrichment_is_idempotent_for_derived_columns(tmp_path: Path) -> None:
    root = tmp_path / "export"
    target = np.ones((1, 4, 4), dtype=np.float32) * 20.0
    context = np.zeros((7, 4, 4), dtype=np.float32)
    context[5] = 3.0
    context[6] = 4.0
    _write_raster(root / "train" / "target.tif", target)
    _write_raster(root / "train" / "context.tif", context)
    frame = pd.DataFrame(
        {
            "sample_id": ["storm_a_0"],
            "storm_id": ["storm_a"],
            "target_path": ["train/target.tif"],
            "context_path": ["train/context.tif"],
            "context_channels": [json.dumps(["x", "x", "x", "x", "x", "u10", "v10"])],
            "target_timestamp": ["2025-01-01T00:00:00Z"],
            "era5_timestamp": ["2025-01-01T00:00:00Z"],
            "center_lat": [2.0],
            "center_lon": [2.0],
            "ibtracs_center_lat": [2.0],
            "ibtracs_center_lon": [2.0],
        }
    )
    (root / "train").mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "train" / "manifest.csv", index=False)

    enrich_manifests(root)
    first = pd.read_csv(root / "train" / "manifest.csv")
    enrich_manifests(root)
    second = pd.read_csv(root / "train" / "manifest.csv")

    pd.testing.assert_frame_equal(first, second, check_dtype=False)
