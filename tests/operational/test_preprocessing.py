import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import torch

from geo2wf.data.datasets.paired_geotiff import PairedImageDataset
from geo2wf.operational.models import prepare
from geo2wf.operational.satellite import BANDS, target_grid
from scripts.export_geo_sar_geotiffs import _make_grid


def test_operational_preprocessing_matches_training_loader(tmp_path):
    lat, lon, at = 22.25, -157.5, "2026-07-02T13:52:08Z"
    rng = np.random.default_rng(25)
    array = rng.uniform(185, 315, (10, 256, 256)).astype("float32")
    array[:, 80:83, 80:84] = np.nan
    array[3, 100:110, 100:110] = np.nan
    valid = np.isfinite(array).all(axis=0)
    split = tmp_path / "test"
    split.mkdir()
    transform = from_origin(lon - 128 * 0.027, lat + 128 * 0.027, 0.027, 0.027)
    for name, values in [
        ("geo", array),
        ("sar", np.full((1, 256, 256), 20.0, dtype="float32")),
    ]:
        with rasterio.open(
            split / f"{name}.tif",
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=len(values),
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=np.nan,
        ) as destination:
            destination.write(values)
    stats = {
        "channels": {
            "geo": {
                band: {"median": 250, "robust_scale": 20, "min": 150, "max": 350}
                for band in BANDS
            },
            "sar": {"wind_speed": {"min": 0, "max": 80}},
        }
    }
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    pd.DataFrame(
        [
            dict(
                sample_id="parity",
                storm_id="CP012026",
                condition_path="test/geo.tif",
                target_path="test/sar.tif",
                condition_channels=json.dumps(BANDS),
                target_channels='["wind_speed"]',
                condition_timestamp=at,
                dt_minutes=0,
                ibtracs_center_lat=lat,
                ibtracs_center_lon=lon,
            )
        ]
    ).to_csv(split / "manifest.csv", index=False)
    reference = PairedImageDataset(
        tmp_path,
        "test",
        center_crop_size=(192, 192),
        use_era5=False,
        normalization="robust-zscore",
        target_normalization="min-max",
        robust_clip=4,
    )[0]
    actual = prepare(array, valid, lat, lon, at, stats)
    for key in actual:
        torch.testing.assert_close(actual[key][0], reference[key], rtol=1e-5, atol=2e-5)
    assert actual["condition"].shape == (1, 14, 192, 192)
    grid = target_grid(lat, lon)
    legacy = _make_grid(lat, lon, 256, 0.027)
    np.testing.assert_allclose(grid[0], legacy[0])
    np.testing.assert_allclose(grid[1], legacy[1])


def test_operational_regrid_matches_exporter():
    from geo2wf.operational.grid import regrid
    from scripts.export_geo_sar_geotiffs import _regrid

    lat, lon = np.meshgrid(
        np.linspace(20, 15, 70), np.linspace(-120, -115, 90), indexing="ij"
    )
    values = (lat + lon).astype("float32")
    values[20:23, 20:23] = np.nan
    target_lat, target_lon = target_grid(17.5, -117.5)
    actual = regrid(values, lat, lon, target_lat, target_lon)
    reference = _regrid(values, lat, lon, target_lat, target_lon)
    np.testing.assert_equal(actual[0], reference[0])
    np.testing.assert_equal(actual[1], reference[1])
