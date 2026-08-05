import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_storm_explorer_data as explorer  # noqa: E402

from export_geostat_images import GEOSTAT_SCALE_MAX_K, GEOSTAT_SCALE_MIN_K  # noqa: E402
from export_storm_explorer_data import (  # noqa: E402
    PMW_COLOR_HIGH,
    PMW_COLOR_LOW,
    PMW_COLOR_MID,
    PMW_SCALE_MAX_K,
    PMW_SCALE_MIN_K,
    export_pmw_array,
)


def test_pmw_export_uses_shared_scale_and_transparent_no_data(tmp_path) -> None:
    lat = np.array([[10.0, 10.0], [11.0, 11.0]], dtype=np.float32)
    lon = np.array([[-50.0, -49.0], [-50.0, -49.0]], dtype=np.float32)
    first = np.array([[150.0, 225.0], [np.nan, np.nan]], dtype=np.float32)
    second = np.array([[225.0, 275.0], [200.0, 175.0]], dtype=np.float32)

    one = export_pmw_array(
        "first",
        first,
        np.isfinite(first),
        lat,
        lon,
        "GMI_GPM",
        "TB_89.0V",
        tmp_path,
    )
    two = export_pmw_array(
        "second",
        second,
        np.isfinite(second),
        lat,
        lon,
        "AMSR2_GCOMW1",
        "TB_A89.0V",
        tmp_path,
    )

    first_image = iio.imread(tmp_path / "first.png")
    second_image = iio.imread(tmp_path / "second.png")
    assert first_image[1, 1, 3] == 0
    assert np.all(first_image[np.isfinite(first), 3] == 225)
    assert np.array_equal(first_image[0, 1, :3], second_image[0, 0, :3])
    assert one["min"] == 150.0
    assert one["max"] == 225.0
    assert one["bounds"] == [[10.0, -50.0], [11.0, -49.0]]
    assert two["sensor"] == "AMSR2_GCOMW1"


def test_pmw_uses_fixed_brightness_temperature_bounds() -> None:
    assert (PMW_SCALE_MIN_K, PMW_SCALE_MAX_K) == (
        GEOSTAT_SCALE_MIN_K,
        GEOSTAT_SCALE_MAX_K,
    )


def test_pmw_uses_a_distinct_color_map(tmp_path) -> None:
    values = np.array(
        [
            [PMW_SCALE_MIN_K, (PMW_SCALE_MIN_K + PMW_SCALE_MAX_K) / 2, PMW_SCALE_MAX_K],
            [PMW_SCALE_MIN_K, (PMW_SCALE_MIN_K + PMW_SCALE_MAX_K) / 2, PMW_SCALE_MAX_K],
        ],
        dtype=np.float32,
    )
    coordinates = np.ones_like(values)

    export_pmw_array(
        "palette",
        values,
        np.ones_like(values, dtype=bool),
        coordinates,
        coordinates,
        "GMI_GPM",
        "TB_89.0V",
        tmp_path,
    )

    image = iio.imread(tmp_path / "palette.png")
    assert np.array_equal(image[0, 0, :3], PMW_COLOR_LOW)
    assert np.array_equal(image[0, 1, :3], PMW_COLOR_MID)
    assert np.array_equal(image[0, 2, :3], PMW_COLOR_HIGH)
    assert not np.array_equal(image[0, 0, :3], [255, 255, 255])


def test_manifest_storm_ids_are_the_dashboard_source_of_truth(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"storm_id": ["EP182023", "AL082025", "EP182023", None]}).to_csv(
        manifest, index=False
    )
    monkeypatch.setattr(explorer, "RAW_MANIFEST", manifest)

    assert explorer.manifest_storm_ids() == ["AL082025", "EP182023"]


def test_nwp_export_reads_the_selected_storm_directory(tmp_path, monkeypatch) -> None:
    nwp_root = tmp_path / "NWP"
    storm_root = nwp_root / "EP182023"
    storm_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "valid_time": ["2023-10-24T00:00:00+00:00"],
            "max_wind_ms": [25.1254],
        }
    ).to_csv(storm_root / "gfs.csv", index=False)
    monkeypatch.setattr(explorer, "NWP_ROOT", nwp_root)

    assert explorer.export_nwp("EP182023") == [
        {
            "id": "gfs",
            "label": "GFS",
            "points": [{"time": "2023-10-24T00:00:00Z", "max": 25.125}],
        }
    ]


def test_intensity_prediction_exports_corrected_scalar_and_category() -> None:
    table = pd.DataFrame(
        {
            "observation_id": ["storm:geo:one"],
            "output_msw_ms": [42.1236],
            "output_category": [2],
            "raw_unet_max_wind_ms": [37.0],
            "correction_ms": [5.1236],
        }
    ).set_index("observation_id")
    assert explorer.intensity_prediction(table, "storm:geo:one") == {
        "max": 42.124,
        "category": 2,
        "raw_unet_max_wind_ms": 37.0,
        "correction_ms": 5.124,
    }
    assert explorer.intensity_prediction(table, "missing") is None
