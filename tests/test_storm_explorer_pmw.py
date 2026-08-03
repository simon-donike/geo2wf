import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_storm_explorer_data import export_pmw_array  # noqa: E402


def test_pmw_export_uses_shared_scale_and_transparent_no_data(tmp_path) -> None:
    lat = np.array([[10.0, 10.0], [11.0, 11.0]], dtype=np.float32)
    lon = np.array([[-50.0, -49.0], [-50.0, -49.0]], dtype=np.float32)
    first = np.array([[150.0, 225.0], [300.0, np.nan]], dtype=np.float32)
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
    assert one["max"] == 300.0
    assert one["bounds"] == [[10.0, -50.0], [11.0, -49.0]]
    assert two["sensor"] == "AMSR2_GCOMW1"
