from pathlib import Path

import numpy as np
import pandas as pd

from scripts.render_pmw_unet_storm_gif import pmw_unet_field, pmw_unet_table
from scripts.run_geo_pmw_unet_inference import _full_bounds


def test_full_bounds_preserve_direct_unet_training_size() -> None:
    bounds = _full_bounds(10.0, -50.0)

    assert np.allclose(bounds.numpy(), [-53.456, -46.544, 6.544, 13.456])


def test_pmw_unet_gif_loader_reads_saved_prediction(tmp_path: Path) -> None:
    storm = "EP182023"
    bundle_path = tmp_path / storm / "pmw-fields" / "frame.npz"
    bundle_path.parent.mkdir(parents=True)
    np.savez_compressed(
        bundle_path,
        brightness_temperature_k=np.full((2, 3), 275.0, dtype=np.float32),
        valid_mask=np.array([[1, 1, 0], [1, 1, 1]], dtype=np.uint8),
        grid_lat=np.full((2, 3), 10.0, dtype=np.float32),
        grid_lon=np.full((2, 3), -50.0, dtype=np.float32),
        source_center=np.array([10.0, -50.0], dtype=np.float32),
    )
    pd.DataFrame(
        [
            {
                "storm_id": storm,
                "timestamp": "2023-10-25T00:00:00Z",
                "npz_path": str(bundle_path.relative_to(tmp_path)),
            }
        ]
    ).to_csv(tmp_path / "dense-pmw-unet-manifest.csv", index=False)

    table = pmw_unet_table(tmp_path, storm)
    field, lat, lon, center = pmw_unet_field(table.iloc[0], tmp_path, storm)

    assert field.shape == lat.shape == lon.shape == (2, 3)
    assert np.isnan(field[0, 2])
    assert center == (10.0, -50.0)
