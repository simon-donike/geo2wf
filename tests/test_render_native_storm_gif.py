import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.render_native_storm_gif import (  # noqa: E402
    PMW_HIGH,
    PMW_LOW,
    PMW_MID,
    TMAX,
    TMIN,
    dense_pmw_field,
    dense_pmw_table,
    pmw_panel,
)


def test_pmw_panel_uses_purple_to_yellow_palette() -> None:
    field = np.empty((256, 256), dtype=np.float32)
    field[:, :85] = TMIN
    field[:, 85:171] = (TMIN + TMAX) / 2
    field[:, 171:] = TMAX

    image = np.asarray(pmw_panel(field, np.ones_like(field, dtype=bool)))

    assert np.array_equal(image[128, 42], PMW_LOW)
    assert np.array_equal(image[128, 128], PMW_MID)
    assert np.array_equal(image[128, 213], PMW_HIGH)


def test_dense_pmw_loader_reads_sharded_synthetic_layout(tmp_path: Path) -> None:
    storm = "EP182023"
    tensor_path = tmp_path / "observations" / storm / "frame.pt"
    tensor_path.parent.mkdir(parents=True)
    torch.save(torch.stack([torch.full((2, 3), value) for value in range(4)]), tensor_path)
    manifest_path = tmp_path / "index-files" / "shards" / f"{storm}.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp": "2023-10-25T00:00:00+00:00",
                "path": str(tensor_path.relative_to(tmp_path)),
                "variables": json.dumps(
                    ["TB_36.5H", "TB_36.5V", "TB_A89.0H", "TB_A89.0V"]
                ),
                "resolution": json.dumps([[12, 7], [12, 7], [5, 3], [5, 3]]),
                "ibtracs_center_lat": 10.0,
                "ibtracs_center_lon": -50.0,
            }
        ]
    ).to_csv(manifest_path, index=False)

    table = dense_pmw_table(tmp_path, storm)
    field, lat, lon, center = dense_pmw_field(table.iloc[0], tmp_path, storm)

    assert np.array_equal(field, np.full((2, 3), 3, dtype=np.float32))
    assert lat.shape == lon.shape == field.shape
    assert center == (10.0, -50.0)
