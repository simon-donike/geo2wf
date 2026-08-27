from pathlib import Path

import pandas as pd

from scripts.run_dense_pmw_unet_inference import dense_manifest, update_storm_table


def test_dense_manifest_finds_sharded_synthetic_layout(tmp_path: Path) -> None:
    manifest = tmp_path / "index-files" / "shards" / "EP182023.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("timestamp\n2023-10-25T00:00:00Z\n")

    assert dense_manifest(tmp_path, "EP182023") == manifest


def test_update_storm_table_preserves_other_storms(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"storm_id": "AL082025", "value": 1},
            {"storm_id": "EP182023", "value": 2},
        ]
    ).to_csv(path, index=False)

    update_storm_table(path, "EP182023", [{"storm_id": "EP182023", "value": 3}])

    assert pd.read_csv(path).to_dict("records") == [
        {"storm_id": "AL082025", "value": 1},
        {"storm_id": "EP182023", "value": 3},
    ]
