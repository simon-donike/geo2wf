from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import from_origin
from torch.utils.data import DataLoader

from data.datamodule import PairedDataModule
from data.dataset import PairedImageDataset


def _write_tiff(path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[-2],
        width=values.shape[-1],
        count=values.shape[0],
        dtype="float32",
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        crs="EPSG:4326",
    ) as destination:
        destination.write(values.astype(np.float32))


def _dataset_root(tmp_path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    _write_tiff(split / "geo.tif", np.full((1, 2, 2), 250.0))
    _write_tiff(split / "sar.tif", np.full((1, 2, 2), 20.0))
    _write_tiff(split / "pmw.tif", np.full((1, 2, 2), 275.0))
    rows = []
    for index in range(3):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "storm_id": "AL012024",
                "condition_path": "train/geo.tif",
                "target_path": "train/sar.tif",
                "pmw_path": "train/pmw.tif" if index < 2 else "",
                "condition_source_type": "geo",
                "target_source_type": "sar",
                "condition_channels": json.dumps(["CMI_C13"]),
                "target_channels": json.dumps(["wind_speed"]),
                "pmw_channels": json.dumps(["brightness_temperature"]),
                "condition_timestamp": "2025-06-21T12:00:00Z",
                "dt_minutes": 0.0,
                "pmw_dt_minutes": 15.0,
                "ibtracs_center_lat": 1.0,
                "ibtracs_center_lon": 1.0,
                "ibtracs_name": "ALBERTO",
                "ibtracs_usa_wind": "45" if index == 0 else "",
            }
        )
    pd.DataFrame(rows).to_csv(split / "manifest.csv", index=False)
    stats = {
        "channels": {
            "geo": {"CMI_C13": {"min": 200.0, "max": 300.0}},
            "sar": {"wind_speed": {"min": 0.0, "max": 80.0}},
            "pmw": {
                "brightness_temperature": {"min": 200.0, "max": 300.0}
            },
        }
    }
    (tmp_path / "stats.json").write_text(json.dumps(stats), encoding="utf-8")


def test_optional_companions_are_disabled_by_default(tmp_path) -> None:
    _dataset_root(tmp_path)

    dataset = PairedImageDataset(tmp_path, "train", target_size=(2, 2))
    sample = dataset[0]

    assert len(dataset) == 3
    assert "pmw" not in sample
    assert "pmw_channels" not in sample["meta"]
    assert "ibtracs" not in sample


def test_enabled_pmw_and_ibtracs_collate_into_stable_batches(tmp_path) -> None:
    _dataset_root(tmp_path)
    dataset = PairedImageDataset(
        tmp_path,
        "train",
        target_size=(2, 2),
        include_pmw=True,
        include_ibtracs=True,
    )

    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    assert len(dataset) == 2
    assert dataset.filtered_missing_pmw_count == 1
    assert batch["pmw"].shape == (2, 1, 2, 2)
    assert torch.allclose(batch["pmw"], torch.full((2, 1, 2, 2), 0.75))
    assert torch.allclose(batch["pmw_physical"], torch.full((2, 1, 2, 2), 275.0))
    assert batch["pmw_mask"].all()
    assert batch["meta"]["pmw_sensor"] == ["", ""]
    assert batch["ibtracs"]["ibtracs_name"] == ["ALBERTO", "ALBERTO"]
    assert batch["ibtracs"]["ibtracs_usa_wind"][0] == 45.0
    assert torch.isnan(batch["ibtracs"]["ibtracs_usa_wind"][1])


def test_datamodule_reads_optional_companion_flags_from_data_config() -> None:
    datamodule = PairedDataModule.from_config(
        {
            "data": {
                "root": "custom/data",
                "include_pmw": True,
                "include_ibtracs": True,
            }
        }
    )

    assert datamodule.include_pmw is True
    assert datamodule.include_ibtracs is True
