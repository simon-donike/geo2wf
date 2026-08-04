from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from export_geo_pmw_geotiffs import (  # noqa: E402
    PMW_CANONICAL_CHANNEL,
    PMW_CHANNELS,
    PMW_SOURCE_CHANNELS,
    PMW_SWATHS,
    _regrid_linear_supported,
    _update_stats,
)
from export_geo_sar_geotiffs import StatsAccumulator  # noqa: E402
from geo2wf.config import compose_config, instantiate_model
from geo2wf.data.contracts import DataSpec
from geo2wf.data.datasets.paired_geotiff import PairedImageDataset
from geo2wf.models.base import PredictionRequest
from geo2wf.models.direct_unet import DirectUNetRegressor


EXPECTED = {
    "AMSR2_GCOMW1": ("TB_A89.0V", "S5"),
    "GMI_GPM": ("TB_89.0V", "S1"),
    "SSMIS_F16": ("TB_91.665V", "S4"),
    "SSMIS_F17": ("TB_91.665V", "S4"),
    "SSMIS_F18": ("TB_91.665V", "S4"),
    "ATMS_NPP": ("TB_88.2QV", "S3"),
    "ATMS_NOAA20": ("TB_88.2QV", "S3"),
    "ATMS_NOAA21": ("TB_88.2QV", "S3"),
    "MHS_METOPB": ("TB_89.0V", "S1"),
    "MHS_METOPC": ("TB_89.0V", "S1"),
    "MHS_NOAA19": ("TB_89.0V", "S1"),
}


def test_near89_sensor_mappings_share_one_canonical_target() -> None:
    assert set(PMW_SOURCE_CHANNELS) == set(EXPECTED)
    for sensor, (source_channel, swath) in EXPECTED.items():
        assert PMW_SOURCE_CHANNELS[sensor] == source_channel
        assert PMW_SWATHS[sensor] == swath
        assert PMW_CHANNELS[sensor] == (PMW_CANONICAL_CHANNEL,)


def test_canonical_target_statistics_pool_platform_samples() -> None:
    stats = StatsAccumulator.create()
    mask = np.ones((1, 2, 2), dtype=bool)
    _update_stats(
        stats, "pmw", [PMW_CANONICAL_CHANNEL], np.full((1, 2, 2), 240.0), mask
    )
    _update_stats(
        stats, "pmw", [PMW_CANONICAL_CHANNEL], np.full((1, 2, 2), 300.0), mask
    )

    channel = stats.to_jsonable()["channels"]["pmw"][PMW_CANONICAL_CHANNEL]
    assert channel["min"] == 240.0
    assert channel["max"] == 300.0
    assert channel["count"] == 8


def test_linear_swath_interpolation_reproduces_plane_and_masks_extrapolation() -> None:
    lat, lon = np.meshgrid(np.arange(3.0), np.arange(3.0), indexing="ij")
    values = 2.0 * lat + 3.0 * lon + 250.0
    grid_lat, grid_lon = np.meshgrid(
        np.array([0.5, 1.5, 3.0]), np.array([0.5, 1.5, 3.0]), indexing="ij"
    )

    result, mask = _regrid_linear_supported(values, lat, lon, grid_lat, grid_lon)

    assert mask[:2, :2].all()
    assert np.allclose(
        result[:2, :2], 2.0 * grid_lat[:2, :2] + 3.0 * grid_lon[:2, :2] + 250.0
    )
    assert not mask[2].any()
    assert not mask[:, 2].any()


def test_linear_swath_interpolation_does_not_bridge_large_scan_gap() -> None:
    lat, lon = np.meshgrid(
        np.arange(2.0), np.array([0.0, 1.0, 5.0, 6.0]), indexing="ij"
    )
    values = 250.0 + lat + lon
    query_lat = np.array([[0.5]])
    query_lon = np.array([[3.0]])

    result, mask = _regrid_linear_supported(values, lat, lon, query_lat, query_lon)

    assert not mask.item()
    assert np.isnan(result.item())


def _batch() -> dict[str, torch.Tensor]:
    shape = (2, 2, 8, 8)
    return {
        "condition": torch.ones(shape),
        "condition_mask": torch.ones(2, 1, 8, 8, dtype=torch.bool),
        "target": torch.full((2, 1, 8, 8), 0.52),
        "target_physical": torch.full((2, 1, 8, 8), 252.0),
        "target_mask": torch.ones(2, 1, 8, 8, dtype=torch.bool),
        "target_norm_offset": torch.full((2, 1), 200.0),
        "target_norm_scale": torch.full((2, 1), 100.0),
        "condition_bounds": torch.tensor([[-1.0, 1.0, -1.0, 1.0]]).expand(2, -1),
        "target_bounds": torch.tensor([[-1.0, 1.0, -1.0, 1.0]]).expand(2, -1),
        "center": torch.zeros(2, 2),
        "sample_id": ["a", "b"],
        "meta": [{}, {}],
    }


def test_direct_unet_is_bounded_uses_kelvin_loss_and_prediction_contract() -> None:
    model = DirectUNetRegressor(
        condition_channels=2,
        base_channels=4,
        channel_mults=(1, 2),
        log_reconstruction_images=False,
    )
    batch = _batch()
    batch["target_mask"][0, :, 0, 0] = False

    normalized = model.predict_normalized(batch)
    objective = model.compute_training_objective(batch)
    with patch.object(model, "log"):
        training_loss = model.training_step(batch, 0)
    prediction = model.predict_batch(batch, PredictionRequest(ensemble_size=3))

    assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    assert torch.allclose(
        prediction.central_physical, torch.full_like(normalized, 250.0)
    )
    assert prediction.samples_physical.shape == (2, 3, 1, 8, 8)
    assert torch.isfinite(training_loss)
    assert objective.loss.detach().item() == pytest.approx(2.0)
    assert objective.components["mae_k"].detach().item() == pytest.approx(2.0)


def test_direct_unet_validates_kelvin_single_channel_data_spec() -> None:
    model = DirectUNetRegressor(
        condition_channels=2, base_channels=4, channel_mults=(1, 2)
    )
    model.validate_data_spec(
        DataSpec(("a", "b"), (PMW_CANONICAL_CHANNEL,), (8, 8), "K")
    )
    with pytest.raises(ValueError, match="target units K"):
        model.validate_data_spec(DataSpec(("a", "b"), ("wind_speed",), (8, 8), "m s-1"))


def _write_tiff(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=values.shape[0],
        dtype="float32",
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        crs="EPSG:4326",
    ) as dst:
        dst.write(values.astype(np.float32))


def test_pmw_dataset_reports_kelvin_units(tmp_path: Path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    _write_tiff(split / "geo.tif", np.full((1, 2, 2), 250.0))
    _write_tiff(split / "pmw.tif", np.full((1, 2, 2), 275.0))
    pd.DataFrame(
        [
            {
                "sample_id": "pmw",
                "storm_id": "AL012025",
                "condition_path": "train/geo.tif",
                "target_path": "train/pmw.tif",
                "condition_source_type": "geo",
                "target_source_type": "pmw",
                "condition_channels": json.dumps(["CMI_C13"]),
                "target_channels": json.dumps([PMW_CANONICAL_CHANNEL]),
                "condition_timestamp": "2025-06-21T12:00:00Z",
                "dt_minutes": 0.0,
                "ibtracs_center_lat": 1.0,
                "ibtracs_center_lon": 1.0,
            }
        ]
    ).to_csv(split / "manifest.csv", index=False)
    stats = {
        "channels": {
            "geo": {"CMI_C13": {"min": 200.0, "max": 300.0}},
            "pmw": {PMW_CANONICAL_CHANNEL: {"min": 200.0, "max": 300.0}},
        }
    }
    (tmp_path / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

    spec = PairedImageDataset(tmp_path, "train", target_size=(2, 2)).data_spec

    assert spec.target_units == "K"
    assert spec.target_channels == (PMW_CANONICAL_CHANNEL,)
    assert spec.spatial_shape == (2, 2)


def test_near89_experiment_composes_direct_unet_contract() -> None:
    config = compose_config(["experiment=geo_pmw_near89_unet"])
    assert config["data"]["root"].endswith("geo_pmw_near89_10bands_era5")
    assert config["data"]["include_test_in_train"] is False
    assert config["model"]["condition_channels"] == 23
    assert config["trainer"]["max_epochs"] == 100
    assert config["trainer"]["checkpoint"]["monitor"] == "val/rmse_k"
    assert isinstance(instantiate_model(config), DirectUNetRegressor)
