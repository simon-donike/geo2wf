from __future__ import annotations

import json
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from data.datamodule import PairedDataModule
from scripts.compare_pmw_evaluations import (
    _criteria as _promotion_criteria,
    _validate_common_cohort,
)
from data.dataset import (
    PMW_BRIGHTNESS_TEMPERATURE_89V,
    PMW_TIME_OFFSET,
    PMW_VALID_MASK,
    PairedImageDataset,
)
from scripts.export_geo_sar_geotiffs import Observation
from scripts.pmw_conditioning import (
    nearest_supported_pmw,
    pmw_audit_row,
    pmw_condition_settings,
    supported_pmw_by_storm,
)
from src.ERA5Residual import ERA5ResidualRegressor
from src.ERA5ResidualDiffusion import ERA5ResidualDiffusion
from train import build_model, load_config


def _write_tiff(path: Path, values: np.ndarray, *, left: float = 0.0) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[-2],
        width=values.shape[-1],
        count=values.shape[0],
        dtype="float32",
        transform=from_origin(left, 2.0, 1.0, 1.0),
        crs="EPSG:4326",
    ) as destination:
        destination.write(values.astype(np.float32))


def _dataset_root(tmp_path: Path, *, shifted_pmw: bool = False) -> None:
    split = tmp_path / "train"
    split.mkdir()
    _write_tiff(split / "geo.tif", np.full((1, 2, 2), 250.0))
    _write_tiff(split / "sar.tif", np.full((1, 2, 2), 20.0))
    pmw_values = np.array([[[275.0, np.nan], [275.0, 275.0]]], dtype=np.float32)
    _write_tiff(split / "pmw.tif", pmw_values, left=1.0 if shifted_pmw else 0.0)
    rows = []
    for index, gap in enumerate((-60.0, 30.0, 61.0, None)):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "storm_id": "AL012024",
                "condition_path": "train/geo.tif",
                "target_path": "train/sar.tif",
                "pmw_path": "train/pmw.tif" if gap is not None else "",
                "condition_source_type": "geo",
                "target_source_type": "sar",
                "condition_channels": json.dumps(["CMI_C13"]),
                "target_channels": json.dumps(["wind_speed"]),
                "pmw_channels": json.dumps(["TB_89.0V"]),
                "condition_timestamp": "2025-06-21T12:00:00Z",
                "dt_minutes": 0.0,
                "pmw_dt_minutes": "" if gap is None else gap,
                "ibtracs_center_lat": 1.0,
                "ibtracs_center_lon": 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(split / "manifest.csv", index=False)
    stats = {
        "channels": {
            "geo": {"CMI_C13": {"min": 200.0, "max": 300.0}},
            "sar": {"wind_speed": {"min": 0.0, "max": 80.0}},
            "pmw": {"TB_89.0V": {"min": 200.0, "max": 300.0}},
        }
    }
    (tmp_path / "stats.json").write_text(json.dumps(stats), encoding="utf-8")


def test_pmw_condition_filters_age_and_appends_value_mask_and_offset(tmp_path) -> None:
    _dataset_root(tmp_path)
    dataset = PairedImageDataset(
        tmp_path,
        "train",
        target_size=(2, 2),
        include_pmw=True,
        pmw_as_condition=True,
        max_pmw_time_gap_hours=1.0,
        pmw_include_time_offset=True,
    )

    first = dataset[0]
    second = dataset[1]

    assert len(dataset) == 2
    assert dataset.filtered_missing_pmw_count == 1
    assert dataset.filtered_stale_pmw_count == 1
    assert first["condition"].shape == (8, 2, 2)
    assert first["meta"]["condition_channels"][-3:] == [
        PMW_BRIGHTNESS_TEMPERATURE_89V,
        PMW_VALID_MASK,
        PMW_TIME_OFFSET,
    ]
    assert first["condition"][-3, 0, 0] == pytest.approx(0.75)
    assert first["condition"][-3, 0, 1] == 0.0
    assert torch.equal(
        first["condition"][-2].bool(),
        torch.tensor([[True, False], [True, True]]),
    )
    assert torch.all(first["condition"][-1] == 0.0)
    assert torch.all(second["condition"][-1] == 0.75)
    assert first["condition_mask"].all()


def test_pmw_condition_rejects_a_different_grid(tmp_path) -> None:
    _dataset_root(tmp_path, shifted_pmw=True)
    dataset = PairedImageDataset(
        tmp_path,
        "train",
        target_size=(2, 2),
        include_pmw=True,
        pmw_as_condition=True,
        max_pmw_time_gap_hours=1.0,
        pmw_include_time_offset=True,
    )

    with pytest.raises(ValueError, match="PMW raster grid"):
        dataset[0]


def test_datamodule_reads_all_pmw_condition_options() -> None:
    datamodule = PairedDataModule.from_config(
        {
            "data": {
                "root": "custom/data",
                "include_pmw": True,
                "pmw_as_condition": True,
                "max_pmw_time_gap_hours": 1.0,
                "pmw_include_time_offset": True,
            }
        }
    )

    assert datamodule.include_pmw is True
    assert datamodule.pmw_as_condition is True
    assert datamodule.max_pmw_time_gap_hours == 1.0
    assert datamodule.pmw_include_time_offset is True


def _observation(
    observation_id: str,
    timestamp: str,
    *,
    source_type: str = "pmw",
    sensor: str = "GMI_GPM",
) -> Observation:
    return Observation(
        observation_id=observation_id,
        storm_id="AL012024",
        split="test",
        source_type=source_type,
        source="test",
        sensor=sensor,
        path=Path("unused.nc"),
        timestamp=pd.Timestamp(timestamp),
        center_lat=20.0,
        center_lon=-60.0,
        ibtracs_center_lat=20.0,
        ibtracs_center_lon=-60.0,
        variables=("TB_89.0V",),
    )


def test_raw_inference_pmw_matching_is_supported_bounded_and_audited() -> None:
    reference = _observation(
        "geo", "2025-01-01T00:00:00Z", source_type="geo", sensor="ABI"
    )
    boundary = _observation("boundary", "2025-01-01T01:00:00Z")
    unsupported = _observation(
        "unsupported", "2025-01-01T00:01:00Z", sensor="ATMS_NOAA20"
    )
    grouped = supported_pmw_by_storm([boundary, unsupported])

    selected, gap, status = nearest_supported_pmw(
        reference, grouped, max_time_gap_hours=1.0
    )

    assert selected == boundary
    assert gap == 60.0
    assert status == "matched"
    audit = pmw_audit_row(reference, selected, gap, status)
    assert audit["pmw_sensor"] == "GMI_GPM"
    assert audit["status"] == "matched"


def test_pmw_experiment_configs_keep_expected_channel_contracts() -> None:
    stage1_config = load_config(
        "configs/config_geo_sar_10bands_era5_pmw_residual.yaml"
    )
    stage2_config = load_config(
        "configs/config_geo_sar_10bands_era5_pmw_diffusion_residual_deterministic.yaml"
    )

    stage1 = build_model(stage1_config)
    stage2_config["model"]["residual"]["baseline"]["checkpoint_path"] = "unused.ckpt"
    baseline = ERA5ResidualRegressor(
        condition_channels=26, base_channels=4, channel_mults=(1, 2)
    )
    with patch("train.load_frozen_deterministic_baseline", return_value=baseline):
        stage2 = build_model(stage2_config)

    assert stage1.condition_channels == 26
    assert stage1.model.stem.in_channels == 29
    assert stage2.base_condition_channels == 27
    assert stage2.model.condition_channels == 29
    assert stage2_config["model"]["unet"]["channels"] == 30
    assert pmw_condition_settings(stage2_config) == (True, 1.0, True)
    incompatible = ERA5ResidualRegressor(
        condition_channels=23, base_channels=4, channel_mults=(1, 2)
    )
    with patch("train.load_frozen_deterministic_baseline", return_value=incompatible):
        with pytest.raises(ValueError, match="condition width"):
            build_model(stage2_config)


def test_pmw_stage1_and_stage2_training_steps_are_finite() -> None:
    mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    batch = {
        "condition": torch.rand(1, 26, 8, 8),
        "condition_mask": mask,
        "target": torch.full((1, 1, 8, 8), 0.375),
        "target_physical": torch.full((1, 1, 8, 8), 30.0),
        "target_mask": mask,
        "target_norm_offset": torch.zeros(1, 1),
        "target_norm_scale": torch.full((1, 1), 80.0),
        "era5_wind_speed": torch.full((1, 1, 8, 8), 0.25),
        "era5_wind_speed_physical": torch.full((1, 1, 8, 8), 20.0),
        "era5_wind_speed_mask": mask,
        "sample_id": ["pmw"],
    }
    stage1 = ERA5ResidualRegressor(
        condition_channels=26, base_channels=4, channel_mults=(1, 2)
    )
    stage2 = ERA5ResidualDiffusion(
        base_condition_channels=27,
        baseline_source="deterministic",
        baseline_model=stage1,
        generated_channels=1,
        num_timesteps=4,
        schedule="cosine",
        model_dim=4,
        model_dim_mults=(1, 2),
        model_channels=30,
        model_out_dim=1,
        sampling_method="ddim",
        sampling_timesteps=2,
    )

    with patch.object(stage1, "log"):
        stage1_loss = stage1.training_step(batch, 0)
    with patch.object(stage2, "log"):
        stage2_loss = stage2.training_step(batch, 0)

    assert torch.isfinite(stage1_loss)
    assert torch.isfinite(stage2_loss)


def test_pmw_promotion_comparison_requires_identical_rows_and_applies_thresholds() -> None:
    cohort = {"sha256": "same", "count": 10, "columns": ["sample_id"]}
    current = {
        "evaluation_rows": cohort,
        "split": "val",
        "pmw_max_time_gap_hours": 1.0,
        "limit_batches": 1.0,
        "pmw_as_condition": False,
        "metrics": {
            "val/peak_structure_score": 10.0,
            "val/robust_peak_mae_ms": 5.0,
            "val/mae_ms": 2.0,
        },
    }
    candidate = {
        **current,
        "pmw_as_condition": True,
        "metrics": {
            "val/peak_structure_score": 9.0,
            "val/robust_peak_mae_ms": 4.5,
            "val/mae_ms": 2.04,
        },
    }

    assert _validate_common_cohort(current, candidate) == cohort
    assert all(item["passed"] for item in _promotion_criteria(1, current, candidate))

    candidate["evaluation_rows"] = {**cohort, "sha256": "different"}
    with pytest.raises(ValueError, match="cohorts differ"):
        _validate_common_cohort(current, candidate)
