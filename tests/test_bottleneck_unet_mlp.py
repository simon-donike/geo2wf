from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import rasterio
from rasterio.transform import from_origin
from matplotlib import pyplot as plt
import torch

from geo2wf.config import compose_config, instantiate_model
from geo2wf.data.contracts import DataSpec
from geo2wf.data.joint_intensity import (
    IBTRACS_MAX_WIND_COMPANION,
    IBTRACS_STRUCTURE_COMPANION,
    JointPairedIntensityDataModule,
)
from geo2wf.data.encoder_intensity import EncoderIBTrACSDataModule
from geo2wf.models.base import PredictionRequest
from geo2wf.models.deterministic_residual import ERA5ResidualRegressor
from geo2wf.models.bottleneck_unet_mlp import (
    BottleneckEncoderMLP,
    BottleneckEncoderMLPRegressor,
    BottleneckUNetMLP,
    BottleneckUNetMLPRegressor,
)

from geo2wf.visualization.wind_fields import plot_validation_reconstruction_batch


def _batch(batch_size: int = 2, channels: int = 3, size: int = 16) -> dict:
    return {
        "condition": torch.randn(batch_size, channels, size, size),
        "condition_mask": torch.ones(batch_size, 1, size, size, dtype=torch.bool),
        "target": torch.full((batch_size, 1, size, size), 0.25),
        "target_physical": torch.full((batch_size, 1, size, size), 20.0),
        "target_mask": torch.ones(batch_size, 1, size, size, dtype=torch.bool),
        "target_norm_offset": torch.zeros(batch_size, 1),
        "target_norm_scale": torch.full((batch_size, 1), 80.0),
        "condition_bounds": torch.tensor([[-1.0, 1.0, -1.0, 1.0]]).expand(
            batch_size, -1
        ),
        "target_bounds": torch.tensor([[-1.0, 1.0, -1.0, 1.0]]).expand(batch_size, -1),
        "center": torch.zeros(batch_size, 2),
        "sample_id": [f"sample-{index}" for index in range(batch_size)],
        "meta": [{"storm_id": f"storm-{index}"} for index in range(batch_size)],
        "intensity_target_ms": torch.linspace(31.25, 44.75, batch_size),
    }


@pytest.mark.parametrize("shape", [(16, 16), (15, 17)])
def test_architecture_preserves_image_shape_and_exposes_bottleneck(shape) -> None:
    model = BottleneckUNetMLP(
        in_channels=4,
        base_channels=4,
        channel_mults=(1, 2, 4),
        intensity_hidden_features=8,
        intensity_dropout=0.0,
    )
    output = model(torch.randn(2, 4, *shape))

    assert output.reconstruction_normalized.shape == (2, 1, *shape)
    assert output.ibtracs_max_wind_ms.shape == (2,)
    assert output.bottleneck.shape[1] == 16
    assert output.reconstruction_normalized.min() >= 0.0
    assert output.reconstruction_normalized.max() <= 1.0
    assert output.ibtracs_max_wind_ms.min() >= 0.0


def test_reconstruction_plot_shows_ibtracs_actual_vs_mlp_prediction() -> None:
    sample = {
        "condition": torch.full((3, 4, 4), 0.5),
        "prediction": torch.full((1, 4, 4), 20.0),
        "target": torch.full((1, 4, 4), 22.0),
        "condition_mask": torch.ones((1, 4, 4), dtype=torch.bool),
        "target_mask": torch.ones((1, 4, 4), dtype=torch.bool),
        "intensity_target_ms": 40.0,
        "intensity_prediction_ms": 37.5,
        "intensity_target_label": "IBTrACS max wind",
        "physical_wind_output": True,
    }

    figure = plot_validation_reconstruction_batch([sample])
    intensity_axis = next(
        axis for axis in figure.axes if axis.get_title().startswith("IBTrACS max wind")
    )

    assert [tick.get_text() for tick in intensity_axis.get_xticklabels()] == [
        "Actual",
        "Predicted",
    ]
    assert [patch.get_height() for patch in intensity_axis.patches] == [40.0, 37.5]
    assert "error -2.5" in intensity_axis.get_title()
    plt.close(figure)


def test_intensity_branch_reaches_encoder_but_not_decoder() -> None:
    model = BottleneckUNetMLP(
        in_channels=4,
        base_channels=4,
        channel_mults=(1, 2),
        intensity_hidden_features=8,
        intensity_dropout=0.0,
    )
    inputs = torch.randn(2, 4, 12, 12)

    model(inputs).ibtracs_max_wind_ms.sum().backward()

    assert model.stem.weight.grad is not None
    assert model.intensity_head.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.decoder.parameters())
    assert model.reconstruction_head.weight.grad is None

    model.zero_grad(set_to_none=True)
    output = model(inputs)
    (
        output.reconstruction_normalized.sum() + output.ibtracs_max_wind_ms.sum()
    ).backward()
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())
    assert model.reconstruction_head.weight.grad is not None


def test_encoder_only_architecture_has_no_decoder_and_backpropagates() -> None:
    model = BottleneckEncoderMLP(
        in_channels=4,
        base_channels=4,
        channel_mults=(1, 2),
        intensity_hidden_features=8,
        intensity_dropout=0.0,
    )
    output = model(torch.randn(2, 4, 12, 12))
    output.intensity_prediction_ms.sum().backward()

    assert output.intensity_prediction_ms.shape == (2,)
    assert output.bottleneck.shape == (2, 8, 6, 6)
    assert not hasattr(model, "decoder")
    assert not hasattr(model, "reconstruction_head")
    assert all("decoder" not in name for name, _ in model.named_parameters())
    assert model.stem.weight.grad is not None
    assert model.intensity_head.weight.grad is not None


def test_joint_objective_uses_continuous_ibtracs_target_and_masks_image() -> None:
    model = BottleneckUNetMLPRegressor(
        condition_channels=3,
        base_channels=4,
        channel_mults=(1, 2),
        intensity_hidden_features=8,
        intensity_dropout=0.0,
        log_reconstruction_images=False,
    )
    batch = _batch()
    batch["condition_mask"][:, :, 0, 0] = False
    batch["condition"][:, :, 0, 0] = 1.0e6
    batch["target_mask"][:, :, 0, 1] = False

    normalized = model.predict_normalized(batch)
    prediction = model.predict_joint(batch)
    objective = model.compute_training_objective(batch)
    changed_invalid_targets = dict(batch)
    changed_invalid_targets["target_physical"] = batch["target_physical"].clone()
    changed_invalid_targets["target_physical"][:, :, 0, :2] = 1.0e9
    masked_objective = model.compute_training_objective(changed_invalid_targets)
    expected_intensity = torch.where(
        (prediction.ibtracs_max_wind_ms - batch["intensity_target_ms"]).abs() <= 5.0,
        0.5 * (prediction.ibtracs_max_wind_ms - batch["intensity_target_ms"]).square(),
        5.0
        * ((prediction.ibtracs_max_wind_ms - batch["intensity_target_ms"]).abs() - 2.5),
    ).mean()

    assert prediction.central_physical.shape == (2, 1, 16, 16)
    assert normalized.bottleneck.ndim == 4
    assert torch.allclose(objective.components["intensity_loss"], expected_intensity)
    assert torch.allclose(
        objective.components["image_loss"],
        masked_objective.components["image_loss"],
    )
    assert "target_category" not in batch
    assert torch.isfinite(objective.loss)


def test_joint_optional_structure_head_outputs_and_contributes_to_loss() -> None:
    model = BottleneckUNetMLPRegressor(
        condition_channels=3,
        base_channels=4,
        channel_mults=(1, 2),
        intensity_hidden_features=8,
        intensity_dropout=0.0,
        structure_head_enabled=True,
        structure_loss_weight=0.5,
        log_reconstruction_images=False,
    )
    batch = _batch()
    structure_keys = (
        "ibtracs_eye_size_km",
        "ibtracs_rmw_km",
        "ibtracs_r34_equivalent_km",
        "ibtracs_r50_equivalent_km",
        "ibtracs_r64_equivalent_km",
    )
    for index, key in enumerate(structure_keys):
        batch[key] = torch.full((2,), 20.0 + 10.0 * index)
        batch[f"{key}_valid"] = torch.ones(2, dtype=torch.bool)

    prediction = model.predict_joint(batch)
    objective = model.compute_training_objective(batch)

    assert prediction.structure_prediction_km.shape == (2, 5)
    assert torch.all(prediction.structure_prediction_km >= 0.0)
    assert objective.components["structure_loss"] > 0.0
    expected = (
        model.image_loss_weight * objective.components["image_loss"]
        + model.intensity_loss_weight * objective.components["intensity_loss"]
        + model.structure_loss_weight * objective.components["structure_loss"]
    )
    assert torch.allclose(objective.loss, expected)


def test_joint_structure_loss_masks_nan_companions_before_huber() -> None:
    model = BottleneckUNetMLPRegressor(
        condition_channels=3,
        base_channels=4,
        channel_mults=(1, 2),
        intensity_hidden_features=8,
        intensity_dropout=0.0,
        structure_head_enabled=True,
        structure_loss_weight=0.25,
        log_reconstruction_images=False,
    )
    batch = _batch()
    structure_keys = (
        "ibtracs_eye_size_km",
        "ibtracs_rmw_km",
        "ibtracs_r34_equivalent_km",
        "ibtracs_r50_equivalent_km",
        "ibtracs_r64_equivalent_km",
    )
    for index, key in enumerate(structure_keys):
        values = torch.full((2,), 30.0 + 10.0 * index)
        valid = torch.ones(2, dtype=torch.bool)
        values[0] = torch.nan
        valid[0] = False
        batch[key] = values
        batch[f"{key}_valid"] = valid

    objective = model.compute_training_objective(batch)
    assert torch.isfinite(objective.loss)
    assert torch.isfinite(objective.components["structure_loss"])
    objective.loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_model_rejects_data_without_continuous_ibtracs_companion() -> None:
    model = BottleneckUNetMLPRegressor(
        condition_channels=3, base_channels=4, channel_mults=(1, 2)
    )
    valid = DataSpec(
        ("a", "b", "c"),
        ("wind_speed",),
        (16, 16),
        "m s-1",
        frozenset({IBTRACS_MAX_WIND_COMPANION}),
    )
    model.validate_data_spec(valid)
    with pytest.raises(ValueError, match="continuous scalar intensity"):
        model.validate_data_spec(
            DataSpec(("a", "b", "c"), ("wind_speed",), (16, 16), "m s-1")
        )


def test_joint_validation_logs_dual_reference_ri_and_field_metrics(
    monkeypatch,
) -> None:
    model = BottleneckUNetMLPRegressor(
        condition_channels=3,
        base_channels=4,
        channel_mults=(1, 2),
        log_reconstruction_images=False,
    )
    model._evaluation_rows["val"] = [
        {
            "sample_id": "sample",
            "storm_id": "storm",
            "prediction_ms": 32.0,
            "ibtracs_target_ms": 30.0,
            "sar_robust_peak_target_ms": 31.0,
            "raw_unet_max_ms": 29.0,
            "raw_unet_robust_peak_ms": 28.0,
            "field_valid_pixels": 2,
            "field_absolute_error_sum": 4.0,
            "field_squared_error_sum": 10.0,
            "field_signed_error_sum": -2.0,
            "is_rapid_intensification": True,
        }
    ]
    logged = {}
    monkeypatch.setattr(
        model,
        "log",
        lambda name, value, **kwargs: logged.__setitem__(name, value),
    )

    model._log_ri_statistics()

    assert logged["val_ri/samples"] == 1.0
    assert "val_ri/ibtracs_category_macro_f1" in logged
    assert "val_ri/ibtracs_raw_unet_bias_ms" in logged
    assert "val_ri/sar_robust_peak_raw_unet_mae_ms" in logged
    assert logged["val_ri/field_mae_ms"] == pytest.approx(2.0)
    assert logged["val_ri/field_bias_ms"] == pytest.approx(-1.0)


def test_deterministic_unet_can_train_and_predict_without_era5() -> None:
    model = ERA5ResidualRegressor(
        condition_channels=3,
        use_era5=False,
        base_channels=4,
        channel_mults=(1, 2),
        off_swath_anchor_weight=0.0,
        peak_loss_weight=0.0,
        log_reconstruction_images=False,
    )
    batch = _batch(channels=3)

    prediction, target, valid_mask, baseline = model._batch_outputs(batch)
    result = model.predict_batch(batch, PredictionRequest(ensemble_size=2))

    assert prediction.shape == target.shape == valid_mask.shape
    assert torch.isfinite(prediction).all()
    assert torch.count_nonzero(baseline) == 0
    assert result.samples_physical.shape == (2, 2, 1, 16, 16)
    assert result.baseline_physical is None


def _write_tiff(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[-2],
        width=values.shape[-1],
        count=values.shape[0],
        dtype="float32",
        transform=from_origin(0.0, 8.0, 1.0, 1.0),
        crs="EPSG:4326",
    ) as destination:
        destination.write(values.astype(np.float32))


def _write_joint_fixture(root: Path, ibtracs_file: Path) -> None:
    stats = {
        "channels": {
            "geo": {"CMI_C13": {"min": 200.0, "max": 300.0}},
            "era5": {
                "precipitable_water": {"min": 0.0, "max": 100.0},
                "sst": {"min": 250.0, "max": 320.0},
                "pressure_msl": {"min": 90000.0, "max": 110000.0},
                "temperature_2m": {"min": 220.0, "max": 330.0},
                "dewpoint_2m": {"min": 200.0, "max": 330.0},
                "u_wind_10m": {"min": -80.0, "max": 80.0},
                "v_wind_10m": {"min": -80.0, "max": 80.0},
            },
            "sar": {"wind_speed": {"min": 0.0, "max": 80.0}},
        }
    }
    root.mkdir(parents=True)
    (root / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

    ibtracs_rows = []
    for split_index, split in enumerate(("train", "val", "test")):
        paired_split = root / split
        paired_split.mkdir()
        rows = []
        storm_id = f"AL0{split_index + 1}2024"
        ibtracs_rows.extend(
            [
                {
                    "USA_ATCF_ID": storm_id,
                    "ISO_TIME": "2024-08-01T00:00:00Z",
                    "USA_WIND": 40.0 + split_index,
                    "USA_EYE": 10.0,
                    "USA_RMW": 20.0,
                    **{
                        f"USA_R{threshold}_{quadrant}": value
                        for threshold, value in ((34, 40.0), (50, 30.0), (64, 20.0))
                        for quadrant in ("NE", "SE", "SW", "NW")
                    },
                },
                {
                    "USA_ATCF_ID": storm_id,
                    "ISO_TIME": "2024-08-01T03:00:00Z",
                    "USA_WIND": 50.0 + split_index,
                    "USA_EYE": 14.0,
                    "USA_RMW": 24.0,
                    **{
                        f"USA_R{threshold}_{quadrant}": value
                        for threshold, value in ((34, 44.0), (50, 34.0), (64, 24.0))
                        for quadrant in ("NE", "SE", "SW", "NW")
                    },
                },
            ]
        )
        for sample_index in range(2):
            sample_id = f"{split}-{sample_index}"
            condition_name = f"{sample_id}-geo.tif"
            context_name = f"{sample_id}-era5.tif"
            target_name = f"{sample_id}-sar.tif"
            _write_tiff(
                paired_split / condition_name,
                np.full((1, 8, 8), 240.0 + sample_index),
            )
            _write_tiff(
                paired_split / context_name,
                np.full((7, 8, 8), 10.0 + sample_index),
            )
            _write_tiff(
                paired_split / target_name,
                np.full((1, 8, 8), 20.0 + sample_index),
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "storm_id": storm_id,
                    "condition_path": f"{split}/{condition_name}",
                    "context_path": f"{split}/{context_name}",
                    "target_path": f"{split}/{target_name}",
                    "condition_source_type": "geo",
                    "context_source_type": "era5",
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
                    "target_source_type": "sar",
                    "condition_channels": json.dumps(["CMI_C13"]),
                    "target_channels": json.dumps(["wind_speed"]),
                    "condition_timestamp": "2024-08-01T00:00:00Z",
                    "target_timestamp": "2024-08-01T01:30:00Z",
                    "dt_minutes": 0.0,
                    "ibtracs_center_lat": 4.0,
                    "ibtracs_center_lon": 4.0,
                }
            )
        pd.DataFrame(rows).to_csv(paired_split / "manifest.csv", index=False)
    pd.DataFrame(ibtracs_rows).to_csv(ibtracs_file, index=False)


def _joint_datamodule(
    root: Path,
    ibtracs_file: Path,
    *,
    use_era5: bool = True,
    intensity_target_source: str = "ibtracs",
    require_sar_valid_center: bool = False,
) -> JointPairedIntensityDataModule:
    return JointPairedIntensityDataModule(
        root=root,
        ibtracs_file=ibtracs_file,
        max_ibtracs_bracket_hours=3.0,
        stats_file=root / "stats.json",
        batch_size=1,
        num_workers=0,
        target_size=(8, 8),
        random_flips=False,
        use_era5=use_era5,
        intensity_target_source=intensity_target_source,
        require_sar_valid_center=require_sar_valid_center,
        normalization="min-max",
        target_normalization="min-max",
    )


def _encoder_datamodule(
    root: Path, ibtracs_file: Path, *, use_era5: bool = True
) -> EncoderIBTrACSDataModule:
    return EncoderIBTrACSDataModule(
        root=root,
        ibtracs_file=ibtracs_file,
        eligibility_cache_dir=root / "cohort-cache",
        max_ibtracs_bracket_hours=3.0,
        stats_file=root / "stats.json",
        batch_size=1,
        num_workers=0,
        target_size=(8, 8),
        random_flips=False,
        use_era5=use_era5,
        intensity_target_source="ibtracs",
        require_sar_valid_center=True,
        normalization="min-max",
        target_normalization="min-max",
    )


def test_encoder_dataset_reuses_cached_joint_cohort_without_reading_sar(
    tmp_path: Path, monkeypatch
) -> None:
    root, ibtracs_file = tmp_path / "paired", tmp_path / "ibtracs.csv"
    _write_joint_fixture(root, ibtracs_file)
    joint = _joint_datamodule(root, ibtracs_file, require_sar_valid_center=True)
    joint.setup(None)
    encoder = _encoder_datamodule(root, ibtracs_file)
    encoder.setup(None)

    for split in ("train_dataset", "val_dataset", "test_dataset"):
        assert (
            getattr(encoder, split).samples["sample_id"].tolist()
            == getattr(joint, split).samples["sample_id"].tolist()
        )

    original_open = rasterio.open

    def reject_sar(path, *args, **kwargs):
        if str(path).endswith("-sar.tif"):
            raise AssertionError(f"condition-only loader opened SAR: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", reject_sar)
    cached = _encoder_datamodule(root, ibtracs_file)
    cached.setup("fit")
    batch = next(iter(cached.train_dataloader()))

    assert batch["condition"].shape[1] == 14
    assert batch["intensity_target_ms"].item() == pytest.approx(45.0 * 0.514444)
    assert {
        "target",
        "target_physical",
        "target_mask",
        "target_norm_offset",
        "target_norm_scale",
    }.isdisjoint(batch)
    assert cached.data_spec.target_channels == ()


def test_data_adapter_filters_and_joins_exact_continuous_target(tmp_path: Path) -> None:
    root, ibtracs_file = tmp_path / "paired", tmp_path / "ibtracs.csv"
    _write_joint_fixture(root, ibtracs_file)
    datamodule = _joint_datamodule(root, ibtracs_file)
    datamodule.setup("fit")

    assert len(datamodule.train_dataset) == 2
    assert len(datamodule.val_dataset) == 2
    assert IBTRACS_MAX_WIND_COMPANION in datamodule.data_spec.companions
    assert IBTRACS_STRUCTURE_COMPANION in datamodule.data_spec.companions
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["intensity_target_ms"].item() == pytest.approx(45.0 * 0.514444)
    assert batch["ibtracs_eye_size_km"].item() == pytest.approx(12.0 * 1.852)
    assert batch["ibtracs_rmw_km"].item() == pytest.approx(22.0 * 1.852)
    assert batch["ibtracs_r34_equivalent_km"].item() == pytest.approx(42.0 * 1.852)
    assert batch["ibtracs_r50_equivalent_km"].item() == pytest.approx(32.0 * 1.852)
    assert batch["ibtracs_r64_equivalent_km"].item() == pytest.approx(22.0 * 1.852)
    assert batch["ibtracs_eye_size_km_valid"].item()
    assert batch["ibtracs_r64_equivalent_km_valid"].item()
    assert batch["condition"].shape[1] == 14
    assert "era5_wind_speed" in batch

    no_era5 = _joint_datamodule(root, ibtracs_file, use_era5=False)
    no_era5.setup("fit")
    no_era5_batch = next(iter(no_era5.train_dataloader()))
    assert no_era5_batch["condition"].shape[1] == 5
    assert "era5_wind_speed" not in no_era5_batch
    assert "target_category" not in batch


def test_target_variants_share_exact_center_valid_cohort(tmp_path: Path) -> None:
    root, ibtracs_file = tmp_path / "paired", tmp_path / "ibtracs.csv"
    _write_joint_fixture(root, ibtracs_file)
    modules = {
        source: _joint_datamodule(
            root,
            ibtracs_file,
            intensity_target_source=source,
            require_sar_valid_center=True,
        )
        for source in ("ibtracs", "sar_robust_peak")
    }
    for datamodule in modules.values():
        datamodule.setup(None)

    for split in ("train_dataset", "val_dataset", "test_dataset"):
        ibtracs = getattr(modules["ibtracs"], split)
        sar = getattr(modules["sar_robust_peak"], split)
        assert (
            ibtracs.samples["sample_id"].tolist() == sar.samples["sample_id"].tolist()
        )
    ibtracs_sample = modules["ibtracs"].train_dataset[0]
    sar_sample = modules["sar_robust_peak"].train_dataset[0]
    assert ibtracs_sample["intensity_target_ms"] == pytest.approx(45.0 * 0.514444)
    assert sar_sample["intensity_target_ms"] == pytest.approx(20.0)
    assert sar_sample["ibtracs_target_ms"] == ibtracs_sample["intensity_target_ms"]
    assert sar_sample["sar_has_valid_center"]
    assert sar_sample["intensity_filtering_counts"]["retained"] == 2


def test_center_filter_inspects_effective_sar_mask(tmp_path: Path) -> None:
    root, ibtracs_file = tmp_path / "paired", tmp_path / "ibtracs.csv"
    _write_joint_fixture(root, ibtracs_file)
    path = root / "train" / "train-0-sar.tif"
    with rasterio.open(path, "r+") as raster:
        values = raster.read(1)
        values[4, 4] = np.nan
        raster.write(values, 1)

    filtered = _joint_datamodule(
        root, ibtracs_file, require_sar_valid_center=True
    )._make_dataset("train", augment=False)
    unfiltered = _joint_datamodule(
        root, ibtracs_file, require_sar_valid_center=False
    )._make_dataset("train", augment=False)

    assert filtered.samples["sample_id"].tolist() == ["train-1"]
    assert filtered.filtered_invalid_sar_center_count == 1
    assert len(unfiltered) == 2
    assert unfiltered[0]["sar_has_valid_center"] is not None
    assert not bool(unfiltered[0]["sar_has_valid_center"])


def test_data_adapter_rejects_conflicts_and_filters_wide_brackets(
    tmp_path: Path,
) -> None:
    root, ibtracs_file = tmp_path / "paired", tmp_path / "ibtracs.csv"
    _write_joint_fixture(root, ibtracs_file)
    fixes = pd.read_csv(ibtracs_file)
    conflict = fixes.iloc[[0]].copy()
    conflict["USA_WIND"] = 99.0
    pd.concat([fixes, conflict], ignore_index=True).to_csv(ibtracs_file, index=False)
    with pytest.raises(ValueError, match="conflicting USA_WIND"):
        _joint_datamodule(root, ibtracs_file)

    fixes.loc[fixes["ISO_TIME"].str.contains("03:00:00"), "ISO_TIME"] = (
        "2024-08-01T04:00:00Z"
    )
    fixes.to_csv(ibtracs_file, index=False)
    datamodule = _joint_datamodule(root, ibtracs_file)
    datamodule.setup("fit")
    assert len(datamodule.train_dataset) == 0
    assert datamodule.train_dataset.filtered_unbracketed_count == 2


def test_lightning_fit_checkpoint_and_hydra_composition(tmp_path: Path) -> None:
    root, ibtracs_file = tmp_path / "paired", tmp_path / "ibtracs.csv"
    _write_joint_fixture(root, ibtracs_file)
    datamodule = _joint_datamodule(root, ibtracs_file)
    model = BottleneckUNetMLPRegressor(
        condition_channels=14,
        base_channels=4,
        channel_mults=(1, 2),
        intensity_hidden_features=8,
        intensity_dropout=0.0,
        log_reconstruction_images=False,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)
    checkpoint = tmp_path / "joint.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = BottleneckUNetMLPRegressor.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    restored.validate_data_spec(datamodule.data_spec)

    config = compose_config(["experiment=bottleneck_unet_mlp"])
    assert config["model"]["intensity_loss_weight"] == 1.0
    assert config["trainer"]["checkpoint"]["monitor"] == "val/loss"
    assert isinstance(instantiate_model(config), BottleneckUNetMLPRegressor)

    max_wind_config = compose_config(["experiment=bottleneck_unet_mlp_max_wind"])
    assert max_wind_config["data"]["use_era5"] is True
    assert max_wind_config["model"]["structure_head_enabled"] is False
    assert max_wind_config["model"]["structure_loss_weight"] == 0.0
    assert max_wind_config["trainer"]["deterministic"] is False
    assert max_wind_config["trainer"]["default_root_dir"] == (
        "logs/latent-structure/max-wind"
    )

    radii_config = compose_config(["experiment=bottleneck_unet_mlp_max_wind_radii"])
    assert radii_config["data"] == max_wind_config["data"]
    assert radii_config["model"]["structure_head_enabled"] is True
    assert radii_config["model"]["structure_loss_weight"] == 0.25
    assert radii_config["model"]["condition_channels"] == 23
    assert radii_config["trainer"]["deterministic"] is False
    assert radii_config["trainer"]["default_root_dir"] == (
        "logs/latent-structure/max-wind-radii"
    )
    assert isinstance(instantiate_model(radii_config), BottleneckUNetMLPRegressor)

    no_era5_config = compose_config(["experiment=bottleneck_unet_mlp_no_era5"])
    assert no_era5_config["data"]["use_era5"] is False
    assert no_era5_config["model"]["condition_channels"] == 14
    assert no_era5_config["trainer"]["early_stopping"]["patience"] == 50
    comparison_no_era5 = compose_config(
        ["experiment=intensity_comparison_unet_no_era5"]
    )
    assert comparison_no_era5["data"]["require_era5"] is True
    assert comparison_no_era5["data"]["use_era5"] is False
    assert comparison_no_era5["model"]["condition_channels"] == 14
    assert comparison_no_era5["model"]["use_era5"] is False


def test_encoder_only_lightning_checkpoint_and_hydra_presets(tmp_path: Path) -> None:
    root, ibtracs_file = tmp_path / "paired", tmp_path / "ibtracs.csv"
    _write_joint_fixture(root, ibtracs_file)
    datamodule = _encoder_datamodule(root, ibtracs_file)
    model = BottleneckEncoderMLPRegressor(
        condition_channels=14,
        base_channels=4,
        channel_mults=(1, 2),
        intensity_hidden_features=8,
        intensity_dropout=0.0,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)
    checkpoint = tmp_path / "encoder-only.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = BottleneckEncoderMLPRegressor.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    restored.validate_data_spec(datamodule.data_spec)
    prediction = restored.predict_batch(next(iter(datamodule.train_dataloader())))
    assert prediction.intensity_prediction_ms.shape == (1,)

    with_era5 = compose_config(["experiment=unet_encoder_mlp_ibtracs"])
    assert with_era5["model"]["condition_channels"] == 23
    assert with_era5["trainer"]["checkpoint"]["monitor"] == ("val/intensity_mae_ms")
    assert isinstance(instantiate_model(with_era5), BottleneckEncoderMLPRegressor)
    without_era5 = compose_config(["experiment=unet_encoder_mlp_ibtracs_no_era5"])
    assert without_era5["data"]["use_era5"] is False
    assert without_era5["model"]["condition_channels"] == 14


@pytest.mark.parametrize("use_era5", [True, False])
def test_one_epoch_joint_smoke_for_era5_matrix(
    tmp_path: Path,
    use_era5: bool,
) -> None:
    target_source = "ibtracs"
    root = tmp_path / f"paired-{use_era5}"
    ibtracs_file = tmp_path / f"ibtracs-{use_era5}.csv"
    _write_joint_fixture(root, ibtracs_file)
    datamodule = _joint_datamodule(
        root,
        ibtracs_file,
        use_era5=use_era5,
        intensity_target_source=target_source,
        require_sar_valid_center=True,
    )
    datamodule.setup("fit")
    model = BottleneckUNetMLPRegressor(
        condition_channels=datamodule.data_spec.condition_channel_count,
        base_channels=4,
        channel_mults=(1, 2),
        intensity_hidden_features=8,
        intensity_dropout=0.0,
        log_reconstruction_images=False,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)

    assert trainer.current_epoch == 1
    assert trainer.callback_metrics["val_ri/samples"] == 0.0
