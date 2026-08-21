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
    JointPairedIntensityDataModule,
)
from geo2wf.models.base import PredictionRequest
from geo2wf.models.deterministic_residual import ERA5ResidualRegressor
from geo2wf.models.bottleneck_unet_mlp import (
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
    with pytest.raises(ValueError, match="continuous IBTrACS"):
        model.validate_data_spec(
            DataSpec(("a", "b", "c"), ("wind_speed",), (16, 16), "m s-1")
        )


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
                },
                {
                    "USA_ATCF_ID": storm_id,
                    "ISO_TIME": "2024-08-01T03:00:00Z",
                    "USA_WIND": 50.0 + split_index,
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
    root: Path, ibtracs_file: Path, *, use_era5: bool = True
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
        normalization="min-max",
        target_normalization="min-max",
    )


def test_data_adapter_filters_and_joins_exact_continuous_target(tmp_path: Path) -> None:
    root, ibtracs_file = tmp_path / "paired", tmp_path / "ibtracs.csv"
    _write_joint_fixture(root, ibtracs_file)
    datamodule = _joint_datamodule(root, ibtracs_file)
    datamodule.setup("fit")

    assert len(datamodule.train_dataset) == 2
    assert len(datamodule.val_dataset) == 2
    assert IBTRACS_MAX_WIND_COMPANION in datamodule.data_spec.companions
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["intensity_target_ms"].item() == pytest.approx(45.0 * 0.514444)
    assert batch["condition"].shape[1] == 14
    assert "era5_wind_speed" in batch

    no_era5 = _joint_datamodule(root, ibtracs_file, use_era5=False)
    no_era5.setup("fit")
    no_era5_batch = next(iter(no_era5.train_dataloader()))
    assert no_era5_batch["condition"].shape[1] == 5
    assert "era5_wind_speed" not in no_era5_batch
    assert "target_category" not in batch


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
    direct_config = compose_config(["experiment=geo_pmw_near89_unet_no_era5"])
    assert direct_config["data"]["use_era5"] is False
    assert direct_config["model"]["condition_channels"] == 14
