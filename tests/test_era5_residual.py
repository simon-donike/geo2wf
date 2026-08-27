from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.ERA5Residual import (
    ERA5ResidualRegressor,
    continuous_high_wind_pixel_weights,
    masked_huber_loss,
    radial_profile_huber_loss,
    robust_top_fraction_peak_loss,
    soft_exceedance_area_loss,
    storm_radius_grid_km,
)
from src.wind_metrics import RADIAL_METRIC_NAMES
from geo2wf.config import compose_config
from train import build_model


def _batch(
    *,
    batch_size: int = 2,
    condition_channels: int = 3,
    height: int = 17,
    width: int = 19,
) -> dict[str, torch.Tensor]:
    era5_physical = torch.full((batch_size, 1, height, width), 20.0)
    return {
        "condition": torch.rand(batch_size, condition_channels, height, width),
        "condition_mask": torch.ones(batch_size, 1, height, width, dtype=torch.bool),
        "target": torch.full((batch_size, 1, height, width), 0.25),
        "target_physical": era5_physical + 2.0,
        "target_mask": torch.ones(batch_size, 1, height, width, dtype=torch.bool),
        "era5_wind_speed": torch.full((batch_size, 1, height, width), 0.25),
        "era5_wind_speed_physical": era5_physical,
        "era5_wind_speed_mask": torch.ones(
            batch_size, 1, height, width, dtype=torch.bool
        ),
        "center": torch.zeros(batch_size, 2),
        "target_bounds": torch.tensor([-1.0, 1.0, -1.0, 1.0]).repeat(batch_size, 1),
    }


def _model(**kwargs) -> ERA5ResidualRegressor:
    return ERA5ResidualRegressor(
        condition_channels=3,
        base_channels=4,
        channel_mults=(1, 2),
        **kwargs,
    )


def test_zero_initialized_residual_is_exact_deterministic_era5_baseline() -> None:
    model = _model().eval()
    batch = _batch()
    inference_batch = {
        key: value
        for key, value in batch.items()
        if key not in {"target", "target_physical", "target_mask"}
    }

    first = model.predict_physical(inference_batch)
    second = model.predict_physical(inference_batch)

    assert first.shape == batch["target_physical"].shape
    assert torch.equal(first, batch["era5_wind_speed_physical"])
    assert torch.equal(first, second)


def test_residual_unet_preserves_odd_spatial_shape() -> None:
    model = _model()
    batch = _batch(height=17, width=19)

    residual = model.predict_residual_ms(batch)

    assert residual.shape == (2, 1, 17, 19)


def test_masked_huber_loss_ignores_invalid_pixels() -> None:
    prediction = torch.tensor([[[[0.0, 2.0, 100.0]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[True, True, False]]]])

    loss = masked_huber_loss(prediction, target, mask, delta=1.0)
    loss.backward()

    assert torch.allclose(loss, torch.tensor(0.75))
    assert prediction.grad is not None
    assert prediction.grad[0, 0, 0, 2] == 0


def test_continuous_high_wind_weights_are_smooth_and_masked() -> None:
    target = torch.tensor([[[[20.0, 25.0, 34.0, 43.0, 50.0]]]])
    mask = torch.tensor([[[[True, True, True, True, False]]]])

    weights = continuous_high_wind_pixel_weights(
        target,
        mask,
        start_ms=25.0,
        full_ms=43.0,
        maximum_weight=4.0,
    )

    assert torch.allclose(
        weights,
        torch.tensor([[[[1.0, 1.0, 2.5, 4.0, 0.0]]]]),
    )


def test_robust_peak_loss_uses_top_fraction_and_backpropagates() -> None:
    prediction = torch.tensor([[[[0.0, 1.0, 5.0, 7.0]]]], requires_grad=True)
    target = torch.tensor([[[[0.0, 2.0, 8.0, 10.0]]]])
    mask = torch.ones_like(target, dtype=torch.bool)

    loss = robust_top_fraction_peak_loss(
        prediction,
        target,
        mask,
        top_fraction=0.5,
        minimum_pixels=1,
        delta_ms=1.0,
    )
    loss.backward()

    # Robust peaks are mean([5, 7])=6 and mean([8, 10])=9.
    assert torch.allclose(loss, torch.tensor(2.5))
    assert prediction.grad is not None
    assert torch.equal(
        prediction.grad != 0,
        torch.tensor([[[[False, False, True, True]]]]),
    )


def test_storm_radius_grid_places_center_pixel_at_zero_radius() -> None:
    reference = torch.zeros(1, 1, 3, 3)

    radius = storm_radius_grid_km(
        reference,
        center=torch.tensor([[0.0, 0.0]]),
        bounds=torch.tensor([[-1.5, 1.5, -1.5, 1.5]]),
    )

    assert radius.shape == (1, 3, 3)
    assert radius[0, 1, 1] == pytest.approx(0.0, abs=1e-5)
    assert radius[0, 0, 0] > radius[0, 1, 1]


def test_training_radius_uses_augmented_distance_channel() -> None:
    reference = torch.zeros(1, 1, 3, 3)
    center = torch.tensor([[0.0, -1.0]])
    bounds = torch.tensor([[-1.5, 1.5, -1.5, 1.5]])
    metadata_radius = storm_radius_grid_km(reference, center, bounds)
    flipped_radius = torch.flip(metadata_radius, dims=(-1,))
    condition = torch.zeros(1, 4, 3, 3)
    condition[:, -4] = flipped_radius / metadata_radius.amax()

    training_radius = ERA5ResidualRegressor._training_radius_grid(
        {
            "condition": condition,
            "center": center,
            "target_bounds": bounds,
        },
        reference,
    )

    assert torch.allclose(training_radius, flipped_radius)


def test_radial_profile_loss_compares_annular_means() -> None:
    prediction = torch.zeros(1, 1, 1, 4, requires_grad=True)
    target = torch.tensor([[[[2.0, 2.0, 4.0, 4.0]]]])
    mask = torch.ones_like(target, dtype=torch.bool)
    radius = torch.tensor([[[5.0, 5.0, 15.0, 15.0]]])

    loss = radial_profile_huber_loss(
        prediction,
        target,
        mask,
        radius,
        max_radius_km=20.0,
        radial_bin_km=10.0,
        minimum_bin_pixels=2,
        delta_ms=1.0,
    )
    loss.backward()

    # The two annular errors are 2 and 4 m/s: Huber values 1.5 and 3.5.
    assert torch.allclose(loss, torch.tensor(2.5))
    assert prediction.grad is not None
    assert torch.all(prediction.grad < 0)


def test_soft_exceedance_area_loss_rewards_matching_threshold_areas() -> None:
    target = torch.tensor([[[[0.0, 20.0, 40.0, 50.0]]]])
    mask = torch.ones_like(target, dtype=torch.bool)
    matching_prediction = target.clone().requires_grad_()
    weak_prediction = torch.zeros_like(target)

    matching_loss = soft_exceedance_area_loss(
        matching_prediction,
        target,
        mask,
        thresholds_ms=(17.0, 33.0, 43.0),
        temperature_ms=2.0,
    )
    weak_loss = soft_exceedance_area_loss(
        weak_prediction,
        target,
        mask,
        thresholds_ms=(17.0, 33.0, 43.0),
        temperature_ms=2.0,
    )
    matching_loss.backward()

    assert matching_loss < weak_loss
    assert matching_prediction.grad is not None
    assert torch.isfinite(matching_prediction.grad).all()


class _ConstantResidual(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, :1] * 0.0 + self.value


def test_training_loss_weakly_anchors_residual_outside_sar_swath() -> None:
    model = _model(
        huber_delta_ms=1.0,
        off_swath_anchor_weight=0.1,
    )
    model.model = _ConstantResidual(2.0)
    batch = _batch(batch_size=1, height=2, width=2)
    batch["target_physical"] = batch["era5_wind_speed_physical"] + 2.0
    batch["target_mask"] = torch.tensor([[[[True, True], [False, False]]]])

    with patch.object(model, "log"):
        loss = model.training_step(batch, 0)

    # The observed residual is exact (zero reconstruction loss). Off-swath,
    # Huber(2, delta=1) = 1.5, weighted by 0.1.
    assert torch.allclose(loss, torch.tensor(0.15))


def test_training_step_combines_and_logs_peak_aware_objectives() -> None:
    model = _model(
        huber_delta_ms=1.0,
        off_swath_anchor_weight=0.0,
        peak_loss_weight=0.1,
        peak_inner_core_radius_km=None,
        radial_profile_loss_weight=0.2,
        exceedance_area_loss_weight=0.3,
    )
    model.model = _ConstantResidual(0.0)
    batch = _batch(batch_size=1, height=2, width=2)

    with (
        patch.object(model, "log") as log_metric,
        patch(
            "src.ERA5Residual.robust_top_fraction_peak_loss",
            return_value=torch.tensor(2.0),
        ),
        patch(
            "src.ERA5Residual.radial_profile_huber_loss",
            return_value=torch.tensor(3.0),
        ),
        patch(
            "src.ERA5Residual.soft_exceedance_area_loss",
            return_value=torch.tensor(4.0),
        ),
    ):
        loss = model.training_step(batch, 0)

    # Huber(2, delta=1)=1.5, plus 0.1*2 + 0.2*3 + 0.3*4.
    assert torch.allclose(loss, torch.tensor(3.5))
    logged_names = {call.args[0] for call in log_metric.call_args_list}
    assert {
        "train/unweighted_reconstruction_loss",
        "train/high_wind_pixel_weight_mean",
        "train/robust_peak_loss",
        "train/radial_profile_loss",
        "train/exceedance_area_loss",
    } <= logged_names


def test_validation_statistics_cover_all_calls_in_physical_units() -> None:
    model = _model()
    first = _batch(batch_size=1, height=2, width=2)
    second = _batch(batch_size=1, height=2, width=2)
    first["target_physical"].fill_(21.0)
    second["target_physical"].fill_(23.0)

    model.on_validation_epoch_start()
    model.validation_step(first, 0, 0)
    model.validation_step(second, 1, 0)

    statistics = model._validation_statistics
    assert statistics[0] == 8  # every valid pixel from both batches
    assert statistics[1] == 16  # four * 1 m/s plus four * 3 m/s
    assert statistics[5] == 16  # zero-init model and ERA5 are identical
    assert statistics[10] == 2  # one robust peak per image
    assert statistics[11] == 4  # robust-peak absolute errors: 1 + 3 m/s
    assert statistics[12] == -4  # both robust peaks are underestimated


def test_validation_tracks_hard_exceedance_area_error_by_threshold() -> None:
    model = _model(exceedance_area_thresholds_ms=(25.0,))
    batch = _batch(batch_size=1, height=2, width=2)
    batch["target_physical"] = torch.tensor([[[[30.0, 30.0], [10.0, 10.0]]]])

    model.on_validation_epoch_start()
    model.validation_step(batch, 0, 0)

    assert torch.allclose(
        model._validation_exceedance_statistics,
        torch.tensor([[1.0, 0.5, -0.5, 0.5]], dtype=torch.float64),
    )
    with patch.object(model, "log") as log_metric:
        model.on_validation_epoch_end()
    logged_names = {call.args[0] for call in log_metric.call_args_list}
    assert "val/r25_area_mae_fraction" in logged_names
    assert "val/r25_area_bias_fraction" in logged_names
    assert "val/era5_r25_area_mae_fraction" in logged_names


def test_peak_structure_score_combines_peak_profile_and_area_errors() -> None:
    model = _model()
    statistics = torch.zeros(model._STAT_COUNT, dtype=torch.float64)
    statistics[10] = 2.0
    statistics[11] = 4.0
    radial_statistics = torch.zeros(len(RADIAL_METRIC_NAMES), 2, dtype=torch.float64)
    radial_index = RADIAL_METRIC_NAMES.index("radial_profile_mae_ms")
    radial_statistics[radial_index] = torch.tensor([6.0, 2.0])
    exceedance_statistics = torch.zeros(
        len(model.exceedance_area_thresholds_ms), 4, dtype=torch.float64
    )
    exceedance_statistics[:, 0] = 2.0
    exceedance_statistics[:, 1] = 0.4

    with patch.object(model, "log") as log_metric:
        model._log_peak_structure_score(
            "val",
            statistics,
            radial_statistics,
            exceedance_statistics,
        )

    log_metric.assert_called_once()
    assert log_metric.call_args.args[0] == "val/peak_structure_score"
    assert log_metric.call_args.args[1] == pytest.approx(4.75)


def test_validation_logs_physical_reconstructions_for_val_and_train() -> None:
    model = _model(validation_reconstruction_batches=1)
    batch = _batch(batch_size=1, height=2, width=2)

    with patch("src.ERA5Residual.log_wandb_reconstruction") as log_images:
        model.validation_step(batch, 0, 0)
        model.validation_step(batch, 1, 0)
        model.validation_step(batch, 0, 1)

    assert log_images.call_count == 2
    val_call, train_call = log_images.call_args_list
    assert val_call.args[:2] == (model, batch)
    assert torch.equal(val_call.args[2], batch["era5_wind_speed_physical"])
    assert val_call.kwargs["wandb_key"] == "images/val_reconstruction"
    assert val_call.kwargs["target_batch"] is batch["target_physical"]
    assert train_call.args[:2] == (model, batch)
    assert torch.equal(train_call.args[2], batch["era5_wind_speed_physical"])
    assert train_call.kwargs["wandb_key"] == "images/train_reconstruction"
    assert train_call.kwargs["target_batch"] is batch["target_physical"]


def test_validation_can_disable_images_without_disabling_statistics() -> None:
    model = _model(
        validation_reconstruction_batches=1,
        log_reconstruction_images=False,
    )
    batch = _batch(batch_size=1, height=2, width=2)

    model.on_validation_epoch_start()
    with patch("src.ERA5Residual.log_wandb_reconstruction") as log_images:
        model.validation_step(batch, 0, 0)
        model.validation_step(batch, 0, 1)

    log_images.assert_not_called()
    assert model._validation_statistics[0] > 0


def test_train_builder_selects_residual_model_and_eye_checkpoint_metric() -> None:
    config = compose_config(
        [
            "model=deterministic_residual",
            "model.condition_channels=3",
            "model.base_channels=4",
            "model.channel_mults=[1,2]",
            "model.off_swath_anchor_weight=0.02",
            "model.validation_reconstruction_batches=2",
        ]
    )

    model = build_model(config)

    assert isinstance(model, ERA5ResidualRegressor)
    assert model.checkpoint_monitor == "val/eye_structure_score"
    assert model.off_swath_anchor_weight == pytest.approx(0.02)
    assert model.validation_reconstruction_batches == 2
    scheduler_config = model.configure_optimizers()["lr_scheduler"]
    assert scheduler_config["monitor"] == "val/peak_structure_score"


def test_residual_training_config_builds() -> None:
    config = compose_config(["experiment=intensity_comparison_unet"])

    model = build_model(config)

    assert isinstance(model, ERA5ResidualRegressor)
    assert model.condition_channels == 23
    assert config["trainer"]["checkpoint"]["monitor"] == "val/eye_structure_score"
    assert config["trainer"]["limit_val_batches"] == 1.0


def test_model_without_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="legacy model config has no _target_"):
        build_model({"model": {"type": "not-a-model"}})
