from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.ERA5Residual import ERA5ResidualRegressor, masked_huber_loss
from train import build_model, load_config


def _batch(
    *,
    batch_size: int = 2,
    condition_channels: int = 3,
    height: int = 17,
    width: int = 19,
) -> dict[str, torch.Tensor]:
    era5_physical = torch.full((batch_size, 1, height, width), 20.0)
    return {
        "condition": torch.rand(
            batch_size, condition_channels, height, width
        ),
        "condition_mask": torch.ones(
            batch_size, 1, height, width, dtype=torch.bool
        ),
        "target": torch.full((batch_size, 1, height, width), 0.25),
        "target_physical": era5_physical + 2.0,
        "target_mask": torch.ones(
            batch_size, 1, height, width, dtype=torch.bool
        ),
        "era5_wind_speed": torch.full(
            (batch_size, 1, height, width), 0.25
        ),
        "era5_wind_speed_physical": era5_physical,
        "era5_wind_speed_mask": torch.ones(
            batch_size, 1, height, width, dtype=torch.bool
        ),
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
    batch["target_mask"] = torch.tensor(
        [[[[True, True], [False, False]]]]
    )

    with patch.object(model, "log"):
        loss = model.training_step(batch, 0)

    # The observed residual is exact (zero reconstruction loss). Off-swath,
    # Huber(2, delta=1) = 1.5, weighted by 0.1.
    assert torch.allclose(loss, torch.tensor(0.15))


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


def test_train_builder_selects_residual_model_and_eye_checkpoint_metric() -> None:
    config = {
        "model": {
            "type": "deterministic_residual",
            "condition_channels": 3,
            "residual": {
                "base_channels": 4,
                "channel_mults": [1, 2],
            },
        },
        "optimization": {"off_swath_anchor_weight": 0.02},
        "validation": {"reconstruction_batches": 2},
    }

    model = build_model(config)

    assert isinstance(model, ERA5ResidualRegressor)
    assert model.checkpoint_monitor == "val/eye_structure_score"
    assert model.off_swath_anchor_weight == pytest.approx(0.02)
    assert model.validation_reconstruction_batches == 2
    scheduler_config = model.configure_optimizers()["lr_scheduler"]
    assert scheduler_config["monitor"] == "val/eye_structure_score"


def test_residual_training_config_builds() -> None:
    config = load_config("configs/config_geo_sar_10bands_era5_residual.yaml")

    model = build_model(config)

    assert isinstance(model, ERA5ResidualRegressor)
    assert model.condition_channels == 19
    assert (
        config["trainer"]["checkpoint"]["monitor"]
        == "val/loss"
    )
    assert config["trainer"]["limit_val_batches"] == 1.0


def test_unknown_model_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported model.type"):
        build_model({"model": {"type": "not-a-model"}})
