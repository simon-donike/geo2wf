from __future__ import annotations

from unittest.mock import patch

import pytorch_lightning as pl
import pytest
import torch
import torch.nn as nn

from src.ERA5Residual import ERA5ResidualRegressor
from src.ERA5ResidualDiffusion import (
    ERA5ResidualDiffusion,
    load_frozen_deterministic_baseline,
)
from train import build_model, load_config


def _batch(
    *,
    batch_size: int = 1,
    condition_channels: int = 3,
    height: int = 8,
    width: int = 8,
) -> dict[str, torch.Tensor]:
    era5_physical = torch.full((batch_size, 1, height, width), 20.0)
    mask = torch.ones(batch_size, 1, height, width, dtype=torch.bool)
    return {
        "condition": torch.rand(
            batch_size, condition_channels, height, width
        ),
        "condition_mask": mask.clone(),
        "target": torch.full((batch_size, 1, height, width), 0.375),
        "target_physical": torch.full(
            (batch_size, 1, height, width), 30.0
        ),
        "target_mask": mask.clone(),
        "target_norm_offset": torch.zeros(batch_size, 1),
        "target_norm_scale": torch.full((batch_size, 1), 80.0),
        "era5_wind_speed": torch.full(
            (batch_size, 1, height, width), 0.25
        ),
        "era5_wind_speed_physical": era5_physical,
        "era5_wind_speed_mask": mask.clone(),
        "sample_id": [f"sample-{index}" for index in range(batch_size)],
    }


def _model(**kwargs) -> ERA5ResidualDiffusion:
    return ERA5ResidualDiffusion(
        base_condition_channels=4,  # Three raw fields plus aggregate mask.
        generated_channels=1,
        num_timesteps=4,
        schedule="cosine",
        model_dim=4,
        model_dim_mults=(1, 2),
        model_channels=7,  # Noisy residual + six prepared condition fields.
        model_out_dim=1,
        sampling_method="ddim",
        sampling_timesteps=2,
        unobserved_loss_weight=0.1,
        **kwargs,
    )


def test_asinh_residual_transform_is_odd_and_round_trips() -> None:
    model = _model()
    residual = torch.tensor([-80.0, -10.0, 0.0, 10.0, 80.0])

    encoded = model.encode_residual(residual)
    decoded = model.decode_residual(encoded)

    assert torch.allclose(encoded, -torch.flip(encoded, dims=[0]))
    assert torch.allclose(decoded, residual, atol=1e-5)
    assert encoded.min() == -1
    assert encoded.max() == 1


def test_residual_target_uses_joint_mask_and_zero_off_swath_anchor() -> None:
    model = _model()
    batch = _batch(height=2, width=2)
    batch["target_physical"] = torch.tensor(
        [[[[30.0, 99.0], [10.0, 99.0]]]]
    )
    batch["target_mask"] = torch.tensor(
        [[[[True, False], [True, False]]]]
    )
    batch["era5_wind_speed_mask"] = torch.tensor(
        [[[[True, True], [True, False]]]]
    )

    prepared = model._prepare_batch_context(batch)
    target, weight = model._prepare_diffusion_target(
        batch["target"], batch["target_mask"], prepared
    )

    positive = model.encode_residual(torch.tensor(10.0))
    negative = model.encode_residual(torch.tensor(-10.0))
    assert torch.allclose(target[0, 0, 0, 0], positive)
    assert torch.allclose(target[0, 0, 1, 0], negative)
    assert target[0, 0, 0, 1] == 0
    assert target[0, 0, 1, 1] == 0
    assert torch.equal(
        weight,
        torch.tensor([[[[1.0, 0.1], [1.0, 0.0]]]]),
    )


def test_condition_appends_exact_baseline_and_mask() -> None:
    model = _model()
    batch = _batch(height=2, width=2)
    batch["era5_wind_speed_mask"][..., 0, 1] = False
    prepared_batch = model._prepare_batch_context(batch)

    condition = model._prepare_condition(batch["condition"], prepared_batch)

    assert condition.shape == (1, 6, 2, 2)
    baseline_feature = condition[:, -2:-1]
    baseline_mask = condition[:, -1:]
    assert torch.allclose(baseline_feature[..., 0, 0], torch.tensor([[[-0.5]]]))
    assert baseline_feature[..., 0, 1] == 0
    assert torch.equal(
        baseline_mask.bool(), batch["era5_wind_speed_mask"]
    )


def test_sampled_residual_is_added_to_baseline_in_physical_units() -> None:
    model = _model()
    prepared = model._prepare_batch_context(_batch(height=2, width=2))
    encoded_ten_ms = model.encode_residual(torch.tensor(10.0))
    sample = torch.full((1, 1, 2, 2), encoded_ten_ms)

    prediction = model._sample_to_prediction(sample, prepared)

    # ERA5 20 m/s + residual 10 m/s, normalized by the 0..80 m/s target map.
    assert torch.allclose(prediction, torch.full_like(prediction, 0.375))


def test_residual_diffusion_training_step_is_finite() -> None:
    model = _model()
    batch = _batch()

    with patch.object(model, "log"):
        loss = model.training_step(batch, 0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_validation_statistics_compare_reconstruction_to_selected_baseline() -> None:
    model = _model()
    batch = _batch(height=2, width=2)
    # Prediction 25, target 30, ERA5 baseline 20 m/s.
    prediction = torch.full((1, 1, 2, 2), 25.0 / 80.0)

    model.on_validation_epoch_start()
    model._accumulate_physical_statistics(
        model._validation_physical_statistics, prediction, batch
    )

    assert torch.equal(
        model._validation_baseline_statistics,
        torch.tensor([4.0, 20.0, 40.0], dtype=torch.float64),
    )


class _FixedDeterministicBaseline(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def predict_physical(self, batch):
        return torch.ones_like(batch["era5_wind_speed_physical"]) * self.value


def test_optional_deterministic_baseline_is_frozen_and_used() -> None:
    baseline = _FixedDeterministicBaseline(24.0)
    model = _model(
        baseline_source="deterministic",
        baseline_model=baseline,
    )
    batch = _batch(height=2, width=2)
    batch["target_physical"].fill_(30.0)

    prepared = model._prepare_batch_context(batch)
    target, _ = model._prepare_diffusion_target(
        batch["target"], batch["target_mask"], prepared
    )

    assert not next(baseline.parameters()).requires_grad
    assert torch.allclose(
        target,
        torch.full_like(target, model.encode_residual(torch.tensor(6.0))),
    )


def test_frozen_baseline_checkpoint_loader(tmp_path) -> None:
    baseline = ERA5ResidualRegressor(
        condition_channels=3,
        base_channels=4,
        channel_mults=(1, 2),
    )
    checkpoint = tmp_path / "baseline.ckpt"
    torch.save(
        {
            "state_dict": baseline.state_dict(),
            "hyper_parameters": dict(baseline.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        checkpoint,
    )

    loaded = load_frozen_deterministic_baseline(checkpoint)

    assert isinstance(loaded, ERA5ResidualRegressor)
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert not loaded.training


def test_train_builder_selects_residual_diffusion() -> None:
    config = {
        "model": {
            "type": "diffusion_residual",
            "in_channels": 4,
            "out_channels": 1,
            "num_timesteps": 4,
            "schedule": "cosine",
            "sampling": {"method": "ddim", "timesteps": 2},
            "sparse_target": {"unobserved_loss_weight": 0.05},
            "residual": {"baseline": {"source": "era5"}},
            "unet": {
                "dim": 4,
                "dim_mults": [1, 2],
                "channels": 7,
                "out_dim": 1,
            },
        },
        "optimization": {"ema": {"enabled": True, "decay": 0.9}},
    }

    model = build_model(config)

    assert isinstance(model, ERA5ResidualDiffusion)
    assert model.model.condition_channels == 6
    assert model.ema_model is not None
    assert model.unobserved_loss_weight == pytest.approx(0.05)


def test_deterministic_builder_requires_checkpoint() -> None:
    config = {
        "model": {
            "type": "diffusion_residual",
            "residual": {"baseline": {"source": "deterministic"}},
        }
    }

    with pytest.raises(ValueError, match="requires.*checkpoint_path"):
        build_model(config)


def test_residual_diffusion_training_config_builds() -> None:
    config = load_config(
        "configs/config_geo_sar_10bands_era5_diffusion_residual.yaml"
    )

    model = build_model(config)

    assert isinstance(model, ERA5ResidualDiffusion)
    assert model.base_condition_channels == 20
    assert model.model.condition_channels == 22
    assert model.model.model.channels == 23


def test_deterministic_residual_diffusion_preset_requires_external_checkpoint() -> None:
    config = load_config(
        "configs/config_geo_sar_10bands_era5_diffusion_residual_deterministic.yaml"
    )

    baseline = config["model"]["residual"]["baseline"]
    assert baseline["source"] == "deterministic"
    assert baseline["checkpoint_path"].endswith("epoch=053-step=13122.ckpt")
    assert config["export"]["grid_size"] == 256
    assert config["data"]["target_size"] == [256, 256]
    assert config["data"]["center_crop_size"] == [192, 192]
    assert config["trainer"]["max_epochs"] == 5000
    assert config["trainer"]["checkpoint"]["monitor"] == "val/loss"
