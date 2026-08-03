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
        "condition": torch.rand(batch_size, condition_channels, height, width),
        "condition_mask": mask.clone(),
        "target": torch.full((batch_size, 1, height, width), 0.375),
        "target_physical": torch.full((batch_size, 1, height, width), 30.0),
        "target_mask": mask.clone(),
        "target_norm_offset": torch.zeros(batch_size, 1),
        "target_norm_scale": torch.full((batch_size, 1), 80.0),
        "era5_wind_speed": torch.full((batch_size, 1, height, width), 0.25),
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


def test_linear_residual_transform_scales_clips_and_round_trips() -> None:
    model = _model(
        residual_transform="linear",
        residual_soft_scale_ms=100.0,
        residual_clip_ms=80.0,
    )
    residual = torch.tensor([-120.0, -40.0, 0.0, 40.0, 120.0])

    encoded = model.encode_residual(residual)
    decoded = model.decode_residual(encoded)

    assert torch.allclose(encoded, torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]))
    assert torch.allclose(decoded, residual.clamp(-80.0, 80.0))


def test_residual_transform_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="residual_transform"):
        _model(residual_transform="logarithmic")


def test_residual_target_uses_joint_mask_and_zero_off_swath_anchor() -> None:
    model = _model()
    batch = _batch(height=2, width=2)
    batch["target_physical"] = torch.tensor([[[[30.0, 99.0], [10.0, 99.0]]]])
    batch["target_mask"] = torch.tensor([[[[True, False], [True, False]]]])
    batch["era5_wind_speed_mask"] = torch.tensor([[[[True, True], [True, False]]]])

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
    assert torch.equal(baseline_mask.bool(), batch["era5_wind_speed_mask"])


def test_classifier_free_dropout_can_preserve_baseline_anchor() -> None:
    model = _model(
        condition_dropout_probability=0.5,
        preserve_baseline_condition_on_dropout=True,
    )
    prepared = model._prepare_batch_context(_batch(height=2, width=2))
    condition = model._prepare_condition(prepared["condition"], prepared)

    with patch("torch.rand", return_value=torch.zeros(1, 1, 1, 1)):
        dropped = model._condition_for_training(condition)
    unconditional = model._unconditional_condition(condition)

    assert torch.count_nonzero(dropped[:, :4]) == 0
    assert torch.equal(dropped[:, 4:], condition[:, 4:])
    assert torch.count_nonzero(unconditional[:, :4]) == 0
    assert torch.equal(unconditional[:, 4:], condition[:, 4:])


def test_default_classifier_free_dropout_retains_legacy_behavior() -> None:
    model = _model(condition_dropout_probability=0.5)
    prepared = model._prepare_batch_context(_batch(height=2, width=2))
    condition = model._prepare_condition(prepared["condition"], prepared)

    with patch("torch.rand", return_value=torch.zeros(1, 1, 1, 1)):
        dropped = model._condition_for_training(condition)

    assert torch.count_nonzero(dropped) == 0


def test_guided_prediction_path_uses_anchored_unconditional_condition() -> None:
    model = _model(
        guidance_scale=1.5,
        preserve_baseline_condition_on_dropout=True,
    )
    batch = _batch(height=2, width=2)
    raw_sample = torch.zeros(1, 1, 2, 2)

    with patch.object(model.model, "forward", return_value=raw_sample) as sample:
        returned_raw, prediction = model._predict_batch(batch, batch_idx=0)

    condition = sample.call_args.args[0]
    unconditional = sample.call_args.kwargs["unconditional_condition"]
    assert torch.equal(returned_raw, raw_sample)
    assert prediction.shape == raw_sample.shape
    assert sample.call_args.kwargs["guidance_scale"] == pytest.approx(1.5)
    assert torch.count_nonzero(unconditional[:, :4]) == 0
    assert torch.equal(unconditional[:, 4:], condition[:, 4:])


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


def test_inner_core_mask_uses_geographic_center_and_bounds() -> None:
    model = _model(inner_core_radius_km=20.0)
    reference = torch.zeros(1, 1, 3, 3)

    mask = model._inner_core_mask(
        {
            "center": torch.tensor([[0.0, 0.0]]),
            "target_bounds": torch.tensor([[-1.0, 1.0, -1.0, 1.0]]),
        },
        reference,
    )

    assert mask.sum() == 1
    assert mask[0, 0, 1, 1]


def test_robust_peak_loss_matches_top_fraction_mean() -> None:
    model = _model(robust_peak_fraction=0.5)
    prediction = torch.tensor([[[[1.0, 2.0], [8.0, 10.0]]]])
    target = torch.tensor([[[[1.0, 2.0], [4.0, 6.0]]]])
    mask = torch.ones_like(prediction)

    loss = model._masked_robust_peak_loss(prediction, target, mask)

    assert loss == pytest.approx(4.0)


def test_multiscale_field_loss_is_phase_aware_masked_and_differentiable() -> None:
    model = _model(multiscale_field_kernel_sizes=(1, 3))
    target = torch.zeros(1, 1, 5, 5)
    target[..., 2, 2] = 10.0
    prediction = torch.roll(target, shifts=1, dims=-1).requires_grad_(True)
    mask = torch.ones_like(target)

    # A circular translation preserves Fourier amplitude, while the spatially
    # pooled absolute-field objective still detects the displaced peak.
    spectrum_loss = model._masked_log_spectrum_loss(prediction, target, mask)
    field_loss = model._masked_multiscale_field_loss(prediction, target, mask)
    field_loss.backward()

    assert spectrum_loss.detach().item() == pytest.approx(0.0, abs=1e-6)
    assert field_loss.detach() > 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    masked = mask.clone()
    masked[..., 2, 2:4] = 0
    assert model._masked_multiscale_field_loss(
        prediction.detach(), target, masked
    ) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kernel_sizes",
    [(), (0,), (2,), (1, 4), (True,)],
)
def test_multiscale_field_kernels_must_be_positive_odd_integers(
    kernel_sizes,
) -> None:
    with pytest.raises(ValueError, match="positive odd integers"):
        _model(multiscale_field_kernel_sizes=kernel_sizes)


def test_annular_residual_loss_matches_target_radial_mean_correction() -> None:
    model = _model(
        radial_profile_bins=2,
        radial_profile_max_radius_km=2.0,
    )
    radius_km = torch.tensor([[[[0.5, 0.5, 1.5, 1.5]]]])
    valid_metadata = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    mask = torch.ones_like(radius_km)
    target = torch.tensor([[[[2.0, 2.0, 3.0, 3.0]]]])
    same_annular_means = torch.tensor([[[[4.0, 0.0, 6.0, 0.0]]]])
    missing_correction = torch.zeros_like(target, requires_grad=True)

    with patch.object(
        model,
        "_storm_radius_km",
        return_value=(radius_km, valid_metadata),
    ):
        matched_loss = model._masked_annular_residual_loss(
            same_annular_means, target, mask, {}
        )
        missing_loss = model._masked_annular_residual_loss(
            missing_correction, target, mask, {}
        )
    missing_loss.backward()

    assert matched_loss == pytest.approx(0.0)
    assert missing_loss.detach() == pytest.approx(2.5)
    assert missing_correction.grad is not None
    assert torch.isfinite(missing_correction.grad).all()
    assert (
        model._masked_annular_residual_loss(
            missing_correction.detach(), target, mask, {}
        )
        == 0
    )


def test_absolute_field_structure_losses_are_differentiable() -> None:
    model = _model(
        robust_peak_fraction=0.25,
        radial_profile_bins=2,
        radial_profile_max_radius_km=2.0,
        exceedance_thresholds_ms=(17.0, 33.0),
    )
    target = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3) + 25.0
    prediction = (target.detach() + 2.0).requires_grad_(True)
    mask = torch.ones_like(target)
    metadata = {
        "center": torch.tensor([[0.0, 0.0]]),
        "target_bounds": torch.tensor([[-0.01, 0.01, -0.01, 0.01]]),
    }

    peak_loss = model._masked_robust_peak_loss(prediction, target, mask)
    radial_loss = model._masked_radial_profile_loss(prediction, target, mask, metadata)
    area_loss = model._masked_exceedance_area_loss(prediction, target, mask)
    total = peak_loss + radial_loss + area_loss
    total.backward()

    assert peak_loss.detach().item() == pytest.approx(2.0)
    assert radial_loss.detach().item() == pytest.approx(2.0)
    assert area_loss > 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) > 0
    assert model._masked_radial_profile_loss(prediction.detach(), target, mask, {}) == 0


def test_structural_and_smoothness_objective_is_finite() -> None:
    model = _model(
        min_snr_gamma=5.0,
        condition_dropout_probability=0.1,
        preserve_baseline_condition_on_dropout=True,
        gradient_loss_weight=0.05,
        spectrum_loss_weight=0.05,
        low_frequency_loss_weight=0.1,
        smoothness_loss_weight=0.02,
        multiscale_field_loss_weight=0.05,
        multiscale_field_kernel_sizes=(1, 3),
        annular_residual_loss_weight=0.05,
        robust_peak_loss_weight=0.05,
        robust_peak_fraction=0.1,
        radial_profile_loss_weight=0.05,
        radial_profile_bins=4,
        radial_profile_max_radius_km=200.0,
        exceedance_area_loss_weight=0.05,
        exceedance_thresholds_ms=(17.0, 33.0, 43.0),
        high_wind_loss_weight=2.0,
        high_gradient_loss_weight=2.0,
        low_frequency_kernel_size=3,
    )
    batch = _batch()
    batch["center"] = torch.tensor([[0.0, 0.0]])
    batch["target_bounds"] = torch.tensor([[-1.0, 1.0, -1.0, 1.0]])

    with (
        patch.object(model, "log") as log,
        patch("torch.randint", return_value=torch.zeros(1, dtype=torch.long)),
    ):
        loss = model.training_step(batch, 0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    logged_names = {call.args[0] for call in log.call_args_list}
    assert "train/multiscale_field_loss_ms" in logged_names
    assert "train/annular_residual_loss_ms" in logged_names


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
            "classifier_free_guidance": {
                "preserve_baseline_condition_on_dropout": True
            },
            "sampling": {"method": "ddim", "timesteps": 2},
            "sparse_target": {"unobserved_loss_weight": 0.05},
            "residual": {
                "transform": "linear",
                "baseline": {"source": "era5"},
                "loss": {
                    "robust_peak_weight": 0.2,
                    "robust_peak_fraction": 0.01,
                    "radial_profile_weight": 0.1,
                    "radial_profile_bins": 8,
                    "exceedance_area_weight": 0.3,
                    "exceedance_thresholds_ms": [20.0, 40.0],
                    "exceedance_temperature_ms": 2.0,
                },
            },
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
    assert model.residual_transform == "linear"
    assert model.preserve_baseline_condition_on_dropout
    assert model.robust_peak_loss_weight == pytest.approx(0.2)
    assert model.robust_peak_fraction == pytest.approx(0.01)
    assert model.radial_profile_loss_weight == pytest.approx(0.1)
    assert model.radial_profile_bins == 8
    assert model.exceedance_area_loss_weight == pytest.approx(0.3)
    assert model.exceedance_thresholds_ms == (20.0, 40.0)
    assert model.exceedance_temperature_ms == pytest.approx(2.0)


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
    config = load_config("configs/config_geo_sar_10bands_era5_diffusion_residual.yaml")

    model = build_model(config)

    assert isinstance(model, ERA5ResidualDiffusion)
    assert model.base_condition_channels == 24
    assert model.model.condition_channels == 26
    assert model.model.model.channels == 27


def test_deterministic_residual_diffusion_preset_declares_checkpoint() -> None:
    config = load_config(
        "configs/config_geo_sar_10bands_era5_diffusion_residual_deterministic.yaml"
    )

    baseline = config["model"]["residual"]["baseline"]
    assert baseline["source"] == "deterministic"
    assert isinstance(baseline["checkpoint_path"], str)
    assert baseline["checkpoint_path"].endswith(".ckpt")
    assert config["export"]["grid_size"] == 256
    assert config["data"]["target_size"] == [256, 256]
    assert config["data"]["center_crop_size"] == [192, 192]
    assert config["trainer"]["max_epochs"] == 10000
    assert (
        config["trainer"]["checkpoint"]["monitor"]
        == "val/probabilistic_refinement_score"
    )
    assert config["data"]["include_test_in_train"] is True
    assert config["optimization"]["min_snr_gamma"] == 5.0
    assert config["validation"]["ensemble_size"] == 4
    assert config["validation"]["probabilistic_score_target_sharpness_ratio"] == 0.9
    assert config["model"]["residual"]["loss"]["smoothness_weight"] == 0.02
    assert config["model"]["sampling"]["guidance_scale"] == 1.2
    assert (
        config["model"]["classifier_free_guidance"]["condition_dropout_probability"]
        == 0.1
    )
