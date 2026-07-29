from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.PixelDiffusion import PixelDiffusionConditional


def _helper() -> PixelDiffusionConditional:
    return object.__new__(PixelDiffusionConditional)


def test_input_and_output_transforms_are_inverse_on_unit_interval() -> None:
    module = _helper()
    values = torch.tensor([-1.0, 0.0, 0.25, 0.5, 1.0, 2.0])

    model_values = module.input_T(values)
    restored = module.output_T(model_values)

    assert torch.allclose(model_values, torch.tensor([-1.0, -1.0, -0.5, 0.0, 1.0, 1.0]))
    assert torch.allclose(restored, values.clamp(0, 1))


def test_unpack_batch_supports_dict_and_tuple_batches() -> None:
    module = _helper()
    condition = torch.zeros(1, 1, 2, 2)
    target = torch.ones(1, 1, 2, 2)
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)

    assert module._unpack_batch((condition, target)) == (condition, target, None)
    assert module._unpack_batch(
        {"condition": condition, "target": target, "target_mask": mask}
    ) == (condition, target, mask)


def test_prepare_condition_appends_single_validity_channel() -> None:
    module = _helper()
    condition = torch.tensor([[[[0.0, 0.5], [1.0, 2.0]]]])
    condition_mask = torch.tensor([[[[True, True], [False, True]], [[True, False], [True, True]]]])

    prepared = module._prepare_condition(
        condition,
        {"condition_mask": condition_mask},
    )

    assert prepared.shape == (1, 2, 2, 2)
    assert torch.allclose(prepared[:, :1], module.input_T(condition))
    assert torch.equal(
        prepared[:, 1:].bool(),
        torch.tensor([[[[True, False], [False, True]]]]),
    )


def test_prepare_target_mask_broadcasts_single_channel_mask() -> None:
    module = _helper()
    mask = torch.tensor([[[True, False], [False, True]]])
    reference = torch.zeros(1, 3, 2, 2)

    prepared = module._prepare_target_mask(mask, reference)

    assert prepared.shape == reference.shape
    assert prepared.dtype == reference.dtype
    assert torch.equal(prepared[:, 0].bool(), mask)
    assert torch.equal(prepared[:, 1].bool(), mask)
    assert torch.equal(prepared[:, 2].bool(), mask)


def test_masked_reconstruction_metrics_only_score_valid_pixels() -> None:
    module = _helper()
    pred = torch.zeros(1, 1, 8, 8)
    target = torch.zeros_like(pred)
    pred[:, :, :, 4:] = 1.0
    target[:, :, :, 4:] = 0.5
    mask = torch.zeros_like(pred, dtype=torch.bool)
    mask[:, :, :, 4:] = True

    psnr, ssim, l1 = module._compute_reconstruction_metrics(pred, target, mask)

    assert torch.isfinite(psnr)
    assert torch.isfinite(ssim)
    assert torch.allclose(l1, torch.tensor(0.5))


def test_sparse_target_completion_uses_era5_with_weak_loss_weight() -> None:
    module = _helper()
    module.sparse_target_fill = "era5"
    module.unobserved_loss_weight = 0.1
    target = torch.tensor([[[[0.8, 0.0], [0.0, 0.2]]]])
    target_mask = torch.tensor([[[[True, False], [False, True]]]])
    era5 = torch.tensor([[[[0.3, 0.4], [0.6, 0.7]]]])
    era5_mask = torch.tensor([[[[True, True], [False, True]]]])

    dense, weight = module._prepare_diffusion_target(
        target,
        target_mask,
        {
            "target_mask": target_mask,
            "era5_wind_speed": era5,
            "era5_wind_speed_mask": era5_mask,
        },
    )

    assert torch.allclose(
        dense,
        torch.tensor([[[[0.8, 0.4], [0.5, 0.2]]]]),
    )
    assert torch.allclose(
        weight,
        torch.tensor([[[[1.0, 0.1], [0.0, 1.0]]]]),
    )


def test_physical_metrics_invert_target_normalization_and_score_era5() -> None:
    module = _helper()
    prediction = torch.tensor([[[[0.5, 0.75]]]])
    mask = torch.ones_like(prediction, dtype=torch.bool)
    metrics = module._physical_reconstruction_metrics(
        prediction,
        {
            "target_physical": torch.tensor([[[[20.0, 35.0]]]]),
            "target_mask": mask,
            "target_norm_offset": torch.tensor([[0.0]]),
            "target_norm_scale": torch.tensor([[40.0]]),
            "era5_wind_speed_physical": torch.tensor([[[[18.0, 30.0]]]]),
            "era5_wind_speed_mask": mask,
        },
    )

    assert torch.allclose(metrics["reconstruction_mae_ms"], torch.tensor(2.5))
    assert torch.allclose(metrics["era5_mae_ms"], torch.tensor(3.5))
    assert torch.allclose(
        metrics["mae_skill_vs_era5"],
        torch.tensor(1.0 - 2.5 / 3.5),
    )


def test_fixed_initial_noise_is_stable_per_sample_id() -> None:
    module = _helper()
    module.validation_seed = 123
    module.model = SimpleNamespace(generated_channels=1)
    condition = torch.zeros(2, 1, 4, 4)
    batch = {"sample_id": ["storm-a", "storm-b"]}

    first = module._fixed_initial_noise(batch, 0, 0, condition)
    second = module._fixed_initial_noise(batch, 99, 1, condition)

    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])


def test_ema_update_moves_toward_raw_model() -> None:
    module = PixelDiffusionConditional(
        condition_channels=1,
        generated_channels=1,
        num_timesteps=4,
        model_dim=4,
        model_dim_mults=(1, 2),
        model_channels=2,
        model_out_dim=1,
        sampling_method="ddim",
        sampling_timesteps=2,
        ema_decay=0.5,
    )
    raw_parameter = next(module.model.parameters())
    ema_parameter = next(module.ema_model.parameters())
    with torch.no_grad():
        raw_parameter.fill_(2.0)
        ema_parameter.zero_()

    module._update_ema()

    assert torch.allclose(ema_parameter, torch.ones_like(ema_parameter))
    assert int(module._ema_updates) == 1


def test_checkpoint_rejects_silent_noise_schedule_change() -> None:
    linear = PixelDiffusionConditional(
        condition_channels=1,
        generated_channels=1,
        num_timesteps=4,
        schedule="linear",
        model_dim=4,
        model_dim_mults=(1, 2),
        model_channels=2,
        model_out_dim=1,
    )
    cosine = PixelDiffusionConditional(
        condition_channels=1,
        generated_channels=1,
        num_timesteps=4,
        schedule="cosine",
        model_dim=4,
        model_dim_mults=(1, 2),
        model_channels=2,
        model_out_dim=1,
    )

    with pytest.raises(ValueError, match="do not match the configured"):
        cosine.on_load_checkpoint({"state_dict": linear.state_dict()})
