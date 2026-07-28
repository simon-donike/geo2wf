from __future__ import annotations

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
