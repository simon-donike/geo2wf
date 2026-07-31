#!/usr/bin/env python3
"""Measure Stage-2 residual tails and propose data-derived transform scales."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import PairedDataModule  # noqa: E402
from src.ERA5Residual import ERA5ResidualRegressor  # noqa: E402
from train import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _quantiles(
    values: torch.Tensor | np.ndarray | list[float],
    probabilities: list[float],
) -> dict[str, float]:
    """Compute exact quantiles without Torch's 2**24-element kernel limit."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    result = np.quantile(np.asarray(values, dtype=np.float32), probabilities)
    return {
        f"q{probability * 100:g}": float(value)
        for probability, value in zip(probabilities, result)
    }


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    datamodule = PairedDataModule.from_config(config)
    datamodule.setup("fit")
    loader = DataLoader(
        datamodule.train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model = ERA5ResidualRegressor.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu"
    )
    device = torch.device(args.device)
    model.eval().to(device)
    residuals = []
    image_robust_peaks = []
    image_maxima = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = _move(batch, device)
            baseline = model.predict_physical(batch)
            target = batch["target_physical"].to(baseline)
            mask = model._valid_mask(batch, baseline).bool()
            residual = target - baseline
            residuals.append(residual[mask].detach().cpu())
            for sample_index in range(residual.shape[0]):
                sample = residual[sample_index][mask[sample_index]]
                if not sample.numel():
                    continue
                image_robust_peaks.append(float(torch.quantile(sample, 0.995)))
                image_maxima.append(float(sample.max()))
    if not residuals:
        raise RuntimeError("no jointly valid residuals were found")
    residual = torch.cat(residuals)
    absolute = residual.abs()
    probabilities = [0.001, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 0.999]
    clip_q999 = float(np.quantile(absolute.detach().cpu().numpy(), 0.999))
    # Round upward to a convenient 5 m/s experimental boundary.
    proposed_clip = max(5.0, math.ceil(clip_q999 / 5.0) * 5.0)
    payload = {
        "schema_version": 1,
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "include_test_in_train": datamodule.include_test_in_train,
        "valid_pixel_count": int(residual.numel()),
        "image_count": len(image_maxima),
        "signed_residual_ms": _quantiles(residual, probabilities),
        "absolute_residual_ms": _quantiles(absolute, probabilities),
        "image_residual_robust_peak_ms": _quantiles(
            torch.tensor(image_robust_peaks), probabilities
        ),
        "image_residual_max_ms": _quantiles(torch.tensor(image_maxima), probabilities),
        "recommended_linear_clip_ms": proposed_clip,
        "recommendation_basis": "ceil(abs residual q99.9 / 5) * 5",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
