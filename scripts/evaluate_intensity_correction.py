#!/usr/bin/env python3
"""Evaluate an intensity-correction checkpoint and scalar calibration baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo2wf.data.intensity import (  # noqa: E402
    UNetIntensityDataset,
    tropical_category_from_wind_ms,
)
from geo2wf.models.intensity_correction import (  # noqa: E402
    UNetIntensityCorrection,
    summarize_intensity_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--comparison-checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Optional image-only or metadata-only checkpoint; repeat as needed.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _model_rows(model, loader, device) -> list[dict[str, object]]:
    model = model.eval().to(device)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            tensor_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            prediction = model.predict_intensity(tensor_batch)
            output = prediction.output_msw_ms.cpu().tolist()
            raw = prediction.raw_unet_max_wind_ms.cpu().tolist()
            predicted_category = prediction.output_category.cpu().tolist()
            target = batch["target_wind_ms"].tolist()
            target_category = batch["target_category"].tolist()
            for index, storm_id in enumerate(batch["storm_id"]):
                rows.append(
                    {
                        "storm_id": storm_id,
                        "prediction_ms": output[index],
                        "target_ms": target[index],
                        "raw_unet_ms": raw[index],
                        "prediction_category": predicted_category[index],
                        "target_category": target_category[index],
                    }
                )
    return rows


def _affine_calibration_rows(cache_root: Path, evaluation_rows):
    training = UNetIntensityDataset(cache_root, "train")
    raw = training.samples["raw_unet_max_wind_ms"].to_numpy(dtype=float)
    target = training.samples["target_wind_ms"].to_numpy(dtype=float)
    design = np.column_stack([raw, np.ones_like(raw)])
    slope, intercept = np.linalg.lstsq(design, target, rcond=None)[0]
    rows = []
    for row in evaluation_rows:
        prediction = max(0.0, slope * float(row["raw_unet_ms"]) + intercept)
        rows.append(
            {
                **row,
                "prediction_ms": prediction,
                "prediction_category": tropical_category_from_wind_ms(prediction),
            }
        )
    return {"slope": float(slope), "intercept_ms": float(intercept)}, rows


def _comparison_paths(values: list[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError("comparison checkpoints must use LABEL=PATH")
        label, path = value.split("=", 1)
        if not label or label in parsed:
            raise ValueError(f"invalid or repeated comparison label: {label!r}")
        parsed[label] = Path(path)
    return parsed


def main() -> None:
    args = parse_args()
    dataset = UNetIntensityDataset(args.cache_root, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = UNetIntensityCorrection.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    model.validate_data_spec(dataset.data_spec)
    rows = _model_rows(model, loader, args.device)
    affine_parameters, affine_rows = _affine_calibration_rows(args.cache_root, rows)
    comparisons = {}
    for label, path in _comparison_paths(args.comparison_checkpoint).items():
        comparison = UNetIntensityCorrection.load_from_checkpoint(
            path, map_location="cpu"
        )
        comparison.validate_data_spec(dataset.data_spec)
        comparisons[label] = {
            "checkpoint": {"path": str(path.resolve()), "sha256": _sha256(path)},
            "metrics": summarize_intensity_rows(
                _model_rows(comparison, loader, args.device)
            ),
        }
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(args.cache_root.resolve()),
        "cache_provenance": dataset.cache_metadata,
        "split": args.split,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": _sha256(args.checkpoint),
        },
        "model": summarize_intensity_rows(rows),
        "affine_calibration": {
            "parameters": affine_parameters,
            "metrics": summarize_intensity_rows(affine_rows),
        },
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} ({len(rows)} evaluation rows)")


if __name__ == "__main__":
    main()
