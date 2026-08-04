#!/usr/bin/env python3
"""Run a trained single-field U-Net intensity correction checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo2wf.data.intensity import UNetIntensityDataset  # noqa: E402
from geo2wf.models.intensity_correction import UNetIntensityCorrection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    return args


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def inference_rows(
    model: UNetIntensityCorrection,
    loader: DataLoader,
    *,
    device: str | torch.device,
) -> list[dict[str, object]]:
    model = model.eval().to(device)
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            tensor_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            prediction = model.predict_intensity(tensor_batch)
            values = {
                "raw_unet_max_wind_ms": prediction.raw_unet_max_wind_ms.cpu().tolist(),
                "correction_ms": prediction.correction_ms.cpu().tolist(),
                "output_msw_ms": prediction.output_msw_ms.cpu().tolist(),
                "output_category": prediction.output_category.cpu().tolist(),
            }
            for index, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "storm_id": batch["storm_id"][index],
                        "observation_timestamp": batch["observation_timestamp"][index],
                        **{name: value[index] for name, value in values.items()},
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    for path in (args.cache_root, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
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
    rows = inference_rows(model, loader, device=args.device)
    _write_csv_atomic(pd.DataFrame(rows), args.output)
    provenance = {
        "checkpoint": str(args.checkpoint.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "cache_schema_version": dataset.cache_metadata["schema_version"],
        "scientific_evaluation": dataset.cache_metadata.get(
            "scientific_evaluation", "unspecified"
        ),
        "split": args.split,
        "samples": len(rows),
    }
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output} ({len(rows)} scalar predictions)")


if __name__ == "__main__":
    main()
