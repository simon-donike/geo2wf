#!/usr/bin/env python3
"""Evaluate a scalar six-hour intensity forecast checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo2wf.data.intensity_forecast import IntensityForecastDataset  # noqa: E402
from geo2wf.models.intensity_forecast import (  # noqa: E402
    IntensityForecastMLP,
    summarize_forecast_rows,
)
from scripts.export_unet_intensity_cache import _sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def forecast_rows(model, loader, device) -> list[dict[str, object]]:
    model = model.eval().to(device)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            tensor_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            prediction = model.predict_forecast(
                tensor_batch["features"], tensor_batch["anchor_wind_ms"]
            )
            rows.extend(model._rows(tensor_batch, prediction))
    return rows


def main() -> None:
    args = parse_args()
    dataset = IntensityForecastDataset(args.cache_root, args.split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = IntensityForecastMLP.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    model.validate_data_spec(dataset.data_spec)
    rows = forecast_rows(model, loader, args.device)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(args.cache_root.resolve()),
        "split": args.split,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": _sha256(args.checkpoint),
        },
        "metrics": summarize_forecast_rows(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} ({len(rows)} forecasts)")


if __name__ == "__main__":
    main()
