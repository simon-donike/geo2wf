#!/usr/bin/env python3
"""Run scalar six-hour maximum-wind forecast inference."""

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

from geo2wf.data.intensity_forecast import IntensityForecastDataset  # noqa: E402
from geo2wf.models.intensity_forecast import IntensityForecastMLP  # noqa: E402


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


def inference_rows(model, loader, device) -> list[dict[str, object]]:
    model = model.eval().to(device)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device)
            anchor = batch["anchor_wind_ms"].to(device)
            prediction = model.predict_forecast(features, anchor)
            output = prediction.output_wind_ms.cpu().tolist()
            delta = prediction.predicted_delta_ms.cpu().tolist()
            for index, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "storm_id": str(batch["storm_id"][index]),
                        "init_timestamp": str(batch["init_timestamp"][index]),
                        "forecast_timestamp": (
                            pd.Timestamp(batch["init_timestamp"][index])
                            + pd.Timedelta(hours=6)
                        ).isoformat(),
                        "anchor_wind_ms": float(anchor[index].cpu()),
                        "predicted_delta_ms": float(delta[index]),
                        "forecast_wind_ms": float(output[index]),
                    }
                )
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
    rows = inference_rows(model, loader, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=args.output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        pd.DataFrame(rows).to_csv(handle, index=False)
    os.replace(temporary, args.output)
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "cache_root": str(args.cache_root.resolve()),
                "split": args.split,
                "samples": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} ({len(rows)} forecasts)")


if __name__ == "__main__":
    main()
