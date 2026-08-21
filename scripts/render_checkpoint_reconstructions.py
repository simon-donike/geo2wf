#!/usr/bin/env python3
"""Render validation reconstructions from a trained checkpoint to local JPEGs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch

from geo2wf.config import instantiate_datamodule
from geo2wf.inference import CheckpointLoader
from geo2wf.tracking import build_reconstruction_figure
from geo2wf.training import build_model, load_config, resolve_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name-prefix", default="reconstruction")
    parser.add_argument(
        "--batch-index",
        type=int,
        action="append",
        dest="batch_indices",
        help="Zero-based validation batch to render; repeat for multiple figures.",
    )
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def main() -> None:
    args = parse_args()
    if args.max_samples < 1 or args.max_samples > 5:
        raise ValueError("--max-samples must be between 1 and 5")
    batch_indices = sorted(set(args.batch_indices or [0]))
    if batch_indices[0] < 0:
        raise ValueError("--batch-index must be non-negative")
    for path in (args.config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = resolve_runtime_config(load_config(args.config))
    datamodule = instantiate_datamodule(config)
    datamodule.setup("fit")
    loaders = datamodule.val_dataloader()
    loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders
    device = torch.device(args.device)
    model = (
        CheckpointLoader.load(
            config,
            args.checkpoint,
            legacy_factory=build_model,
            strict=True,
        )
        .eval()
        .to(device)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    wanted = set(batch_indices)
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader):
            if batch_index not in wanted:
                if batch_index > batch_indices[-1]:
                    break
                continue
            batch = move_to_device(cpu_batch, device)
            prediction = model.predict_physical(batch)
            bound_prediction = getattr(model, "_bound_prediction", None)
            if callable(bound_prediction):
                prediction = bound_prediction(prediction)
            figure = build_reconstruction_figure(
                batch,
                prediction,
                target_batch=batch["target_physical"],
                physical_wind_output=True,
                max_samples=args.max_samples,
            )
            output = (
                args.output_dir / f"{args.name_prefix}-batch-{batch_index + 1:02d}.jpg"
            )
            figure.savefig(
                output, dpi=110, bbox_inches="tight", pil_kwargs={"quality": 90}
            )
            plt.close(figure)
            written.append(str(output))
            wanted.remove(batch_index)
            if not wanted:
                break
    if wanted:
        raise IndexError(
            f"Validation loader ended before batch indices {sorted(wanted)}"
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "batch_indices": batch_indices,
        "max_samples": args.max_samples,
        "images": written,
    }
    manifest_path = args.output_dir / f"{args.name_prefix}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(written)} reconstruction figures and {manifest_path}")


if __name__ == "__main__":
    main()
