#!/usr/bin/env python3
"""Evaluate one checkpoint on a common PMW-matched validation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytorch_lightning as pl
import torch

from data import PairedDataModule
from train import build_model, load_config, resolve_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/geotiff/geo_sar_10bands_era5_v2_pmw"),
    )
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--pmw-max-time-gap-hours", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--limit-batches", type=float, default=1.0)
    args = parser.parse_args()
    if args.pmw_max_time_gap_hours <= 0:
        parser.error("--pmw-max-time-gap-hours must be positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _evaluation_rows(dataset) -> dict[str, object]:
    """Fingerprint the ordered validation rows for exact cohort comparisons."""
    columns = [
        column
        for column in ("sample_id", "storm_id", "pmw_path", "pmw_dt_minutes")
        if column in dataset.samples
    ]
    rows = dataset.samples.loc[:, columns].fillna("").astype(str)
    serialized = rows.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return {
        "count": len(rows),
        "columns": columns,
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    for path in (args.config, args.checkpoint, args.data_root):
        if not path.exists():
            raise FileNotFoundError(path)
    config = load_config(str(args.config))
    data = config.setdefault("data", {})
    data["root"] = str(args.data_root)
    data["stats_file"] = str(args.stats or args.data_root / "stats.json")
    # Current controls load PMW only to select the exact same rows; PMW
    # candidates additionally have pmw_as_condition in their own config.
    data["include_pmw"] = True
    data["max_pmw_time_gap_hours"] = float(args.pmw_max_time_gap_hours)
    config = resolve_runtime_config(config)

    datamodule = PairedDataModule.from_config(config)
    datamodule.setup("fit")
    validation_loader = datamodule.val_dataloader()[0]
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.device,
        logger=False,
        enable_checkpointing=False,
        deterministic=True,
        limit_val_batches=args.limit_batches,
    )
    results = trainer.validate(model, dataloaders=validation_loader, verbose=False)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": _sha256(args.checkpoint),
        },
        "data_root": str(args.data_root.resolve()),
        "split": data.get("val_split", "val"),
        "samples": len(datamodule.val_dataset),
        "evaluation_rows": _evaluation_rows(datamodule.val_dataset),
        "pmw_max_time_gap_hours": args.pmw_max_time_gap_hours,
        "limit_batches": args.limit_batches,
        "pmw_as_condition": bool(data.get("pmw_as_condition", False)),
        "training_cohort_note": (
            "Candidate and current-control checkpoints may have different training "
            "cohorts; this report only equalizes evaluation rows."
        ),
        "metrics": results[0] if results else {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(f"Wrote {args.output} ({payload['samples']} matched validation samples)")


if __name__ == "__main__":
    main()
