#!/usr/bin/env python3
"""Publish the canonical GEO–ERA5–SAR corpus manifest with the documentation."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/geotiff/geo_sar_10bands_era5/manifest.csv"
DEFAULT_OUTPUT = ROOT / "docs/assets/data/training-corpus-manifest.csv"
REQUIRED_COLUMNS = {
    "sample_id",
    "split",
    "storm_id",
    "condition_path",
    "context_path",
    "target_path",
    "geo_timestamp",
    "sar_timestamp",
}


def publish(source: Path, output: Path) -> tuple[int, int]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = REQUIRED_COLUMNS - set(columns)
        if missing:
            raise ValueError(f"Corpus manifest is missing columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Corpus manifest contains no samples")
    invalid_splits = {row["split"] for row in rows} - {"train", "val", "test"}
    if invalid_splits:
        raise ValueError(f"Corpus manifest has unexpected splits: {sorted(invalid_splits)}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Corpus manifest sample_id values are not unique")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    return len(rows), len(columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows, columns = publish(args.source, args.output)
    print(f"Published {rows} samples and {columns} columns to {args.output}")


if __name__ == "__main__":
    main()
