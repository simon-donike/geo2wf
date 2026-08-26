from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.export_training_corpus_manifest import publish


HEADERS = [
    "sample_id",
    "split",
    "storm_id",
    "condition_path",
    "context_path",
    "target_path",
    "geo_timestamp",
    "sar_timestamp",
]


def write_manifest(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADERS)
        writer.writerows(rows)


def test_publish_validates_and_copies_the_complete_manifest(tmp_path: Path) -> None:
    source = tmp_path / "manifest.csv"
    output = tmp_path / "published.csv"
    write_manifest(
        source,
        [
            ["one", "train", "AL01", "geo", "era5", "sar", "t1", "t1"],
            ["two", "test", "EP01", "geo", "era5", "sar", "t2", "t2"],
        ],
    )

    assert publish(source, output) == (2, len(HEADERS))
    assert output.read_bytes() == source.read_bytes()


def test_publish_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    source = tmp_path / "manifest.csv"
    write_manifest(
        source,
        [
            ["same", "train", "AL01", "geo", "era5", "sar", "t1", "t1"],
            ["same", "val", "AL02", "geo", "era5", "sar", "t2", "t2"],
        ],
    )

    with pytest.raises(ValueError, match="not unique"):
        publish(source, tmp_path / "published.csv")
