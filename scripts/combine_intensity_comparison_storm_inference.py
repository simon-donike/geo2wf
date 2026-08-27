#!/usr/bin/env python3
"""Combine per-storm dense inference jobs into the two report inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_intensity_comparison_storm_inference import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    STORMS,
    _atomic_csv,
    _atomic_json,
    _cohort_fingerprint,
    _summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parts-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "parts"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _combine_regime(parts_root: Path, output_root: Path, regime: str) -> pd.DataFrame:
    label = "with-era5" if regime == "with" else "without-era5"
    frames = []
    metadata: list[dict[str, Any]] = []
    for storm in STORMS:
        roots = [parts_root / f"{regime}-{storm}"]
        if not all(
            (roots[0] / f"{label}{suffix}").is_file() for suffix in (".csv", ".json")
        ):
            roots = sorted(parts_root.glob(f"{regime}-{storm}-shard*"))
        if not roots:
            raise FileNotFoundError(f"missing inference part for {regime}-{storm}")
        for root in roots:
            csv_path = root / f"{label}.csv"
            json_path = root / f"{label}.json"
            if not csv_path.is_file() or not json_path.is_file():
                raise FileNotFoundError(f"missing inference part below {root}")
            frame = pd.read_csv(csv_path)
            if "inference_valid" not in frame:
                frame["inference_valid"] = True
                frame["inference_issue"] = None
            if set(frame["storm_id"].astype(str)) != {storm}:
                raise ValueError(f"{csv_path} does not contain exactly storm {storm}")
            frames.append(frame)
            metadata.append(json.loads(json_path.read_text(encoding="utf-8")))

    frame = pd.concat(frames, ignore_index=True).sort_values(
        ["storm_id", "observation_timestamp", "observation_id"]
    )
    if frame["observation_id"].duplicated().any():
        raise ValueError(f"{regime} inference parts contain duplicate observation IDs")
    checkpoint_sets = {
        json.dumps(item["checkpoints"], sort_keys=True) for item in metadata
    }
    if len(checkpoint_sets) != 1:
        raise ValueError(f"{regime} inference parts used different checkpoints")
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "conditioning": label,
        "interpretation": "dense_validation_storm_case_study",
        "cohort": {
            "samples": len(frame),
            "storms": int(frame["storm_id"].nunique()),
            "storm_ids": list(STORMS),
            "source_splits": sorted(frame["source_split"].unique().tolist()),
            "sha256": _cohort_fingerprint(frame),
            "evaluated_samples": int(frame["inference_valid"].sum()),
            "invalid_samples": int((~frame["inference_valid"]).sum()),
        },
        "target": metadata[0]["target"],
        "checkpoints": metadata[0]["checkpoints"],
        "correction_cache_scientific_evaluation": metadata[0][
            "correction_cache_scientific_evaluation"
        ],
        "parts": [item["cohort"] for item in metadata],
        "metrics": _summarize(frame),
    }
    csv_path = output_root / f"{label}.csv"
    _atomic_csv(frame, csv_path)
    _atomic_json(payload, csv_path.with_suffix(".json"))
    print(f"Wrote {csv_path} ({len(frame)} rows)")
    return frame


def combine(args: argparse.Namespace) -> None:
    with_era5 = _combine_regime(args.parts_root, args.output_root, "with")
    without_era5 = _combine_regime(args.parts_root, args.output_root, "without")
    columns = [
        "observation_id",
        "storm_id",
        "observation_timestamp",
        "target_ms",
        "ri_24h_change_ms",
        "is_rapid_intensification",
    ]
    if (
        not with_era5[columns]
        .reset_index(drop=True)
        .equals(without_era5[columns].reset_index(drop=True))
    ):
        raise ValueError("with-ERA5 and no-ERA5 combined cohorts differ")
    print("Verified identical ERA5/no-ERA5 dense inference cohorts")


def main() -> None:
    combine(parse_args())


if __name__ == "__main__":
    main()
