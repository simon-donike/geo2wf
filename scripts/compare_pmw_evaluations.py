#!/usr/bin/env python3
"""Compare PMW and current checkpoint reports on an identical validation cohort."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(report: dict[str, Any], name: str) -> float:
    metrics = report.get("metrics", {})
    for key in (f"val/{name}", name):
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"evaluation report has no metric {name!r}")


def _lower_is_better(
    current: dict[str, Any], candidate: dict[str, Any], metric: str
) -> dict[str, Any]:
    current_value = _metric(current, metric)
    candidate_value = _metric(candidate, metric)
    return {
        "metric": metric,
        "direction": "lower",
        "current": current_value,
        "candidate": candidate_value,
        "relative_change": (
            (candidate_value - current_value) / current_value
            if current_value != 0.0
            else None
        ),
        "passed": candidate_value < current_value,
    }


def _validate_common_cohort(
    current: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    current_rows = current.get("evaluation_rows")
    candidate_rows = candidate.get("evaluation_rows")
    if not current_rows or not candidate_rows:
        raise ValueError(
            "both reports must include evaluation_rows; regenerate them with "
            "scripts/evaluate_checkpoint.py"
        )
    fields = ("sha256", "count", "columns")
    mismatches = [
        field for field in fields if current_rows.get(field) != candidate_rows.get(field)
    ]
    if mismatches:
        raise ValueError(
            "evaluation cohorts differ in: " + ", ".join(mismatches)
        )
    if current.get("split") != candidate.get("split"):
        raise ValueError("evaluation splits differ")
    if float(current.get("pmw_max_time_gap_hours")) != float(
        candidate.get("pmw_max_time_gap_hours")
    ):
        raise ValueError("PMW time-gap limits differ")
    if float(current.get("limit_batches", 1.0)) != 1.0 or float(
        candidate.get("limit_batches", 1.0)
    ) != 1.0:
        raise ValueError("promotion comparisons require full validation (limit_batches=1.0)")
    if current.get("pmw_as_condition"):
        raise ValueError("the current report must use a non-PMW control config")
    if not candidate.get("pmw_as_condition"):
        raise ValueError("the candidate report must use a PMW condition config")
    return current_rows


def _criteria(
    stage: int, current: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    if stage == 1:
        criteria = [
            _lower_is_better(current, candidate, "peak_structure_score"),
            _lower_is_better(current, candidate, "robust_peak_mae_ms"),
        ]
        current_mae = _metric(current, "mae_ms")
        candidate_mae = _metric(candidate, "mae_ms")
        criteria.append(
            {
                "metric": "mae_ms",
                "direction": "candidate <= current * 1.02",
                "current": current_mae,
                "candidate": candidate_mae,
                "relative_change": (
                    (candidate_mae - current_mae) / current_mae
                    if current_mae != 0.0
                    else None
                ),
                "passed": candidate_mae <= current_mae * 1.02,
            }
        )
        return criteria

    criteria = [
        _lower_is_better(current, candidate, "probabilistic_refinement_score"),
        _lower_is_better(current, candidate, "ensemble_crps_ms"),
    ]
    baseline_skill = _metric(candidate, "mae_skill_vs_baseline")
    criteria.append(
        {
            "metric": "mae_skill_vs_baseline",
            "direction": "candidate >= 0",
            "current": None,
            "candidate": baseline_skill,
            "relative_change": None,
            "passed": baseline_skill >= 0.0,
        }
    )
    return criteria


def main() -> None:
    args = parse_args()
    current = _load(args.current)
    candidate = _load(args.candidate)
    cohort = _validate_common_cohort(current, candidate)
    criteria = _criteria(args.stage, current, candidate)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "promote": all(item["passed"] for item in criteria),
        "criteria": criteria,
        "evaluation_rows": cohort,
        "current_report": str(args.current.resolve()),
        "candidate_report": str(args.candidate.resolve()),
        "training_cohort_note": (
            "The checkpoints may have different training cohorts; promotion is based "
            "on their identical PMW-matched validation rows."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    verdict = "PROMOTE" if payload["promote"] else "DO NOT PROMOTE"
    print(f"{verdict}: wrote {args.output}")


if __name__ == "__main__":
    main()
