#!/usr/bin/env python3
"""Combine matched ERA5 and no-ERA5 comparison JSON into one Markdown report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_intensity_models import (
    _markdown_result_table,
    _methodology_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-era5", type=Path, required=True)
    parser.add_argument("--without-era5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload.get("table"), list) or not payload["table"]:
        raise ValueError(f"comparison JSON has no non-empty table: {resolved}")
    return payload


def combined_markdown_report(
    with_era5: Mapping[str, Any], without_era5: Mapping[str, Any]
) -> str:
    if with_era5.get("split") != without_era5.get("split"):
        raise ValueError("ERA5 reports use different evaluation splits")
    with_cohort = with_era5.get("cohort", {})
    without_cohort = without_era5.get("cohort", {})
    if with_cohort.get("sha256") != without_cohort.get("sha256"):
        raise ValueError("ERA5 reports do not use the exact same cohort")
    with_bootstrap = with_era5.get("paired_storm_bootstrap", {})
    without_bootstrap = without_era5.get("paired_storm_bootstrap", {})
    for key in ("repetitions", "seed"):
        if with_bootstrap.get(key) != without_bootstrap.get(key):
            raise ValueError(f"ERA5 reports use different bootstrap {key}")

    split = str(with_era5["split"])
    with_rows = with_era5["table"]
    without_rows = without_era5["table"]
    samples = int(with_cohort["samples"])
    storms = int(with_cohort["storms"])
    lines = [
        f"# Intensity model comparison with and without ERA5 ({split})",
        "",
        "The two regimes use the exact same observations and targets. **With ERA5** "
        "adds the available ERA5 context channels; its separate U-Net predicts a "
        "correction to ERA5 wind. **Without ERA5** removes those channels; its "
        "separate U-Net predicts the wind field directly. Models are trained "
        "independently within each regime.",
        "",
        "## With ERA5",
        "",
        *_markdown_result_table(with_rows),
        "",
        "## Without ERA5",
        "",
        *_markdown_result_table(without_rows),
        "",
        *_methodology_markdown(
            split=split,
            samples=samples,
            storms=storms,
            bootstrap_repetitions=int(with_bootstrap["repetitions"]),
            bootstrap_seed=int(with_bootstrap["seed"]),
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = combined_markdown_report(_load(args.with_era5), _load(args.without_era5))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
