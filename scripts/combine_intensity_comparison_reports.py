#!/usr/bin/env python3
"""Combine matched ERA5 and no-ERA5 comparison JSON into one Markdown report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _combined_rows(
    with_era5: Mapping[str, Any], without_era5: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for conditioning, payload in (
        ("with_era5", with_era5),
        ("without_era5", without_era5),
    ):
        for subset, key in (
            ("all", "table"),
            ("rapid_intensification", "rapid_intensification_table"),
        ):
            rows.extend(
                {
                    "conditioning": conditioning,
                    "subset": subset,
                    **row,
                }
                for row in _report_rows(payload, key, subset=subset)
            )
    return rows


def _report_rows(
    payload: Mapping[str, Any], key: str, *, subset: str
) -> list[dict[str, Any]]:
    """Add SSIM scene coverage to reports produced before that table column."""

    rows = [dict(row) for row in payload.get(key, [])]
    for row in rows:
        if "field_ssim_scenes" in row:
            continue
        model_key = str(row.get("model_key", ""))
        if subset == "all":
            field = payload.get("models", {}).get(model_key, {}).get("field")
        else:
            field = payload.get("rapid_intensification_field", {}).get(model_key)
        row["field_ssim_scenes"] = (
            field.get("ssim_scenes") if isinstance(field, Mapping) else None
        )
    return rows


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
    with_rows = _report_rows(with_era5, "table", subset="all")
    without_rows = _report_rows(without_era5, "table", subset="all")
    with_ri_rows = _report_rows(
        with_era5,
        "rapid_intensification_table",
        subset="rapid_intensification",
    )
    without_ri_rows = _report_rows(
        without_era5,
        "rapid_intensification_table",
        subset="rapid_intensification",
    )
    if bool(with_ri_rows) != bool(without_ri_rows):
        raise ValueError("ERA5 reports do not have matching RI result coverage")
    if with_era5.get("image_quality") != without_era5.get("image_quality"):
        raise ValueError("ERA5 reports do not use the same image-quality contract")
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
        *(
            [
                "### With ERA5: rapid-intensification phases",
                "",
                *_markdown_result_table(with_ri_rows),
                "",
            ]
            if with_ri_rows
            else []
        ),
        "## Without ERA5",
        "",
        *_markdown_result_table(without_rows),
        "",
        *(
            [
                "### Without ERA5: rapid-intensification phases",
                "",
                *_markdown_result_table(without_ri_rows),
                "",
            ]
            if without_ri_rows
            else []
        ),
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
    with_era5 = _load(args.with_era5)
    without_era5 = _load(args.without_era5)
    report = combined_markdown_report(with_era5, without_era5)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, output)
    rows = _combined_rows(with_era5, without_era5)
    csv_output = output.with_suffix(".csv")
    csv_temporary = csv_output.with_suffix(csv_output.suffix + ".tmp")
    frame = pd.DataFrame(rows)
    if "field_ssim_scenes" in frame:
        frame["field_ssim_scenes"] = frame["field_ssim_scenes"].astype("Int64")
    frame.to_csv(csv_temporary, index=False)
    os.replace(csv_temporary, csv_output)
    json_output = output.with_suffix(".json")
    json_temporary = json_output.with_suffix(json_output.suffix + ".tmp")
    json_temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "split": with_era5["split"],
                "cohort": with_era5["cohort"],
                "rapid_intensification": {
                    "threshold_kt": 30.0,
                    "window_hours": 24.0,
                },
                "image_quality": with_era5.get("image_quality"),
                "sources": {
                    "with_era5": {
                        "path": str(args.with_era5.expanduser().resolve()),
                        "sha256": _sha256(args.with_era5.expanduser().resolve()),
                    },
                    "without_era5": {
                        "path": str(args.without_era5.expanduser().resolve()),
                        "sha256": _sha256(args.without_era5.expanduser().resolve()),
                    },
                },
                "results": rows,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(json_temporary, json_output)
    print(f"Wrote {output}, {csv_output}, and {json_output}")


if __name__ == "__main__":
    main()
