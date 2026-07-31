from __future__ import annotations

import json

import pandas as pd

from scripts.collect_ablation_results import (
    collect_calibration,
    collect_inference,
)


def test_collector_scores_inference_aggregations_and_intensity_bins(tmp_path) -> None:
    run_dir = tmp_path / "inference" / "guidance_1p2"
    storm_dir = run_dir / "AL012025"
    storm_dir.mkdir(parents=True)
    metadata = {
        "status": "complete",
        "guidance_scale": 1.2,
        "ensemble_size": 10,
        "ensemble_kind": "stochastic_members_single_checkpoint",
        "aggregation": {"legacy_output_columns": "median"},
        "checkpoint": {"sha256": "abc"},
        "include_test_in_train": True,
    }
    (run_dir / "run-metadata.json").write_text(json.dumps(metadata))
    pd.DataFrame(
        {
            "ibtracs_msw_ms": [20.0, 50.0],
            "output_msw_ms_member_median": [18.0, 40.0],
            "output_msw_ms_member_mean": [19.0, 42.0],
            "output_msw_ms_member_p10": [15.0, 35.0],
            "output_msw_ms_member_p90": [25.0, 45.0],
        }
    ).to_csv(storm_dir / "inference-summary.csv", index=False)

    rows = collect_inference(tmp_path)

    overall = next(
        row
        for row in rows
        if row.get("prediction_column") == "output_msw_ms_member_median"
        and row["metric_scope"] == "overall"
    )
    intense = next(
        row
        for row in rows
        if row.get("prediction_column") == "output_msw_ms_member_median"
        and row.get("target_intensity_lower_ms") == 43.0
    )
    interval = next(row for row in rows if row.get("aggregation") == "member_p10_p90")
    assert overall["guidance_scale"] == 1.2
    assert overall["metric.bias_ms"] == -6.0
    assert intense["metric.bias_ms"] == -10.0
    assert interval["metric.coverage"] == 0.5


def test_collector_preserves_calibration_protocol_and_bin_metrics(tmp_path) -> None:
    output = tmp_path / "calibration" / "guidance_1p5" / "msw" / "isotonic"
    output.mkdir(parents=True)
    (output / "calibration.json").write_text(
        json.dumps(
            {
                "prediction_column": "output_msw_ms_member_median",
                "target_column": "ibtracs_msw_ms",
                "fit_count": 20,
                "storms": ["A", "B"],
                "model": {"method": "isotonic"},
            }
        )
    )
    evaluation = {
        "raw": {
            "count": 20,
            "bias_ms": -4.0,
            "mae_ms": 5.0,
            "rmse_ms": 6.0,
            "correlation": 0.8,
            "by_target_intensity": [
                {
                    "lower_ms": 43.0,
                    "upper_ms": 60.0,
                    "count": 4,
                    "bias_ms": -10.0,
                    "mae_ms": 10.0,
                    "rmse_ms": 11.0,
                }
            ],
        },
        "calibrated_in_sample": None,
        "calibrated_leave_one_storm_out": None,
    }
    (output / "calibration-evaluation.json").write_text(json.dumps(evaluation))

    rows = collect_calibration(tmp_path)

    assert len(rows) == 2
    assert rows[0]["evaluation_protocol"] == "uncalibrated"
    assert rows[0]["guidance_scale"] == 1.5
    assert rows[0]["calibration_method"] == "isotonic"
    assert rows[1]["metric_scope"] == "target_intensity_bin"
    assert rows[1]["metric.bias_ms"] == -10.0
