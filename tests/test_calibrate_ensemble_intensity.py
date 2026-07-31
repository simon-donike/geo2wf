import json
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.calibrate_ensemble_intensity import (
    apply_calibrator,
    error_metrics,
    fit_calibrator,
    interval_metrics,
    main,
)


def test_isotonic_calibration_is_monotone_and_clips_extrapolation() -> None:
    prediction = np.array([10.0, 20.0, 30.0, 40.0])
    target = np.array([8.0, 25.0, 24.0, 60.0])

    model = fit_calibrator(prediction, target, "isotonic")
    calibrated = apply_calibrator(model, np.array([0.0, 10.0, 25.0, 50.0]))

    assert np.all(np.diff(model["y_thresholds"]) >= 0)
    assert calibrated[0] == model["y_thresholds"][0]
    assert calibrated[-1] == model["y_thresholds"][-1]
    assert calibrated[1] <= calibrated[2] <= calibrated[3]


def test_affine_calibration_corrects_compressed_intensity_range() -> None:
    target = np.array([10.0, 20.0, 40.0, 60.0])
    prediction = 0.5 * target + 10.0

    model = fit_calibrator(prediction, target, "affine")
    calibrated = apply_calibrator(model, prediction)
    before = error_metrics(prediction, target, [0.0, 33.0, np.inf])
    after = error_metrics(calibrated, target, [0.0, 33.0, np.inf])

    assert after["mae_ms"] < before["mae_ms"]
    assert after["mae_ms"] < 1e-8
    assert len(after["by_target_intensity"]) == 2


def test_affine_calibration_uses_constant_optimum_for_nonpositive_slope() -> None:
    prediction = np.array([10.0, 20.0, 30.0])
    target = np.array([30.0, 20.0, 10.0])

    model = fit_calibrator(prediction, target, "affine")

    assert model == {
        "method": "affine",
        "slope": 0.0,
        "intercept": 20.0,
        "output_min_ms": 0.0,
        "output_max_ms": 80.0,
    }
    np.testing.assert_allclose(apply_calibrator(model, prediction), 20.0)


def test_affine_calibration_handles_near_constant_predictor() -> None:
    prediction = np.array([10.0, 10.0 + 1e-13, 10.0 + 2e-13])
    target = np.array([10.0, 20.0, 30.0])

    model = fit_calibrator(prediction, target, "affine")

    assert model["slope"] == 0.0
    assert model["intercept"] == 20.0


def test_constant_prediction_has_json_safe_null_correlation() -> None:
    prediction = np.array([20.0, 20.0, 20.0])
    target = np.array([10.0, 20.0, 30.0])

    metrics = error_metrics(prediction, target, [0.0, np.inf])

    assert metrics["correlation"] is None
    json.dumps(metrics, allow_nan=False)


def test_calibrator_clips_affine_and_isotonic_to_physical_bounds() -> None:
    prediction = np.array([0.0, 1.0, 2.0])
    target = np.array([-100.0, 40.0, 200.0])

    for method in ("affine", "isotonic"):
        model = fit_calibrator(
            prediction,
            target,
            method,
            output_min_ms=5.0,
            output_max_ms=55.0,
        )
        calibrated = apply_calibrator(model, prediction)

        assert model["output_min_ms"] == 5.0
        assert model["output_max_ms"] == 55.0
        assert calibrated.min() >= 5.0
        assert calibrated.max() <= 55.0


def test_calibrator_rejects_invalid_physical_bounds() -> None:
    with pytest.raises(ValueError, match="maximum greater than minimum"):
        fit_calibrator(
            np.array([10.0, 20.0]),
            np.array([20.0, 30.0]),
            "affine",
            output_min_ms=40.0,
            output_max_ms=40.0,
        )


def test_interval_metrics_reports_coverage_and_width() -> None:
    metrics = interval_metrics(
        np.array([10.0, 20.0, np.nan]),
        np.array([30.0, 30.0, 50.0]),
        np.array([20.0, 35.0, 40.0]),
    )

    assert metrics == {
        "count": 2,
        "coverage": 0.5,
        "mean_width_ms": 15.0,
    }


def test_main_calibrates_related_columns_and_records_interval_diagnostics(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "calibration"
    frame = pd.DataFrame(
        {
            "storm_id": ["A", "A", "B", "B"],
            "ibtracs_msw_ms": [20.0, 40.0, 30.0, 50.0],
            "output_msw_ms_member_median": [10.0, 20.0, 15.0, 25.0],
            "output_msw_ms_member_p10": [-20.0, 10.0, 5.0, 15.0],
            "output_msw_ms_member_p90": [30.0, 120.0, 25.0, 100.0],
            "output_msw_ms_member_mean": [11.0, 21.0, 16.0, 26.0],
            "output_msw_ms_medoid": [12.0, 22.0, 17.0, 27.0],
            "output_msw_ms_mean_field": [9.0, 19.0, 14.0, 24.0],
            "output_msw_ms_median_field": [10.0, 20.0, 15.0, 25.0],
            "output_robust_peak_ms_member_p10": [1.0, 2.0, 3.0, 4.0],
        }
    )
    frame.to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_ensemble_intensity.py",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--method",
            "affine",
            "--output-min-ms",
            "5",
            "--output-max-ms",
            "55",
        ],
    )

    main()

    calibrated = pd.read_csv(output_dir / "calibrated-predictions.csv")
    related = (
        "member_median",
        "member_p10",
        "member_p90",
        "member_mean",
        "medoid",
        "mean_field",
        "median_field",
    )
    for suffix in related:
        for calibration_suffix in ("_calibrated", "_calibrated_loso"):
            column = f"output_msw_ms_{suffix}{calibration_suffix}"
            assert column in calibrated
            assert calibrated[column].between(5.0, 55.0).all()
    assert "output_robust_peak_ms_member_p10_calibrated" not in calibrated

    payload = json.loads((output_dir / "calibration.json").read_text())
    assert payload["output_bounds_ms"] == {"minimum": 5.0, "maximum": 55.0}
    assert payload["model"]["output_min_ms"] == 5.0
    assert payload["model"]["output_max_ms"] == 55.0

    evaluation = json.loads((output_dir / "calibration-evaluation.json").read_text())
    assert evaluation["raw"]["p10_p90_interval_count"] == 4
    assert evaluation["raw"]["p10_p90_mean_width_ms"] == 66.25
    assert evaluation["calibrated_in_sample"]["p10_p90_mean_width_ms"] <= 50.0
    assert evaluation["calibrated_leave_one_storm_out"]["p10_p90_coverage"] is not None
