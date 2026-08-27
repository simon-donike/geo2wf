from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.postprocess_diffusion_ablations import (
    _metrics_for_table,
    _smooth_residual,
    _variant_name,
)


def test_variant_name_is_machine_safe_and_explicit() -> None:
    assert _variant_name(0.25, 8.0, 3) == "gain_0p25_cap_8_median"
    assert _variant_name(1.0, float("inf"), 0) == "gain_1_cap_none_raw"


def test_smoothing_preserves_invalid_mask_and_removes_isolated_spike() -> None:
    residual = np.zeros((1, 5, 5), dtype=np.float32)
    residual[0, 2, 2] = 10.0
    valid = np.ones((5, 5), dtype=bool)
    valid[0, 0] = False
    smoothed = _smooth_residual(residual, valid, 3)
    assert smoothed[0, 2, 2] == 0.0
    assert np.isnan(smoothed[0, 0, 0])


def test_metrics_include_skill_and_high_wind_stratum() -> None:
    table = pd.DataFrame(
        {
            "ibtracs_msw_ms": [20.0, 40.0],
            "baseline_msw_ms": [10.0, 30.0],
            "baseline_robust_peak_ms": [10.0, 30.0],
            "output_msw_ms": [20.0, 35.0],
            "output_robust_peak_ms": [18.0, 34.0],
        }
    )
    metrics = _metrics_for_table(table)
    assert metrics["msw_vs_ibtracs_msw"]["mae_ms"] == 2.5
    assert metrics["msw_vs_ibtracs_msw"]["mae_skill_vs_baseline"] == 0.75
    assert metrics["robust_peak_high_wind_ge_33_ms"]["count"] == 1
