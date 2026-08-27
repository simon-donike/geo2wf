from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from geo2wf.data.joint_intensity import (
    KNOT_TO_MS,
    _center_mask_value,
    _interpolate_ibtracs_wind,
    _ri_diagnostics,
    _sar_intensity_diagnostics,
)
from scripts.analyze_sar_ibtracs_divergence import summarize_divergence


def _write_sar(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        transform=from_origin(0.0, float(values.shape[0]), 1.0, 1.0),
        crs="EPSG:4326",
        nodata=np.nan,
    ) as destination:
        destination.write(values.astype(np.float32), 1)


def test_sar_robust_peak_uses_ceiling_top_fraction_and_finite_mask(
    tmp_path: Path,
) -> None:
    values = np.arange(1000, dtype=np.float32).reshape(20, 50)
    values[0, 0] = np.nan
    path = tmp_path / "sar.tif"
    _write_sar(path, values)
    paired = SimpleNamespace(
        root=tmp_path,
        target_size=(20, 50),
        center_crop_size=None,
    )
    diagnostics = _sar_intensity_diagnostics(
        paired,
        pd.Series(
            {
                "target_path": "sar.tif",
                "ibtracs_center_lat": 10.5,
                "ibtracs_center_lon": 25.5,
            }
        ),
        robust_peak_fraction=0.005,
    )

    finite = values[np.isfinite(values)]
    count = math.ceil(finite.size * 0.005)
    expected = np.sort(finite)[-count:].mean()
    assert diagnostics["valid_pixels"] == 999
    assert diagnostics["max_wind_ms"] == 999.0
    assert diagnostics["robust_peak_ms"] == pytest.approx(float(expected))
    assert diagnostics["has_valid_center"] is True


def test_center_validity_respects_crop_boundaries_and_invalid_cells() -> None:
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    mask[0, 1, 2] = False
    bounds = torch.tensor([2.0, 6.0, 2.0, 6.0])

    assert _center_mask_value(mask, bounds, 5.5, 2.0)
    assert not _center_mask_value(mask, bounds, 4.5, 4.5)
    assert not _center_mask_value(mask, bounds, 5.5, 6.0)
    assert not _center_mask_value(mask, bounds, 2.0, 3.5)
    assert not _center_mask_value(mask, bounds, math.nan, 3.5)


def _fixes(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([row[0] for row in rows], utc=True),
            "wind_kt": [row[1] for row in rows],
        }
    )


def test_ibtracs_interpolation_endpoints_and_ri_threshold_equality() -> None:
    interpolation_fixes = _fixes(
        [("2024-01-01T00:00Z", 40.0), ("2024-01-01T03:00Z", 70.0)]
    )
    lower = _interpolate_ibtracs_wind(
        interpolation_fixes,
        "2024-01-01T00:00Z",
        max_bracket_hours=3.0,
    )
    upper = _interpolate_ibtracs_wind(
        interpolation_fixes,
        "2024-01-01T03:00Z",
        max_bracket_hours=3.0,
    )
    assert lower is not None and lower["target_wind_ms"] == pytest.approx(
        40 * KNOT_TO_MS
    )
    assert upper is not None and upper["target_wind_ms"] == pytest.approx(
        70 * KNOT_TO_MS
    )

    ri_fixes = _fixes([("2024-01-01T00:00Z", 40.0), ("2024-01-02T00:00Z", 70.0)])
    change, is_ri = _ri_diagnostics(
        ri_fixes,
        "2024-01-02T00:00Z",
        current_wind_ms=70.0 * KNOT_TO_MS,
        max_bracket_hours=3.0,
        threshold_kt=30.0,
        window_hours=24.0,
    )
    assert change == pytest.approx(30.0 * KNOT_TO_MS)
    assert is_ri is True


def test_ri_requires_interpolatable_24_hour_history() -> None:
    fixes = _fixes([("2024-01-01T12:00Z", 40.0), ("2024-01-02T00:00Z", 80.0)])
    change, is_ri = _ri_diagnostics(
        fixes,
        "2024-01-02T00:00Z",
        current_wind_ms=80.0 * KNOT_TO_MS,
        max_bracket_hours=3.0,
        threshold_kt=30.0,
        window_hours=24.0,
    )
    assert math.isnan(change)
    assert is_ri is False


def test_divergence_summary_reports_signed_errors_quantiles_and_bootstrap() -> None:
    frame = pd.DataFrame(
        {
            "storm_id": ["a", "a", "b", "b"],
            "ibtracs_target_ms": [10.0, 20.0, 30.0, 40.0],
            "sar_robust_peak_target_ms": [12.0, 18.0, 33.0, 39.0],
        }
    )
    summary = summarize_divergence(
        frame,
        "sar_robust_peak_target_ms",
        repetitions=20,
        seed=7,
    )

    assert summary["samples"] == 4
    assert summary["storms"] == 2
    assert summary["bias_ms"] == pytest.approx(0.5)
    assert summary["mae_ms"] == pytest.approx(2.0)
    assert set(summary["signed_error_quantiles_ms"]) == {
        "p05",
        "p25",
        "p50",
        "p75",
        "p95",
    }
    assert summary["storm_bootstrap"]["repetitions"] == 20
    assert len(summary["storm_bootstrap"]["mae_ms_95ci"]) == 2
