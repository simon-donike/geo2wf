from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scripts.publish_current_experiment_results import (
    _regime_series,
    add_era5_maximum_wind,
    storm_metric_rows,
    storm_prediction_rows,
)


def test_era5_maximum_uses_nearest_time_and_storm_centered_crop(
    tmp_path: Path,
) -> None:
    era5_path = tmp_path / "era5.nc"
    latitudes = np.array([6.0, 9.0, 10.0, 11.0, 14.0])
    longitudes = np.array([16.0, 19.0, 20.0, 21.0, 24.0])
    u = np.full((2, 5, 5), 100.0, dtype=np.float32)
    v = np.zeros_like(u)
    u[0, 1:4, 1:4] = 3.0
    v[0, 1:4, 1:4] = 4.0
    u[1, 1:4, 1:4] = 5.0
    v[1, 1:4, 1:4] = 12.0
    dataset = xr.Dataset(
        {
            "u_wind_10m": (("time", "y", "x"), u),
            "v_wind_10m": (("time", "y", "x"), v),
            "latitude": (
                ("time", "y"),
                np.broadcast_to(latitudes, (2, len(latitudes))),
            ),
            "longitude": (
                ("time", "x"),
                np.broadcast_to(longitudes, (2, len(longitudes))),
            ),
        },
        coords={"time": pd.to_datetime(["2025-08-01T00:00", "2025-08-01T03:00"])},
    )
    dataset.to_netcdf(era5_path, group="rectilinear", engine="h5netcdf")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{"source_type": "era5", "storm_id": "AL082025", "path": era5_path.name}]
    ).to_csv(manifest_path, index=False)
    frame = pd.DataFrame(
        {
            "observation_id": ["scene-1"],
            "storm_id": ["AL082025"],
            "observation_timestamp": [pd.Timestamp("2025-08-01T01:40:00Z")],
            "input_path": ["geo_[+10.000deg_+20.000deg].nc"],
        }
    )

    result = add_era5_maximum_wind(frame, manifest_path, tmp_path)

    assert result.loc[0, "era5_max_wind_ms"] == 13.0
    assert result.loc[0, "era5_valid_timestamp"] == "2025-08-01T03:00:00+00:00"


def _storm_frame(regime: str) -> pd.DataFrame:
    era5_token = "era5" if regime == "with_era5" else "no_era5"
    frame = pd.DataFrame(
        {
            "observation_id": ["scene-1", "scene-2"],
            "storm_id": ["AL082025", "AL082025"],
            "observation_timestamp": pd.to_datetime(
                ["2025-08-01T00:00:00Z", "2025-08-01T01:00:00Z"]
            ),
            "target_ms": [20.0, 30.0],
            "inference_valid": [True, True],
            "is_rapid_intensification": [False, True],
            "unet_raw_max_ms": [21.0, 32.0],
            f"latent_sar_{era5_token}_max_wind_max_wind_ms": [19.0, 27.0],
            "era5_max_wind_ms": [18.0, 26.0],
        }
    )
    return frame


def test_storm_exports_include_each_regime_ri_metrics_and_era5_reference() -> None:
    frames = {regime: _storm_frame(regime) for regime in ("with_era5", "without_era5")}
    series = {regime: _regime_series(frame, regime) for regime, frame in frames.items()}

    predictions = storm_prediction_rows(frames, series)
    metrics = storm_metric_rows(predictions)

    assert len(predictions) == 10
    assert set(predictions["conditioning"]) == {
        "with_era5",
        "without_era5",
        "reference",
    }
    selected = metrics.set_index(["model_key", "conditioning", "subset"])
    raw = selected.loc[("unet_raw", "with_era5", "all_three_storms")]
    assert raw["samples"] == 2
    assert raw["mae_ms"] == 1.5
    assert raw["rmse_ms"] == np.sqrt(2.5)
    ri = selected.loc[("unet_raw", "with_era5", "ri_three_storms")]
    assert ri["samples"] == 1
    assert ri["mae_ms"] == 2.0
    era5 = selected.loc[("era5_max_wind", "reference", "all_three_storms")]
    assert era5["mae_ms"] == 3.0
