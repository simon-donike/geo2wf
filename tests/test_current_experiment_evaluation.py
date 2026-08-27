from __future__ import annotations

import pandas as pd
import pytest

from scripts.evaluate_current_experiments import _standard_rows
from scripts.run_intensity_comparison_storm_inference import _extra_metrics


def test_standard_rows_include_required_all_and_ri_metrics() -> None:
    config = {
        "model": {
            "_target_": (
                "geo2wf.models.bottleneck_unet_mlp." "BottleneckUNetMLPRegressor"
            )
        }
    }
    metrics = {
        "val/image_mae_ms": 2.0,
        "val/image_psnr_db": 30.0,
        "val/image_ssim": 0.9,
        "val/intensity_mae_ms": 3.0,
        "val/intensity_rmse_ms": 4.0,
        "val/structure_rmw_mae_km": 5.0,
        "val/structure_rmw_rmse_km": 6.0,
        "val_ri/field_mae_ms": 7.0,
        "val_ri/field_psnr_db": 25.0,
        "val_ri/field_ssim": 0.8,
        "val_ri/ibtracs_mae_ms": 8.0,
        "val_ri/ibtracs_rmse_ms": 9.0,
        "val_ri/ibtracs_rmw_mae_km": 10.0,
        "val_ri/ibtracs_rmw_rmse_km": 11.0,
    }

    rows = _standard_rows("joint", config, metrics)
    indexed = {
        (row["subset"], row["output"], row["target"], row["metric"]): row
        for row in rows
    }

    assert (
        indexed[("all_validation", "image_reconstruction", "wind_field", "l1")]["value"]
        == 2.0
    )
    assert (
        indexed[("all_validation", "scalar_head", "maximum_wind", "rmse")]["value"]
        == 4.0
    )
    assert (
        indexed[("ri_validation", "scalar_head", "maximum_wind", "mae")]["value"] == 8.0
    )
    assert (
        indexed[("ri_validation", "image_derived_radius", "rmw", "rmse")]["value"]
        == 11.0
    )


def test_encoder_rows_mark_image_metrics_not_applicable() -> None:
    config = {
        "model": {
            "_target_": (
                "geo2wf.models.bottleneck_unet_mlp." "BottleneckEncoderMLPRegressor"
            )
        }
    }
    rows = _standard_rows(
        "encoder",
        config,
        {"val/intensity_mae_ms": 2.0, "val/intensity_rmse_ms": 3.0},
    )
    image = next(
        row
        for row in rows
        if row["subset"] == "all_validation"
        and row["output"] == "image_reconstruction"
        and row["metric"] == "l1"
    )
    assert image["value"] is None
    assert image["available"] is False
    assert "no jointly evaluated image decoder" in image["availability_reason"]


def test_three_storm_extra_metrics_include_ri_and_radii() -> None:
    frame = pd.DataFrame(
        {
            "is_rapid_intensification": [False, True],
            "target_ms": [20.0, 30.0],
            "target_eye_size_km": [float("nan"), float("nan")],
            "target_rmw_km": [40.0, 50.0],
            "target_r34_km": [100.0, 120.0],
            "target_r50_km": [70.0, 80.0],
            "target_r64_km": [30.0, 40.0],
            "model_max_wind_ms": [22.0, 34.0],
            "model_rmw_km": [45.0, 55.0],
            "model_image_rmw_km": [42.0, 54.0],
        }
    )

    metrics = _extra_metrics(frame, ["model"])["model"]

    assert metrics["all_three_storms"]["maximum_wind"]["mae_ms"] == 3.0
    assert metrics["rapid_intensification"]["maximum_wind"]["rmse_ms"] == 4.0
    assert metrics["all_three_storms"]["radii"]["scalar_head"]["rmw"][
        "rmse_km"
    ] == pytest.approx(5.0)
    assert (
        metrics["rapid_intensification"]["radii"]["image_derived"]["rmw"]["mae_km"]
        == 4.0
    )
