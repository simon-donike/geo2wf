from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest

from scripts.combine_intensity_target_validation import combine


MODEL_KEYS = ("unet_raw_max", "unet_correction", "joint_unet_mlp")


def _summary(samples: int = 2) -> dict:
    return {
        "samples": samples,
        "storms": samples,
        "regression": {"mae_ms": 1.0, "rmse_ms": 1.2, "bias_ms": -0.2},
        "raw_unet_baseline": {
            "mae_ms": 1.5,
            "rmse_ms": 1.7,
            "bias_ms": -0.4,
        },
        "correction": {"mean_ms": 0.2, "mean_abs_ms": 0.3},
        "storm_macro_mae_ms": 1.0,
        "category": {
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "within_one_accuracy": 1.0,
        },
        "per_storm": {},
    }


def _result(path: Path, era5: str, target: str, unet_hash: str) -> Path:
    reference = {
        "overall": {key: _summary() for key in MODEL_KEYS},
        "overall_storm_bootstrap": {
            "models": {key: {"mae_ms_95ci": [0.5, 1.5]} for key in MODEL_KEYS}
        },
        "rapid_intensification": {key: _summary(1) for key in MODEL_KEYS},
        "rapid_intensification_storm_bootstrap": {
            "models": {key: {"mae_ms_95ci": [0.6, 1.6]} for key in MODEL_KEYS}
        },
    }
    row = {
        "sample_id": "sample",
        "storm_id": "storm",
        "prediction_ms": 20.0,
        "ibtracs_target_ms": 19.0,
        "sar_robust_peak_target_ms": 18.0,
        "raw_unet_max_ms": 17.0,
        "raw_unet_robust_peak_ms": 16.0,
        "is_rapid_intensification": True,
    }
    payload = {
        "schema_version": 2,
        "split": "val",
        "conditioning": {"label": era5, "use_era5": era5 == "with_era5"},
        "cohort": {"sha256": "same-cohort", "samples": 2, "storms": 2},
        "target_fingerprint": {"source": target, "sha256": target},
        "data_config": {"path": "config.yaml", "sha256": "config"},
        "checkpoints": {
            "unet": {"path": "unet.ckpt", "sha256": unet_hash},
            "correction": {
                "path": "correction.ckpt",
                "sha256": f"correction-{era5}-{target}",
            },
            "joint": {
                "path": "joint.ckpt",
                "sha256": f"joint-{era5}-{target}",
            },
        },
        "models": {key: {"label": key} for key in MODEL_KEYS},
        "reference_evaluation": {
            "ibtracs": reference,
            "sar_robust_peak": reference,
        },
        "prediction_rows": {key: [row] for key in MODEL_KEYS},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _args(tmp_path: Path) -> Namespace:
    results = []
    for era5 in ("with_era5", "without_era5"):
        for target in ("ibtracs", "sar_robust_peak"):
            results.append(
                _result(
                    tmp_path / f"{era5}-{target}.json",
                    era5,
                    target,
                    f"unet-{era5}",
                )
            )
    divergence = tmp_path / "divergence.json"
    divergence.write_text(
        json.dumps(
            {
                "table": [
                    {
                        "subset": "all",
                        "diagnostic": "sar_robust_peak",
                        "samples": 1,
                        "storms": 1,
                        "bias_ms": -1.0,
                        "mae_ms": 1.0,
                        "rmse_ms": 1.0,
                    }
                ],
                "rows": [
                    {
                        "ibtracs_target_ms": 20.0,
                        "sar_robust_peak_target_ms": 19.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return Namespace(
        result=results,
        divergence=divergence,
        output=tmp_path / "combined.json",
        wandb_project="geo2wf",
        wandb_name=None,
        wandb_group="test",
        documentation=None,
        disable_wandb=True,
    )


def test_combiner_writes_dual_reference_overall_ri_and_prediction_artifacts(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    payload = combine(args)

    assert len(payload["metrics"]) == 40
    assert {row["subset"] for row in payload["metrics"]} == {
        "overall",
        "rapid_intensification",
    }
    assert {row["evaluation_reference"] for row in payload["metrics"]} == {
        "ibtracs",
        "sar_robust_peak",
    }
    assert args.output.is_file()
    assert args.output.with_suffix(".csv").is_file()
    assert args.output.with_suffix(".md").is_file()
    assert args.output.with_name("combined-predictions.csv").is_file()
    assert args.output.with_name("combined-divergence.png").is_file()


def test_combiner_rejects_a_mismatched_target_cohort(tmp_path: Path) -> None:
    args = _args(tmp_path)
    payload = json.loads(args.result[0].read_text(encoding="utf-8"))
    payload["cohort"]["sha256"] = "different"
    args.result[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cohort fingerprint"):
        combine(args)


def test_combiner_updates_the_marked_experiment_results(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.documentation = tmp_path / "experiment.md"
    args.documentation.write_text(
        "# Experiment\n\n"
        "## Results\n\n"
        "<!-- matched-validation-results:start -->\n"
        "Pending.\n"
        "<!-- matched-validation-results:end -->\n\n"
        "## Methods\n",
        encoding="utf-8",
    )

    combine(args)

    rendered = args.documentation.read_text(encoding="utf-8")
    assert "Pending." not in rendered
    assert "completed seed-42 validation matrix" in rendered
    assert "| ERA5 | Trained target | Model |" in rendered
    assert "MAE, m/s (kt); 95% CI" in rendered
    assert "kt); 95% CI" in rendered
    assert "## Methods" in rendered
