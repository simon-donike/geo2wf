from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from scripts.evaluate_intensity_models import (
    _assert_common_cohort,
    _cluster_bootstrap,
    _cohort_fingerprint as evaluation_fingerprint,
    _markdown_table,
    _raw_rows,
    _table_rows,
)
from scripts.export_joint_intensity_cache import (
    _cohort_fingerprint as export_fingerprint,
)
from scripts.run_intensity_model_comparison import _training_result


def _rows() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "sample-b",
            "storm_id": "EP022024",
            "split": "val",
            "target_timestamp": "2024-08-02T00:00:00+00:00",
            "observation_timestamp": "2024-08-02T00:00:00+00:00",
            "target_wind_ms": 36.0,
            "target_ms": 36.0,
            "target_category": 1,
            "prediction_ms": 34.0,
            "prediction_category": 1,
            "raw_unet_ms": 31.0,
            "correction_ms": 3.0,
        },
        {
            "sample_id": "sample-a",
            "storm_id": "AL012024",
            "split": "val",
            "target_timestamp": "2024-08-01T00:00:00+00:00",
            "observation_timestamp": "2024-08-01T00:00:00+00:00",
            "target_wind_ms": 20.0,
            "target_ms": 20.0,
            "target_category": 0,
            "prediction_ms": 21.0,
            "prediction_category": 0,
            "raw_unet_ms": 18.0,
            "correction_ms": 3.0,
        },
    ]


def test_export_and_evaluation_use_the_same_order_independent_fingerprint() -> None:
    rows = _rows()
    exported = export_fingerprint(rows)
    evaluated = evaluation_fingerprint(pd.DataFrame(reversed(rows)))

    assert exported == evaluated
    assert exported["samples"] == 2
    assert exported["storms"] == 2


def test_paired_storm_bootstrap_is_deterministic_and_raw_delta_is_zero() -> None:
    correction = _rows()
    raw = _raw_rows(correction)
    joint = [{**row, "prediction_ms": row["target_ms"]} for row in correction]
    models = {
        "unet_raw_max": raw,
        "unet_correction": correction,
        "joint_unet_mlp": joint,
    }

    first = _cluster_bootstrap(models, repetitions=50, seed=7)
    second = _cluster_bootstrap(models, repetitions=50, seed=7)

    assert first == second
    assert first["models"]["unet_raw_max"]["mae_delta_vs_unet_raw_max_ms_95ci"] == [
        0.0,
        0.0,
    ]


def test_common_cohort_audit_rejects_different_sample_ids() -> None:
    class Dataset:
        def __init__(self, sample_ids: list[str]) -> None:
            self.samples = pd.DataFrame({"sample_id": sample_ids})

        def __len__(self) -> int:
            return len(self.samples)

    joint = Dataset(["a", "b"])
    cache = Dataset(["a", "c"])
    with pytest.raises(ValueError, match="cohorts differ"):
        _assert_common_cohort(joint, cache, {"a": {}, "c": {}})


def test_table_marks_scalar_only_correction_field_metrics_as_missing() -> None:
    summary = {
        "samples": 2,
        "storms": 2,
        "regression": {"mae_ms": 1.0, "rmse_ms": 1.5, "bias_ms": 0.25},
        "storm_macro_mae_ms": 1.0,
        "category": {"accuracy": 0.5, "macro_f1": 0.4, "within_one_accuracy": 1.0},
    }
    summaries = {
        name: summary for name in ("unet_raw_max", "unet_correction", "joint_unet_mlp")
    }
    fields = {
        "unet_raw_max": {"mae_ms": 2.0, "rmse_ms": 3.0, "bias_ms": -1.0},
        "joint_unet_mlp": {"mae_ms": 1.5, "rmse_ms": 2.5, "bias_ms": 0.0},
    }
    bootstrap = {"models": {name: {"mae_ms_95ci": [0.5, 1.5]} for name in summaries}}

    rows = _table_rows(summaries, fields, bootstrap)
    correction = next(row for row in rows if row["model_key"] == "unet_correction")
    markdown = _markdown_table(rows, split="val")

    assert correction["field_mae_ms"] is None
    assert "U-Net + correction" in markdown
    assert "—" in markdown


def test_training_result_uses_recorded_best_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "20260820-120000_modular"
    checkpoint = run / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (run / "resolved-config.yaml").write_text("seed: 42\n", encoding="utf-8")
    (run / "result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "best_model_path": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )

    selected, config = _training_result(tmp_path)

    assert selected == checkpoint.resolve()
    assert config == (run / "resolved-config.yaml").resolve()
