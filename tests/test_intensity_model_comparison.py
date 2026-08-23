from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from geo2wf.data.datasets.paired_geotiff import PairedImageDataset
from scripts.combine_intensity_comparison_reports import combined_markdown_report
from scripts.evaluate_intensity_models import (
    _assert_common_cohort,
    _cluster_bootstrap,
    _cohort_fingerprint as evaluation_fingerprint,
    _reference_evaluation,
    _target_fingerprint as evaluation_target_fingerprint,
    _markdown_table,
    _raw_rows,
    _table_rows,
)
from scripts.export_joint_intensity_cache import (
    _cohort_fingerprint as export_fingerprint,
    _target_fingerprint as export_target_fingerprint,
)
from scripts.run_intensity_model_comparison import _source_storm_counts
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


def test_cohort_fingerprint_is_label_independent_but_target_hash_is_not() -> None:
    ibtracs = []
    sar = []
    for row in _rows():
        shared = {
            **row,
            "ibtracs_target_ms": row["target_wind_ms"],
            "sar_robust_peak_target_ms": float(row["target_wind_ms"]) + 2.0,
        }
        ibtracs.append(
            {
                **shared,
                "intensity_target_source": "ibtracs",
                "target_wind_ms": shared["ibtracs_target_ms"],
            }
        )
        sar.append(
            {
                **shared,
                "intensity_target_source": "sar_robust_peak",
                "target_wind_ms": shared["sar_robust_peak_target_ms"],
            }
        )

    assert export_fingerprint(ibtracs) == export_fingerprint(sar)
    assert (
        export_target_fingerprint(ibtracs)["sha256"]
        != export_target_fingerprint(sar)["sha256"]
    )
    assert export_target_fingerprint(ibtracs) == evaluation_target_fingerprint(
        pd.DataFrame(reversed(ibtracs))
    )


def test_dual_reference_evaluation_splits_ri_and_handles_empty_ri() -> None:
    rows = []
    for index, is_ri in enumerate((False, True)):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "storm_id": f"storm-{index}",
                "prediction_ms": 22.0 + index,
                "target_ms": 20.0 + index,
                "target_category": 0,
                "prediction_category": 0,
                "raw_unet_ms": 19.0 + index,
                "raw_unet_max_ms": 19.0 + index,
                "raw_unet_robust_peak_ms": 18.0 + index,
                "ibtracs_target_ms": 20.0 + index,
                "sar_robust_peak_target_ms": 21.0 + index,
                "correction_ms": 3.0,
                "is_rapid_intensification": is_ri,
            }
        )
    models = {
        name: rows for name in ("unet_raw_max", "unet_correction", "joint_unet_mlp")
    }
    summary = _reference_evaluation(models, bootstrap_repetitions=0, bootstrap_seed=42)

    assert summary["ibtracs"]["rapid_intensification"]["unet_raw_max"]["samples"] == 1
    assert summary["sar_robust_peak"]["overall"]["joint_unet_mlp"]["samples"] == 2

    no_ri = [{**row, "is_rapid_intensification": False} for row in rows]
    empty_summary = _reference_evaluation(
        {name: no_ri for name in models},
        bootstrap_repetitions=0,
        bootstrap_seed=42,
    )
    assert empty_summary["ibtracs"]["rapid_intensification"] is None


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
    assert "## Metric definitions" in markdown
    assert "Field bias" in markdown


def test_combined_report_requires_and_documents_the_same_cohort() -> None:
    rows = [
        {
            "model": "Example",
            "model_key": "example",
            "samples": 2,
            "storms": 2,
            "intensity_mae_ms": 1.0,
            "intensity_mae_95ci_low_ms": 0.5,
            "intensity_mae_95ci_high_ms": 1.5,
            "intensity_mae_delta_vs_unet_raw_max_ms": -0.25,
            "intensity_mae_delta_95ci_low_ms": -0.5,
            "intensity_mae_delta_95ci_high_ms": -0.1,
            "intensity_rmse_ms": 1.2,
            "intensity_bias_ms": -0.2,
            "storm_macro_mae_ms": 0.9,
            "category_accuracy": 0.5,
            "category_macro_f1": 0.4,
            "within_one_category_accuracy": 1.0,
            "field_mae_ms": 2.0,
            "field_rmse_ms": 2.5,
            "field_bias_ms": -0.1,
        }
    ]
    payload = {
        "split": "val",
        "cohort": {"sha256": "same", "samples": 2, "storms": 2},
        "paired_storm_bootstrap": {"repetitions": 2000, "seed": 42},
        "table": rows,
    }

    report = combined_markdown_report(payload, payload)

    assert "## With ERA5" in report
    assert "## Without ERA5" in report
    assert "2,000 paired cluster-bootstrap repetitions" in report
    assert report.count("## Metric definitions") == 1

    different = {**payload, "cohort": {**payload["cohort"], "sha256": "other"}}
    with pytest.raises(ValueError, match="exact same cohort"):
        combined_markdown_report(payload, different)


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


def test_source_storm_counts_reports_absent_and_held_out_storms(
    tmp_path: Path,
) -> None:
    for split, storms in {
        "train": ["AL012020"],
        "val": ["EP182023", "EP182023"],
        "test": ["AL092019"],
    }.items():
        directory = tmp_path / split
        directory.mkdir()
        pd.DataFrame({"storm_id": storms}).to_csv(
            directory / "manifest.csv", index=False
        )

    counts = _source_storm_counts(tmp_path, ["AL092019", "EP132019", "EP182023"])

    assert counts["AL092019"] == {"train": 0, "val": 0, "test": 1}
    assert counts["EP132019"] == {"train": 0, "val": 0, "test": 0}
    assert counts["EP182023"] == {"train": 0, "val": 2, "test": 0}


def test_no_era5_regime_filters_to_the_era5_available_cohort(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    pd.DataFrame(
        {
            "sample_id": ["with-era5", "without-era5"],
            "context_path": ["train/context.tif", ""],
            "context_source_type": ["era5", ""],
        }
    ).to_csv(split_dir / "manifest.csv", index=False)
    (tmp_path / "stats.json").write_text("{}", encoding="utf-8")

    dataset = PairedImageDataset(
        tmp_path,
        "train",
        require_era5=True,
        use_era5=False,
    )

    assert dataset.samples["sample_id"].tolist() == ["with-era5"]
    assert dataset.filtered_missing_era5_count == 1
