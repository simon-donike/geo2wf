from __future__ import annotations

from types import SimpleNamespace
import sys

from geo2wf.models.intensity_correction import summarize_intensity_rows
from geo2wf.tracking.intensity_media import (
    log_wandb_intensity_evaluation,
    select_intensity_plot_storms,
)


def _rows() -> list[dict[str, object]]:
    rows = []
    storm_counts = {"storm-short": 1, "storm-long": 4, "storm-mid": 3, "storm-two": 2}
    for storm_index, (storm_id, count) in enumerate(storm_counts.items()):
        for fix_index in range(count):
            target = 20.0 + storm_index + fix_index
            rows.append(
                {
                    "sample_id": f"{storm_id}-{fix_index}",
                    "storm_id": storm_id,
                    "observation_timestamp": (f"2022-08-{fix_index + 1:02d}T00:00:00Z"),
                    "prediction_ms": target + 1.0,
                    "target_ms": target,
                    "raw_unet_ms": target - 2.0,
                    "correction_ms": 3.0,
                    "prediction_category": 0,
                    "target_category": 0,
                }
            )
    return rows


def test_validation_plot_selects_exactly_three_information_rich_storms() -> None:
    rows = _rows()
    assert select_intensity_plot_storms(rows) == [
        "storm-long",
        "storm-mid",
        "storm-two",
    ]
    assert select_intensity_plot_storms(rows, preferred_storm_ids=["storm-short"]) == [
        "storm-short",
        "storm-long",
        "storm-mid",
    ]


def test_wandb_validation_logs_three_storm_plot_and_tables(monkeypatch) -> None:
    rows = _rows()
    logged: list[tuple[dict[str, object], int]] = []

    class FakeExperiment:
        def log(self, payload, step):
            logged.append((payload, step))

    class FakeImage:
        def __init__(self, figure):
            self.figure = figure

    class FakeTable:
        def __init__(self, *, columns, data):
            self.columns = columns
            self.data = data

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Image=FakeImage, Table=FakeTable),
    )
    experiment = FakeExperiment()
    trainer = SimpleNamespace(
        is_global_zero=True,
        sanity_checking=False,
        loggers=[SimpleNamespace(experiment=experiment)],
    )
    module = SimpleNamespace(_trainer=trainer, global_step=17)

    log_wandb_intensity_evaluation(
        module,
        rows,
        summarize_intensity_rows(rows),
        prefix="val",
        storm_count=3,
    )

    assert len(logged) == 1
    payload, step = logged[0]
    assert step == 17
    assert set(payload) == {
        "val/three_storm_intensity_comparison",
        "val/three_storm_predictions",
        "val/per_storm_metrics",
        "val/category_confusion_matrix",
    }
    prediction_table = payload["val/three_storm_predictions"]
    plotted_storms = {row[1] for row in prediction_table.data}
    assert plotted_storms == {"storm-long", "storm-mid", "storm-two"}
    assert len(payload["val/per_storm_metrics"].data) == 4
    assert len(payload["val/category_confusion_matrix"].data) == 7
