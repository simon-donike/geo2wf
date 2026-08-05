from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from geo2wf.data.intensity import KNOT_TO_MS
from geo2wf.data.intensity_forecast import (
    FORECAST_FEATURE_NAMES,
    IntensityForecastDataModule,
    IntensityForecastDataSpec,
    forecast_features,
)
from geo2wf.models.intensity_forecast import IntensityForecastMLP
from geo2wf.tracking.forecast_media import (
    log_wandb_ri_forecasts,
    recursive_ri_rows,
)
from scripts.export_intensity_forecast_cache import (
    _deduplicate_manifest,
    build_historical_rows,
    earliest_ri_onset,
    read_ibtracs_forecast_tracks,
    select_ri_validation_cases,
)


def _ibtracs(path: Path, storm: str = "AL012010") -> pd.DataFrame:
    rows = []
    for index, hour in enumerate(range(0, 43, 3)):
        timestamp = pd.Timestamp("2010-08-01T00:00:00Z") + pd.Timedelta(hours=hour)
        rows.append(
            {
                "USA_ATCF_ID": storm,
                "SEASON": 2010,
                "ISO_TIME": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "USA_WIND": 30 + 5 * index,
                "USA_SSHS": 0,
            }
        )
    frame = pd.DataFrame(rows)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(frame.columns) + "\n")
        handle.write("Year, , ,kts,1\n")
        frame.to_csv(handle, index=False, header=False)
    return frame


def test_ibtracs_export_uses_exact_scalar_history_and_skips_units(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ibtracs.csv"
    _ibtracs(path)
    tracks = read_ibtracs_forecast_tracks(path)
    assert tracks.iloc[0]["storm_id"] == "AL012010"
    rows = build_historical_rows(
        tracks, start_year=2000, train_end_year=2018, val_end_year=2022
    )["pretrain_train"]
    selected = next(
        row for row in rows if row["init_timestamp"].startswith("2010-08-01T12:00")
    )
    assert selected["anchor_wind_ms"] == 50 * KNOT_TO_MS
    assert selected["wind_minus_6h_ms"] == 40 * KNOT_TO_MS
    assert selected["wind_minus_12h_ms"] == 30 * KNOT_TO_MS
    assert selected["target_wind_ms"] == 60 * KNOT_TO_MS
    assert np.isclose(selected["target_delta_ms"], 10 * KNOT_TO_MS)


def test_matched_deduplication_prefers_time_then_coverage_then_id() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "c",
                "storm_id": "one",
                "target_timestamp": "2020-01-01T00:00:00Z",
                "target_gap_minutes": 10,
                "valid_fraction": 0.9,
            },
            {
                "sample_id": "b",
                "storm_id": "one",
                "target_timestamp": "2020-01-01T00:00:00Z",
                "target_gap_minutes": -5,
                "valid_fraction": 0.8,
            },
            {
                "sample_id": "a",
                "storm_id": "one",
                "target_timestamp": "2020-01-01T00:00:00Z",
                "target_gap_minutes": 5,
                "valid_fraction": 0.8,
            },
        ]
    )
    selected = _deduplicate_manifest(frame)
    assert selected.iloc[0]["sample_id"] == "a"


def _spec() -> IntensityForecastDataSpec:
    return IntensityForecastDataSpec(
        feature_names=FORECAST_FEATURE_NAMES,
        feature_mean=(0.0,) * 5,
        feature_std=(1.0,) * 5,
    )


def test_zero_initialized_forecast_is_nonnegative_persistence() -> None:
    model = IntensityForecastMLP(hidden_features=(8, 4))
    model.validate_data_spec(_spec())
    anchor = torch.tensor([20.0, -2.0])
    features = torch.from_numpy(
        np.stack([forecast_features(20, 18, 17), forecast_features(-2, 1, 2)])
    )
    output = model.predict_forecast(features, anchor)
    assert torch.allclose(output.output_wind_ms, torch.tensor([20.0, 0.0]))


def test_recursive_second_step_consumes_first_prediction() -> None:
    model = IntensityForecastMLP(hidden_features=(8, 4), dropout=0.0)
    model.validate_data_spec(_spec())
    with torch.no_grad():
        model.delta_head.weight.zero_()
        model.delta_head.bias.fill_(2.0)
    plus_6, plus_12 = model.predict_two_steps(
        torch.tensor([20.0]), torch.tensor([18.0]), torch.tensor([17.0])
    )
    assert plus_6.item() == 22.0
    assert plus_12.item() == 24.0


def _ri_tracks() -> pd.DataFrame:
    rows = []
    for storm in ("WP282025", "WP112024", "AL092024"):
        start = pd.Timestamp("2025-01-01T00:00:00Z")
        for hour in range(0, 37, 6):
            rows.append(
                {
                    "storm_id": storm,
                    "timestamp": start + pd.Timedelta(hours=hour),
                    "wind_kt": 40 + 10 * (hour / 6),
                    "category": 0,
                }
            )
    return pd.DataFrame(rows)


def _ri_cases() -> list[dict[str, object]]:
    cases = []
    for storm in ("WP282025", "WP112024", "AL092024"):
        cases.append(
            {
                "storm_id": storm,
                "sample_id": storm + "-sample",
                "init_timestamp": "2025-01-01T00:00:00+00:00",
                "ri_onset_timestamp": "2025-01-01T00:00:00+00:00",
                "anchor_wind_ms": 20.0,
                "current_ibtracs_wind_ms": 19.0,
                "wind_minus_6h_ms": 18.0,
                "wind_minus_12h_ms": 16.0,
                "target_plus_6h_wind_ms": 24.0,
                "target_plus_12h_wind_ms": 29.0,
            }
        )
    return cases


def test_ri_onset_and_case_selection_are_exact_and_bounded() -> None:
    tracks = _ri_tracks()
    assert earliest_ri_onset(tracks, "WP282025") == pd.Timestamp("2025-01-01T00:00:00Z")
    rows = []
    for case in _ri_cases():
        rows.append(
            {
                **case,
                "split": "val",
                "target_wind_ms": case["target_plus_6h_wind_ms"],
                "target_plus_12h_wind_ms": case["target_plus_12h_wind_ms"],
            }
        )
    selected = select_ri_validation_cases(rows, tracks)
    assert [case["storm_id"] for case in selected] == [
        "WP282025",
        "WP112024",
        "AL092024",
    ]


def test_wandb_ri_media_logs_three_storms_and_two_horizons(monkeypatch) -> None:
    model = IntensityForecastMLP(hidden_features=(8, 4), dropout=0.0)
    model.validate_data_spec(_spec())
    logged = []

    class Experiment:
        def log(self, payload, step):
            logged.append((payload, step))

    class Image:
        def __init__(self, figure):
            self.figure = figure

    class Table:
        def __init__(self, *, columns, data):
            self.columns = columns
            self.data = data

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=Image, Table=Table))
    model._trainer = SimpleNamespace(
        is_global_zero=True,
        sanity_checking=False,
        loggers=[SimpleNamespace(experiment=Experiment())],
    )
    before = set(plt.get_fignums())
    log_wandb_ri_forecasts(model, _ri_cases())
    assert len(logged) == 1
    payload, _ = logged[0]
    assert set(payload) == {
        "val/ri_two_step_forecast",
        "val/ri_two_step_forecasts",
    }
    assert len(payload["val/ri_two_step_forecasts"].data) == 6
    assert set(plt.get_fignums()) == before
    rows = recursive_ri_rows(model, _ri_cases())
    assert {row["horizon_hours"] for row in rows} == {6, 12}


def _write_cache_split(root: Path, split: str, storm: str) -> None:
    rows = []
    for index in range(2):
        anchor = 20.0 + index
        target = anchor + 3.0
        rows.append(
            {
                "sample_id": f"{split}-{index}",
                "storm_id": storm,
                "split": split,
                "init_timestamp": f"2020-01-0{index + 1}T00:00:00Z",
                "anchor_wind_ms": anchor,
                "wind_minus_6h_ms": anchor - 1,
                "wind_minus_12h_ms": anchor - 2,
                "target_wind_ms": target,
                "target_delta_ms": 3.0,
                "source_kind": "synthetic",
            }
        )
    path = root / split
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path / "manifest.csv", index=False)


def test_forecast_lightning_fit_and_checkpoint_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    (root / "cache-metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "feature_scaler": {
                    "names": list(FORECAST_FEATURE_NAMES),
                    "mean": [0.0] * 5,
                    "std": [1.0] * 5,
                },
                "ri_validation_cases": [],
            }
        ),
        encoding="utf-8",
    )
    _write_cache_split(root, "train", "AL012020")
    _write_cache_split(root, "val", "EP022021")
    _write_cache_split(root, "test", "WP032022")
    datamodule = IntensityForecastDataModule(root, batch_size=2, ri_storm_ids=[])
    model = IntensityForecastMLP(hidden_features=(8, 4), log_wandb_ri_media=False)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)
    checkpoint = tmp_path / "forecast.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = IntensityForecastMLP.load_from_checkpoint(checkpoint, map_location="cpu")
    restored.validate_data_spec(datamodule.data_spec)
    assert restored.feature_count == 5
