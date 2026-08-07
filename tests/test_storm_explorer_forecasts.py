import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_storm_explorer_data as explorer  # noqa: E402


METRICS = ["max_wind_m_s", "rmw_km"] + [
    f"{radius}_{quadrant}_km"
    for radius in ("r34", "r50", "r64")
    for quadrant in ("ne", "se", "sw", "nw")
]


def forecast_row(storm_id="AL082025", *, with_sar=False):
    row = {
        "storm_id": storm_id,
        "reference_timestamp": "2025-09-27 04:54:49",
        "target_timestamp": "2025-09-27T18:54:49+02:00",
        "target_provenance": "GEO",
    }
    for prefix in ("predicted", "ibtracs", "sar_derived"):
        for index, metric in enumerate(METRICS):
            row[f"{prefix}_{metric}"] = float(index + 1)
            row[f"{prefix}_{metric}_valid"] = prefix != "sar_derived" or with_sar
    row["predicted_max_wind_m_s"] = 54.9704
    row["predicted_rmw_km"] = 9.999
    row["predicted_r34_ne_km"] = 0.0
    row["ibtracs_max_wind_m_s"] = 50.1236
    return row


def write_forecast(root, storm_id="AL082025", *, row=None, summary=None):
    storm_dir = root / storm_id
    storm_dir.mkdir(parents=True)
    pd.DataFrame([row or forecast_row(storm_id)]).to_csv(
        storm_dir / "samples.csv", index=False
    )
    payload = summary or {
        "storm_id": storm_id,
        "split": "val",
        "evaluated_samples": 1,
        "window_hours": 12.0,
        "window_length": 12,
        "forecast_lead_hours": 12.0,
    }
    (storm_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    return storm_dir


def test_missing_forecast_directory_is_optional(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(explorer, "FORECAST_ROOT", tmp_path)

    assert explorer.build_forecast_export("EP182023") == (None, None)


def test_forecast_export_normalizes_and_compacts_browser_data(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write_forecast(source)
    monkeypatch.setattr(explorer, "FORECAST_ROOT", source)
    monkeypatch.setattr(explorer, "FORECAST_OUTPUT_DIR", output)

    metadata = explorer.export_forecast("AL082025")
    payload = json.loads((output / "AL082025.json").read_text(encoding="utf-8"))
    point = payload["points"][0]

    assert metadata == {
        "id": "convlstm",
        "label": "ConvLSTM",
        "metrics": ["max", "rmw"],
        "file": "forecasts/AL082025.json",
        "lead_hours": 12.0,
        "window_hours": 12.0,
        "window_length": 12,
        "split": "val",
        "count": 1,
    }
    assert payload["model"] == {
        "id": "convlstm",
        "label": "ConvLSTM",
        "metrics": ["max", "rmw"],
    }
    assert payload["lead_hours"] == 12.0
    assert point["issue_time"] == "2025-09-27T04:54:49Z"
    assert point["valid_time"] == "2025-09-27T16:54:49Z"
    assert point["target_source"] == "geo"
    assert point["predicted"]["max"] == 54.97
    assert point["predicted"]["rmw"] is None
    assert point["predicted"]["r34"]["ne"] == 0.0
    assert point["ibtracs"]["max"] == 50.124
    assert point["sar"] is None


def test_forecast_export_keeps_valid_sparse_sar_metrics(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    row = forecast_row(with_sar=True)
    row["sar_derived_r64_nw_km_valid"] = False
    row["sar_derived_r64_nw_km"] = float("nan")
    write_forecast(source, row=row)
    monkeypatch.setattr(explorer, "FORECAST_ROOT", source)

    _, payload = explorer.build_forecast_export("AL082025")

    assert payload["points"][0]["sar"]["max"] == 1.0
    assert payload["points"][0]["sar"]["r64"]["nw"] is None


def test_mlp_forecast_export_and_multi_model_manifest(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write_forecast(source)
    write_forecast(
        source / "mlp",
        summary={
            "storm_id": "AL082025",
            "split": "retrospective",
            "evaluated_samples": 1,
            "window_hours": 12,
            "window_length": 3,
            "forecast_lead_hours": 12,
            "model_style": "mlp",
        },
    )
    monkeypatch.setattr(explorer, "FORECAST_ROOT", source)
    monkeypatch.setattr(explorer, "FORECAST_OUTPUT_DIR", output)

    metadata = explorer.export_forecasts("AL082025")

    assert metadata["default_model"] == "mlp"
    assert metadata["file"] == "forecasts/AL082025-mlp.json"
    assert [model["label"] for model in metadata["models"]] == [
        "ConvLSTM",
        "MLP",
    ]
    assert metadata["models"][1]["metrics"] == ["max"]
    mlp_payload = json.loads((output / "AL082025-mlp.json").read_text(encoding="utf-8"))
    assert mlp_payload["model"]["id"] == "mlp"


def test_present_forecast_directory_requires_complete_summary(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    write_forecast(
        source,
        summary={
            "storm_id": "AL082025",
            "split": "val",
            "evaluated_samples": 1,
        },
    )
    monkeypatch.setattr(explorer, "FORECAST_ROOT", source)

    with pytest.raises(ValueError, match="missing"):
        explorer.build_forecast_export("AL082025")


def test_present_forecast_directory_requires_complete_samples(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    storm_dir = source / "AL082025"
    storm_dir.mkdir(parents=True)
    pd.DataFrame({"storm_id": ["AL082025"]}).to_csv(
        storm_dir / "samples.csv", index=False
    )
    (storm_dir / "summary.json").write_text(
        json.dumps(
            {
                "storm_id": "AL082025",
                "split": "val",
                "evaluated_samples": 1,
                "window_hours": 12,
                "window_length": 12,
                "forecast_lead_hours": 12,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(explorer, "FORECAST_ROOT", source)

    with pytest.raises(ValueError, match="missing"):
        explorer.build_forecast_export("AL082025")


@pytest.mark.parametrize(
    ("row_change", "summary_change", "message"),
    [
        ({}, {"forecast_lead_hours": 24}, "12-hour lead"),
        ({"target_timestamp": "2025-09-27T17:54:49Z"}, {}, "13-hour lead"),
    ],
)
def test_forecast_export_rejects_non_twelve_hour_contract(
    tmp_path, monkeypatch, row_change, summary_change, message
) -> None:
    source = tmp_path / "source"
    row = forecast_row()
    row.update(row_change)
    summary = {
        "storm_id": "AL082025",
        "split": "val",
        "evaluated_samples": 1,
        "window_hours": 12.0,
        "window_length": 12,
        "forecast_lead_hours": 12.0,
    }
    summary.update(summary_change)
    write_forecast(source, row=row, summary=summary)
    monkeypatch.setattr(explorer, "FORECAST_ROOT", source)

    with pytest.raises(ValueError, match=message):
        explorer.build_forecast_export("AL082025")
