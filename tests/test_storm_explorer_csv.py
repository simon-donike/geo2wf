import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_storm_explorer_data as explorer  # noqa: E402


def test_observation_csv_flattens_records_and_preserves_lists(tmp_path) -> None:
    payload = {
        "storms": [
            {
                "id": "AL012026",
                "name": "TEST",
                "basin": "North Atlantic",
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-01-01T03:00:00Z",
                "inference_available": True,
                "available_models": ["vit", "diffusion"],
                "records": [
                    {
                        "time": "2026-01-01T00:00:00Z",
                        "category": 1,
                        "diffusion_prediction": {
                            "max": 42.5,
                            "uncertainty": {"metrics": {"max": {"p90": 46.2}}},
                        },
                        "geo_overlay": {"bounds": [[10.0, -50.0], [12.0, -48.0]]},
                    }
                ],
            }
        ]
    }
    output = tmp_path / "storm-data.csv"

    assert explorer.write_observation_csv(payload, output) == 1

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["storm_id"] == "AL012026"
    assert json.loads(rows[0]["available_models"]) == ["vit", "diffusion"]
    assert rows[0]["diffusion_prediction.max"] == "42.5"
    assert rows[0]["diffusion_prediction.uncertainty.metrics.max.p90"] == "46.2"
    assert json.loads(rows[0]["geo_overlay.bounds"]) == [
        [10.0, -50.0],
        [12.0, -48.0],
    ]


def test_observation_csv_uses_union_of_record_columns(tmp_path) -> None:
    payload = {
        "storms": [
            {
                "id": "EP012026",
                "records": [
                    {"time": "2026-01-01T00:00:00Z", "sar": None},
                    {"time": "2026-01-01T03:00:00Z", "sar": {"max": 30.0}},
                ],
            }
        ]
    }
    output = tmp_path / "storm-data.csv"

    explorer.write_observation_csv(payload, output)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "sar.max" in rows[0]
    assert rows[0]["sar.max"] == ""
    assert rows[1]["sar.max"] == "30.0"
