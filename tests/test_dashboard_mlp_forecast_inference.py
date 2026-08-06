from pathlib import Path

import pandas as pd
import torch

from scripts.run_dashboard_mlp_forecast_inference import (
    inference_rows,
    write_storm_bundle,
)


class PersistencePlusTwo(torch.nn.Module):
    def predict_two_steps(self, anchor, minus_6, minus_12):
        del minus_6, minus_12
        return anchor + 1, anchor + 2


def tracks(storm_id="AL082025"):
    start = pd.Timestamp("2025-09-01T00:00:00Z")
    return pd.DataFrame(
        [
            {
                "storm_id": storm_id,
                "timestamp": start + pd.Timedelta(hours=hour),
                "wind_kt": 30 + hour,
                "category": 0,
            }
            for hour in range(0, 37, 3)
        ]
    )


def test_dashboard_mlp_inference_emits_compatible_twelve_hour_rows() -> None:
    rows = inference_rows(PersistencePlusTwo(), tracks(), "AL082025", "cpu")

    assert len(rows) == 5
    assert pd.Timestamp(rows[0]["target_timestamp"]) - pd.Timestamp(
        rows[0]["reference_timestamp"]
    ) == pd.Timedelta(hours=12)
    assert rows[0]["predicted_max_wind_m_s_valid"] is True
    assert rows[0]["predicted_rmw_km_valid"] is False
    assert rows[0]["target_provenance"] == "ibtracs"


def test_dashboard_mlp_bundle_records_model_style(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    rows = inference_rows(PersistencePlusTwo(), tracks(), "AL082025", "cpu")

    write_storm_bundle(tmp_path / "output", checkpoint, "AL082025", rows)

    summary = (tmp_path / "output/AL082025/summary.json").read_text(encoding="utf-8")
    samples = pd.read_csv(tmp_path / "output/AL082025/samples.csv")
    assert '"model_style": "mlp"' in summary
    assert len(samples) == len(rows)
