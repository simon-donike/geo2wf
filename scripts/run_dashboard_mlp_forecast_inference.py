#!/usr/bin/env python3
"""Run the trained scalar MLP as retrospective +12 h dashboard forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo2wf.data.intensity import KNOT_TO_MS  # noqa: E402
from geo2wf.models.intensity_forecast import IntensityForecastMLP  # noqa: E402
from scripts.export_intensity_forecast_cache import (  # noqa: E402
    _track_lookup,
    _window,
    read_ibtracs_forecast_tracks,
)


DEFAULT_STORMS = ("AL082025", "EP112025", "EP182023")
QUADRANTS = ("ne", "se", "sw", "nw")
RADII = ("r34", "r50", "r64")
METRICS = ("max_wind_m_s", "rmw_km") + tuple(
    f"{radius}_{quadrant}_km" for radius in RADII for quadrant in QUADRANTS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "logs/20260805-180538_modular/checkpoints/epoch=051-step=14976.ckpt",
    )
    parser.add_argument(
        "--ibtracs-file",
        type=Path,
        default=ROOT / "data/IBTrACs/ibtracs.ALL.list.v04r01.csv",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "inference/forecasts/mlp"
    )
    parser.add_argument("--storm-id", action="append", dest="storm_ids")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty_metrics(row: dict[str, object], prefix: str) -> None:
    for metric in METRICS:
        row[f"{prefix}_{metric}"] = np.nan
        row[f"{prefix}_{metric}_valid"] = False


def inference_rows(
    model: IntensityForecastMLP,
    tracks: pd.DataFrame,
    storm_id: str,
    device: str,
) -> list[dict[str, object]]:
    lookup = _track_lookup(tracks)
    candidates = []
    for record in tracks.loc[tracks["storm_id"].eq(storm_id)].itertuples(index=False):
        issue_time = pd.Timestamp(record.timestamp)
        window = _window(lookup, storm_id, issue_time)
        if window is None or not np.isfinite(window["plus_12h_wind_kt"]):
            continue
        candidates.append((issue_time, window))
    if not candidates:
        raise ValueError(f"No complete +12 h IBTrACS windows for {storm_id}")

    anchor = torch.tensor(
        [item[1]["current_wind_kt"] * KNOT_TO_MS for item in candidates],
        dtype=torch.float32,
        device=device,
    )
    minus_6 = torch.tensor(
        [item[1]["minus_6h_wind_kt"] * KNOT_TO_MS for item in candidates],
        dtype=torch.float32,
        device=device,
    )
    minus_12 = torch.tensor(
        [item[1]["minus_12h_wind_kt"] * KNOT_TO_MS for item in candidates],
        dtype=torch.float32,
        device=device,
    )
    model = model.eval().to(device)
    with torch.inference_mode():
        _, plus_12 = model.predict_two_steps(anchor, minus_6, minus_12)
    predictions = plus_12.cpu().tolist()

    rows = []
    for (issue_time, window), prediction in zip(candidates, predictions):
        valid_time = issue_time + pd.Timedelta(hours=12)
        row: dict[str, object] = {
            "sample_id": f"mlp:{storm_id}:{issue_time:%Y%m%dT%H%M%SZ}",
            "storm_id": storm_id,
            "reference_timestamp": issue_time.isoformat(),
            "target_timestamp": valid_time.isoformat(),
            "target_provenance": "ibtracs",
        }
        for prefix in ("predicted", "ibtracs", "sar_derived"):
            _empty_metrics(row, prefix)
        row["predicted_max_wind_m_s"] = float(prediction)
        row["predicted_max_wind_m_s_valid"] = True
        row["ibtracs_max_wind_m_s"] = float(window["plus_12h_wind_kt"]) * KNOT_TO_MS
        row["ibtracs_max_wind_m_s_valid"] = True
        rows.append(row)
    return rows


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    os.replace(temporary, path)


def write_storm_bundle(
    output_root: Path,
    checkpoint: Path,
    storm_id: str,
    rows: list[dict[str, object]],
) -> None:
    storm_dir = output_root / storm_id
    samples = pd.DataFrame(rows)
    _write_text_atomic(storm_dir / "samples.csv", samples.to_csv(index=False))
    summary = {
        "storm_id": storm_id,
        "split": "retrospective",
        "evaluated_samples": len(rows),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "model_style": "mlp",
        "model_label": "MLP",
        "input_source": "ibtracs",
        "window_hours": 12.0,
        "window_length": 3,
        "forecast_lead_hours": 12.0,
    }
    _write_text_atomic(
        storm_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Forecast checkpoint does not exist: {args.checkpoint}"
        )
    tracks = read_ibtracs_forecast_tracks(args.ibtracs_file)
    model = IntensityForecastMLP.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    storm_ids = tuple(args.storm_ids or DEFAULT_STORMS)
    total = 0
    for storm_id in storm_ids:
        rows = inference_rows(model, tracks, storm_id, args.device)
        write_storm_bundle(args.output_root, args.checkpoint, storm_id, rows)
        total += len(rows)
        print(f"Wrote {storm_id}: {len(rows)} MLP +12 h forecasts")
    print(f"Wrote {total} forecasts to {args.output_root}")


if __name__ == "__main__":
    main()
