"""Report forecast errors without mixing hindcasts and issued live predictions."""

from __future__ import annotations

import csv
import gzip

from .common import iso, utc, file_hash
from .models import MANIFEST, resolve_file, version


def training_storms(model_root):
    path = resolve_file(MANIFEST["models"]["nowcast"]["training_manifest"], model_root)
    if file_hash(path) != MANIFEST["models"]["nowcast"]["training_manifest_sha256"]:
        raise ValueError("Training membership manifest checksum mismatch")
    with gzip.open(path, "rt") as handle:
        rows = csv.DictReader(handle)
        if "storm_id" not in rows.fieldnames:
            raise ValueError("training membership unavailable")
        return {r["storm_id"] for r in rows}


def evaluate(store, model_root, start=None, end=None):
    training = training_storms(model_root)
    groups = {}
    anchors = []
    forecast_version = version("forecast") + ":" + version("nowcast")
    for storm in store.storms():
        reference = {
            f["time"]: f["wind_ms"]
            for f in storm.get("track", [])
            if f["wind_ms"] is not None
        }
        for forecast in store.forecasts(storm["id"]):
            if forecast["model_version"] != forecast_version:
                continue
            if start is not None and utc(forecast["anchor_time"]) < utc(start):
                continue
            if end is not None and utc(forecast["anchor_time"]) > utc(end):
                continue
            anchors.append(forecast["anchor_time"])
            a, b, _ = forecast["input_vmax_ms"]
            for prediction in forecast["predictions"]:
                target = reference.get(prediction["valid_time"])
                if target is None:
                    continue
                memberships = ["all"] + (
                    ["excluded_from_nowcast_training"]
                    if storm["id"] not in training
                    else []
                )
                for membership in memberships:
                    key = (
                        f"{forecast['kind']}/{membership}/+{prediction['lead_hours']}h"
                    )
                    groups.setdefault(key, []).append(
                        {
                            "storm_id": storm["id"],
                            "error": prediction["vmax_ms"] - target,
                            "persistence_error": a - target,
                            "trend_error": max(
                                0, a + (a - b) * prediction["lead_hours"] / 6
                            )
                            - target,
                        }
                    )
    report = {
        "schema_version": 1,
        "generated_at": iso(),
        "experimental": True,
        "reference": "NHC/CPHC BEST exact valid-time fixes; revised tracks possible",
        "forecast_training": "IBTrACS 2000–2018",
        "model_version": forecast_version,
        "anchor_window": {
            "start": iso(start) if start is not None else None,
            "end": iso(end) if end is not None else None,
        },
        "anchor_range": (
            {"start": min(anchors), "end": max(anchors)} if anchors else None
        ),
        "nowcast_training_manifest_sha256": MANIFEST["models"]["nowcast"][
            "training_manifest_sha256"
        ],
        "note": "Retrospective results are not evidence of as-issued operational skill. Missing reference times are excluded.",
        "groups": {},
    }
    for key, rows in groups.items():
        import math

        errors = [r["error"] for r in rows]
        storm_mae = [
            sum(abs(r["error"]) for r in rows if r["storm_id"] == sid)
            / sum(r["storm_id"] == sid for r in rows)
            for sid in {r["storm_id"] for r in rows}
        ]
        report["groups"][key] = {
            "samples": len(rows),
            "storms": len(storm_mae),
            "storm_ids": sorted({r["storm_id"] for r in rows}),
            "mae_ms": sum(map(abs, errors)) / len(rows),
            "rmse_ms": math.sqrt(sum(e * e for e in errors) / len(rows)),
            "bias_ms": sum(errors) / len(rows),
            "storm_macro_mae_ms": sum(storm_mae) / len(storm_mae),
            "persistence_mae_ms": sum(abs(r["persistence_error"]) for r in rows)
            / len(rows),
            "trend_mae_ms": sum(abs(r["trend_error"]) for r in rows) / len(rows),
        }
    store.put_status("evaluation", report)
    return report
