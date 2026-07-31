"""Export browser-ready metrics and SAR overlays from inference bundles."""

import json
import math
import re
from datetime import datetime
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from export_geostat_images import GEOSTAT_SCALE_MAX_K, GEOSTAT_SCALE_MIN_K, export_geostat_image

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "inference" / "inf_anna"
MODEL_B_ROOT = ROOT / "inference" / "inf_simon"
MODEL_C_ROOT = ROOT / "inference" / "inf_model_c"
NWP_ROOT = ROOT / "inference" / "NWP"
OUTPUT_PATH = ROOT / "docs" / "explorer" / "storm-data.json"
SAR_IMAGE_DIR = OUTPUT_PATH.parent / "sar"
GEO_IMAGE_DIR = OUTPUT_PATH.parent / "geo"
CORE_RADIUS_KM = 100.0
RMW_BIN_KM = 10.0
SAR_SCALE_MIN_MS = 0.0
POSTPROCESS_C02_P99_MAX = 0.4
POSTPROCESS_SMOOTHING_HOURS = 6.0
POSTPROCESS_EDGE_PADDING_ACQUISITIONS = 1
SAR_SCALE_MAX_MS = 60.0
NWP_STORM_ID = "AL082025"
NWP_LABELS = {"aifs": "AIFS", "aifs2": "AIFS2", "era5": "ERA5", "gfs": "GFS", "graphcast": "GraphCast", "pangu": "Pangu"}


def finite_number(value):
    value = float(value)
    return round(value, 3) if math.isfinite(value) else None


def field_metrics(field, mask, distance):
    field = field.squeeze().float()
    valid = mask.bool() & torch.isfinite(field) & torch.isfinite(distance)
    values = field[valid]
    if not values.numel():
        return {key: None for key in ("max", "p90", "mean", "core_mean", "rmw")}
    core = values[distance[valid] <= CORE_RADIUS_KM]
    radial_means, radial_centers = [], []
    for lower in torch.arange(0, float(distance[valid].max()), RMW_BIN_KM):
        annulus = valid & (distance >= lower) & (distance < lower + RMW_BIN_KM)
        if int(annulus.sum()) >= 8:
            radial_means.append(float(field[annulus].mean()))
            radial_centers.append(float(lower + RMW_BIN_KM / 2))
    rmw = radial_centers[radial_means.index(max(radial_means))] if radial_means else None
    return {
        "max": finite_number(values.max()),
        "p90": finite_number(torch.quantile(values, 0.9)),
        "mean": finite_number(values.mean()),
        "core_mean": finite_number(core.mean()) if core.numel() else None,
        "rmw": finite_number(rmw) if rmw is not None else None,
    }


def export_sar_image(observation_id, bundle):
    field = bundle["sar_data"].squeeze().float().numpy()
    mask = bundle["sar_mask"].bool().numpy() & np.isfinite(field)
    valid = field[mask]
    if not valid.size:
        return None
    low, high = np.nanmin(valid), np.nanmax(valid)
    stretched = np.nan_to_num(np.clip((field - SAR_SCALE_MIN_MS) / (SAR_SCALE_MAX_MS - SAR_SCALE_MIN_MS), 0, 1))
    green = np.array([35, 139, 105], dtype=np.float32)
    yellow = np.array([249, 205, 67], dtype=np.float32)
    red = np.array([213, 62, 48], dtype=np.float32)
    lower = green + stretched[..., None] * 2 * (yellow - green)
    upper = yellow + (stretched[..., None] - 0.5) * 2 * (red - yellow)
    rgb = np.where((stretched <= 0.5)[..., None], lower, upper).astype(np.uint8)
    rgba = np.concatenate([rgb, np.where(mask, 220, 0).astype(np.uint8)[..., None]], axis=-1)
    filename = re.sub(r"[^a-zA-Z0-9_-]+", "_", observation_id) + ".png"
    iio.imwrite(SAR_IMAGE_DIR / filename, rgba)
    lat, lon = bundle["grid_lat"].numpy(), bundle["grid_lon"].numpy()
    return {
        "image": f"sar/{filename}",
        "bounds": [
            [finite_number(np.nanmin(lat)), finite_number(np.nanmin(lon))],
            [finite_number(np.nanmax(lat)), finite_number(np.nanmax(lon))],
        ],
        "min": finite_number(low),
        "max": finite_number(high),
    }


def export_nwp():
    """Return NWP maximum-wind tracks in the browser format."""
    series = []
    for path in sorted(NWP_ROOT.glob("*.csv")):
        key = path.stem.lower()
        if key not in NWP_LABELS:
            continue
        table = pd.read_csv(path)
        if not {"valid_time", "max_wind_ms"}.issubset(table.columns):
            raise ValueError(f"{path} must contain valid_time and max_wind_ms")
        points = [{"time": pd.Timestamp(row.valid_time).isoformat().replace("+00:00", "Z"), "max": finite_number(row.max_wind_ms)} for row in table.itertuples(index=False) if pd.notna(row.max_wind_ms)]
        series.append({"id": key, "label": NWP_LABELS[key], "points": points})
    return series

def export_storm(storm_dir):
    summary = pd.read_csv(storm_dir / "inference-summary.csv").sort_values("observation_timestamp").reset_index(drop=True)
    model_b_path = MODEL_B_ROOT / storm_dir.name / "inference-summary.csv"
    model_b = pd.read_csv(model_b_path).set_index("observation_id")
    model_c_path = MODEL_C_ROOT / storm_dir.name / "inference-summary.csv"
    model_c = pd.read_csv(model_c_path).set_index("observation_id")
    missing_model_b = set(summary["observation_id"]) - set(model_b.index)
    if missing_model_b:
        raise ValueError(f"{model_b_path} is missing {len(missing_model_b)} explorer observations")
    missing_model_c = set(summary["observation_id"]) - set(model_c.index)
    if missing_model_c:
        raise ValueError(f"{model_c_path} is missing {len(missing_model_c)} explorer observations")
    timestamps = pd.to_datetime(summary["observation_timestamp"])
    targets = pd.date_range(timestamps.iloc[0], timestamps.iloc[-1], freq="3h")
    geo_indices = {int((timestamps - target).abs().argmin()) for target in targets}
    records = []
    for index, row in enumerate(summary.itertuples(index=False)):
        bundle = torch.load(storm_dir / row.inference_path, map_location="cpu", weights_only=False)
        c02 = bundle["input"][bundle["input_channels"].index("CMI_C02")]
        c02_valid = c02[bundle["input_mask"].bool() & torch.isfinite(c02)]
        c02_p99 = finite_number(torch.quantile(c02_valid, 0.99)) if c02_valid.numel() else None
        geo_overlay = export_geostat_image(row.observation_id, bundle, GEO_IMAGE_DIR) if index in geo_indices else None
        prediction = field_metrics(bundle["output"], bundle["input_mask"], bundle["distance_to_center"])
        prediction["r64"] = finite_number(bundle["output_metrics"]["output_r64_km"])
        model_b_row = model_b.loc[row.observation_id]
        model_b_prediction = {"max": finite_number(model_b_row.output_msw_ms), "p90": finite_number(model_b_row.output_p90_ms), "mean": finite_number(model_b_row.output_mean_ms), "core_mean": finite_number(model_b_row.output_core_mean_ms), "rmw": finite_number(model_b_row.output_rmw_km), "r64": finite_number(model_b_row.output_r64_km)}
        model_c_row = model_c.loc[row.observation_id]
        model_c_prediction = {"max": finite_number(model_c_row.output_msw_ms), "p90": finite_number(model_c_row.output_p90_ms), "mean": finite_number(model_c_row.output_mean_ms), "core_mean": finite_number(model_c_row.output_core_mean_ms), "rmw": finite_number(model_c_row.output_rmw_km), "r64": finite_number(model_c_row.output_r64_km)}
        sar = None
        sar_overlay = None
        if bundle["sar_data"] is not None:
            sar = field_metrics(bundle["sar_data"], bundle["sar_mask"], bundle["distance_to_center"])
            sar_overlay = export_sar_image(row.observation_id, bundle)
        meta = bundle["meta"]
        records.append({
            "time": datetime.fromisoformat(str(row.observation_timestamp)).isoformat() + "Z",
            "lat": finite_number(meta["ibtracs_center_lat"]),
            "lon": finite_number(meta["ibtracs_center_lon"]),
            "category": int(row.ibtracs_category),
            "ibtracs_msw": finite_number(row.ibtracs_msw_ms),
            "prediction": prediction,
            "model_b_prediction": model_b_prediction,
            "model_c_prediction": model_c_prediction,
            "c02_p99": c02_p99,
            "postprocess_excluded": c02_p99 is not None and c02_p99 > POSTPROCESS_C02_P99_MAX,
            "geo_overlay": geo_overlay,
            "sar": sar,
            "sar_overlay": sar_overlay,
            "sar_dt_minutes": finite_number(row.sar_dt_minutes) if pd.notna(row.sar_dt_minutes) else None,
        })
    base_exclusions = [record["postprocess_excluded"] for record in records]
    for index, record in enumerate(records):
        start = max(0, index - POSTPROCESS_EDGE_PADDING_ACQUISITIONS)
        end = min(len(records), index + POSTPROCESS_EDGE_PADDING_ACQUISITIONS + 1)
        record["postprocess_excluded"] = any(base_exclusions[start:end])
    storm = {
        "id": storm_dir.name,
        "basin": "North Atlantic" if storm_dir.name.startswith("AL") else "Eastern Pacific",
        "start": records[0]["time"], "end": records[-1]["time"],
        "sar_matches": sum(record["sar"] is not None for record in records),
        "records": records,
    }
    if storm_dir.name == NWP_STORM_ID:
        storm["nwp"] = export_nwp()
    return storm


def main():
    SAR_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    GEO_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for existing in GEO_IMAGE_DIR.glob("*.webp"):
        existing.unlink()
    storms = [export_storm(path) for path in sorted(INPUT_ROOT.iterdir()) if path.is_dir()]
    payload = {
        "generated_from": "inference/inf_anna",
        "geo_interval_hours": 3,
        "postprocessing": {"enabled_default": True, "c02_p99_max": POSTPROCESS_C02_P99_MAX, "smoothing_hours": POSTPROCESS_SMOOTHING_HOURS, "edge_padding_acquisitions": POSTPROCESS_EDGE_PADDING_ACQUISITIONS, "method": "centered median + cubic smoothstep interpolation"},
        "geostat_color_scale": {"min": GEOSTAT_SCALE_MIN_K, "mid": (GEOSTAT_SCALE_MIN_K + GEOSTAT_SCALE_MAX_K) / 2, "max": GEOSTAT_SCALE_MAX_K, "unit": "K", "channel": "CMI_C15"},
        "sar_color_scale": {"min": SAR_SCALE_MIN_MS, "mid": 30.0, "max": SAR_SCALE_MAX_MS, "unit": "m/s"},
        "metrics": {
            "max": {"label": "Maximum wind", "unit": "m/s"},
            "p90": {"label": "90th-percentile wind", "unit": "m/s"},
            "mean": {"label": "Mean wind", "unit": "m/s"},
            "core_mean": {"label": "Inner-core mean", "unit": "m/s", "note": "Within 100 km"},
            "rmw": {"label": "Radius of maximum wind", "unit": "km"},
            "r64": {"label": "Radius of 64-knot winds", "unit": "km"},
        },
        "models": {"model_a": {"label": "Model A", "metrics": ["max", "p90", "mean", "core_mean", "rmw", "r64"]}, "model_b": {"label": "Model B", "metrics": ["max", "p90", "mean", "core_mean", "rmw", "r64"]}, "model_c": {"label": "Model C", "metrics": ["max", "p90", "mean", "core_mean", "rmw", "r64"]}},
        "storms": storms,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} with {sum(len(s['records']) for s in storms)} observations and {sum(s['sar_matches'] for s in storms)} SAR overlays")


if __name__ == "__main__":
    main()
