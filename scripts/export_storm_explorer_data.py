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
from export_geostat_images import (
    GEOSTAT_SCALE_MAX_K,
    GEOSTAT_SCALE_MIN_K,
    export_geostat_array,
    export_geostat_image,
)
from export_geo_sar_geotiffs import (
    PMW_CHANNELS,
    _load_geo_channels,
    _load_pmw_channels,
    _load_sar_channels,
    _read_manifest,
    _regrid,
)

ROOT = Path(__file__).resolve().parents[1]
VIT_ROOT = ROOT / "inference" / "inf_vit"
UNET_ROOT = ROOT / "inference" / "inf_unet"
DIFFUSION_ROOT = ROOT / "inference" / "inf_diffusion"
NWP_ROOT = ROOT / "inference" / "NWP"
RAW_INPUT_ROOT = ROOT / "inference" / "inf_data"
RAW_MANIFEST = RAW_INPUT_ROOT / "index-files" / "observation_manifest_v6.csv"
OUTPUT_PATH = ROOT / "docs" / "explorer" / "storm-data.json"
SAR_IMAGE_DIR = OUTPUT_PATH.parent / "sar"
PMW_IMAGE_DIR = OUTPUT_PATH.parent / "pmw"
GEO_IMAGE_DIR = OUTPUT_PATH.parent / "geo"
CORE_RADIUS_KM = 100.0
RMW_BIN_KM = 10.0
SAR_SCALE_MIN_MS = 0.0
POSTPROCESS_C02_P99_MAX = 0.4
POSTPROCESS_SMOOTHING_HOURS = 6.0
POSTPROCESS_EDGE_PADDING_ACQUISITIONS = 1
SAR_SCALE_MAX_MS = 60.0
PMW_SCALE_MIN_K = GEOSTAT_SCALE_MIN_K
PMW_SCALE_MAX_K = GEOSTAT_SCALE_MAX_K
NWP_LABELS = {
    "aifs": "AIFS",
    "aifs2": "AIFS2",
    "era5": "ERA5",
    "gfs": "GFS",
    "graphcast": "GraphCast",
    "pangu": "Pangu",
}


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
    rmw = (
        radial_centers[radial_means.index(max(radial_means))] if radial_means else None
    )
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
    return export_sar_array(
        observation_id,
        field,
        mask,
        bundle["grid_lat"].numpy(),
        bundle["grid_lon"].numpy(),
    )


def export_sar_array(observation_id, field, mask, lat, lon):
    """Export a physical SAR field without requiring an inference bundle."""
    field = np.asarray(field, dtype=np.float32).squeeze()
    mask = np.asarray(mask, dtype=bool).squeeze() & np.isfinite(field)
    valid = field[mask]
    if not valid.size:
        return None
    low, high = np.nanmin(valid), np.nanmax(valid)
    stretched = np.nan_to_num(
        np.clip(
            (field - SAR_SCALE_MIN_MS) / (SAR_SCALE_MAX_MS - SAR_SCALE_MIN_MS), 0, 1
        )
    )
    green = np.array([35, 139, 105], dtype=np.float32)
    yellow = np.array([249, 205, 67], dtype=np.float32)
    red = np.array([213, 62, 48], dtype=np.float32)
    lower = green + stretched[..., None] * 2 * (yellow - green)
    upper = yellow + (stretched[..., None] - 0.5) * 2 * (red - yellow)
    rgb = np.where((stretched <= 0.5)[..., None], lower, upper).astype(np.uint8)
    rgba = np.concatenate(
        [rgb, np.where(mask, 220, 0).astype(np.uint8)[..., None]], axis=-1
    )
    filename = re.sub(r"[^a-zA-Z0-9_-]+", "_", observation_id) + ".png"
    iio.imwrite(SAR_IMAGE_DIR / filename, rgba)
    return {
        "image": f"sar/{filename}",
        "bounds": [
            [finite_number(np.nanmin(lat)), finite_number(np.nanmin(lon))],
            [finite_number(np.nanmax(lat)), finite_number(np.nanmax(lon))],
        ],
        "min": finite_number(low),
        "max": finite_number(high),
    }


def export_pmw_array(
    observation_id,
    field,
    mask,
    lat,
    lon,
    sensor,
    channel,
    output_dir=PMW_IMAGE_DIR,
):
    """Export one physical high-frequency PMW band on a shared scale."""
    field = np.asarray(field, dtype=np.float32).squeeze()
    lat = np.asarray(lat, dtype=np.float32).squeeze()
    lon = np.asarray(lon, dtype=np.float32).squeeze()
    coordinate_mask = np.isfinite(lat) & np.isfinite(lon)
    mask = np.asarray(mask, dtype=bool).squeeze() & np.isfinite(field) & coordinate_mask
    valid = field[mask]
    if not valid.size:
        return None
    stretched = np.nan_to_num(
        np.clip(
            (field - PMW_SCALE_MIN_K) / (PMW_SCALE_MAX_K - PMW_SCALE_MIN_K),
            0,
            1,
        )
    )
    cold = np.array([255, 255, 255], dtype=np.float32)
    warm = np.array([22, 82, 180], dtype=np.float32)
    rgb = (cold + stretched[..., None] * (warm - cold)).astype(np.uint8)
    rgba = np.concatenate(
        [rgb, np.where(mask, 225, 0).astype(np.uint8)[..., None]], axis=-1
    )
    filename = re.sub(r"[^a-zA-Z0-9_-]+", "_", observation_id) + ".png"
    output_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_dir / filename, rgba)
    return {
        "image": f"pmw/{filename}",
        "bounds": [
            [
                finite_number(np.nanmin(lat[coordinate_mask])),
                finite_number(np.nanmin(lon[coordinate_mask])),
            ],
            [
                finite_number(np.nanmax(lat[coordinate_mask])),
                finite_number(np.nanmax(lon[coordinate_mask])),
            ],
        ],
        "min": finite_number(valid.min()),
        "max": finite_number(valid.max()),
        "sensor": sensor,
        "channel": channel,
    }


def export_pmw_observations(storm, raw_records):
    """Export PMW on the nearest displayed geostationary-image extent."""
    start = pd.Timestamp(storm["start"])
    end = pd.Timestamp(storm["end"])
    geo_overlays = [
        (pd.Timestamp(record["time"]), record["geo_overlay"])
        for record in storm["records"]
        if record["geo_overlay"] is not None
    ]
    if not geo_overlays:
        raise ValueError(f"Storm {storm['id']} has no geostationary overlay extents")
    observations = []
    for observation in sorted(
        (
            record
            for record in raw_records
            if record.storm_id == storm["id"]
            and record.source_type == "pmw"
            and start <= record.timestamp <= end
        ),
        key=lambda record: record.timestamp,
    ):
        channel = PMW_CHANNELS[observation.sensor][0]
        field, lat, lon = _load_pmw_channels(observation, [channel])[channel]
        _, geo_overlay = min(
            geo_overlays,
            key=lambda item: abs((item[0] - observation.timestamp).total_seconds()),
        )
        (south, west), (north, east) = geo_overlay["bounds"]
        size = int(geo_overlay["size"])
        grid_lon, grid_lat = np.meshgrid(
            np.linspace(west, east, size),
            np.linspace(north, south, size),
        )
        field, mask = _regrid(field, lat, lon, grid_lat, grid_lon)
        overlay = export_pmw_array(
            observation.observation_id,
            field,
            mask,
            grid_lat,
            grid_lon,
            observation.sensor,
            channel,
        )
        if overlay is None:
            continue
        center_lat = observation.ibtracs_center_lat
        center_lon = observation.ibtracs_center_lon
        if center_lat is None or center_lon is None:
            center_lat = np.nanmedian(lat[mask])
            center_lon = np.nanmedian(lon[mask])
        observations.append(
            {
                "time": observation.timestamp.isoformat().replace("+00:00", "Z"),
                "lat": finite_number(center_lat),
                "lon": finite_number(center_lon),
                "overlay": overlay,
            }
        )
    return observations


def export_nwp(storm_id):
    """Return NWP maximum-wind tracks in the browser format."""
    series = []
    storm_root = NWP_ROOT / storm_id
    if not storm_root.is_dir():
        raise ValueError(
            f"Missing NWP directory for manifest storm {storm_id}: {storm_root}"
        )
    for path in sorted(storm_root.glob("*.csv")):
        key = path.stem.lower()
        if key not in NWP_LABELS:
            continue
        table = pd.read_csv(path)
        if not {"valid_time", "max_wind_ms"}.issubset(table.columns):
            raise ValueError(f"{path} must contain valid_time and max_wind_ms")
        points = [
            {
                "time": pd.Timestamp(row.valid_time).isoformat().replace("+00:00", "Z"),
                "max": finite_number(row.max_wind_ms),
            }
            for row in table.itertuples(index=False)
            if pd.notna(row.max_wind_ms)
        ]
        series.append({"id": key, "label": NWP_LABELS[key], "points": points})
    return series


def _physical_distance_km(lat, lon, center):
    lat = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon, dtype=np.float64))
    center_lat, center_lon = np.deg2rad(center)
    dlat = lat - center_lat
    dlon = lon - center_lon
    value = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(center_lat) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    )
    return (
        6371.0
        * 2.0
        * np.arctan2(
            np.sqrt(np.clip(value, 0.0, 1.0)),
            np.sqrt(np.clip(1.0 - value, 0.0, 1.0)),
        )
    )


def _selected_raw_geo(geo_records, sar_records):
    timestamps = pd.DatetimeIndex([record.timestamp for record in geo_records])
    targets = pd.date_range(timestamps[0], timestamps[-1], freq="3h")
    indices = {int(np.argmin(np.abs(timestamps - target))) for target in targets}
    for sar in sar_records:
        indices.add(int(np.argmin(np.abs(timestamps - sar.timestamp))))
    return [geo_records[index] for index in sorted(indices)]


def export_raw_storm(storm_id, raw_records, raw_metadata):
    """Export observations and imagery for a storm whose inference is pending."""
    geo_records = sorted(
        (
            record
            for record in raw_records
            if record.storm_id == storm_id
            and record.source_type == "geo"
            and record.ibtracs_center is not None
        ),
        key=lambda record: record.timestamp,
    )
    sar_records = sorted(
        (
            record
            for record in raw_records
            if record.storm_id == storm_id
            and record.source_type == "sar"
            and record.ibtracs_center is not None
        ),
        key=lambda record: record.timestamp,
    )
    if not geo_records:
        raise ValueError(f"{RAW_MANIFEST} has no usable GEO rows for {storm_id}")

    selected_geo = _selected_raw_geo(geo_records, sar_records)
    records = []
    matched_sar_ids = set()
    for geo in selected_geo:
        c15, lat, lon = _load_geo_channels(geo, ["CMI_C15"])["CMI_C15"]
        valid = np.isfinite(c15) & np.isfinite(lat) & np.isfinite(lon)
        geo_overlay = export_geostat_array(
            geo.observation_id,
            c15,
            valid,
            lat,
            lon,
            GEO_IMAGE_DIR,
        )

        nearest_sar = min(
            sar_records,
            key=lambda record: abs((record.timestamp - geo.timestamp).total_seconds()),
            default=None,
        )
        sar = None
        sar_overlay = None
        sar_dt_minutes = None
        if nearest_sar is not None:
            sar_dt_minutes = (
                nearest_sar.timestamp - geo.timestamp
            ).total_seconds() / 60.0
            if abs(sar_dt_minutes) <= 30.0:
                field, sar_lat, sar_lon = _load_sar_channels(
                    nearest_sar, ["wind_speed"]
                )["wind_speed"]
                sar_mask = (
                    np.isfinite(field) & np.isfinite(sar_lat) & np.isfinite(sar_lon)
                )
                distance = _physical_distance_km(
                    sar_lat, sar_lon, nearest_sar.ibtracs_center
                )
                sar = field_metrics(
                    torch.from_numpy(field),
                    torch.from_numpy(sar_mask),
                    torch.from_numpy(distance),
                )
                sar_overlay = export_sar_array(
                    nearest_sar.observation_id,
                    field,
                    sar_mask,
                    sar_lat,
                    sar_lon,
                )
                matched_sar_ids.add(nearest_sar.observation_id)
            else:
                sar_dt_minutes = None

        metadata = raw_metadata.get(geo.observation_id, {})
        wind_kt = metadata.get("usa_wind")
        category = metadata.get("usa_sshs")
        records.append(
            {
                "time": geo.timestamp.isoformat().replace("+00:00", "Z"),
                "lat": finite_number(geo.ibtracs_center_lat),
                "lon": finite_number(geo.ibtracs_center_lon),
                "category": int(category) if category is not None else None,
                "ibtracs_msw": (
                    finite_number(float(wind_kt) * 0.514444)
                    if wind_kt is not None
                    else None
                ),
                "vit_prediction": None,
                "unet_prediction": None,
                "diffusion_prediction": None,
                "c02_p99": None,
                "postprocess_excluded": False,
                "geo_overlay": geo_overlay,
                "sar": sar,
                "sar_overlay": sar_overlay,
                "sar_dt_minutes": (
                    finite_number(sar_dt_minutes)
                    if sar_dt_minutes is not None
                    else None
                ),
            }
        )

    return {
        "id": storm_id,
        "name": "OTIS" if storm_id == "EP182023" else storm_id,
        "basin": "North Atlantic" if storm_id.startswith("AL") else "Eastern Pacific",
        "start": records[0]["time"],
        "end": records[-1]["time"],
        "sar_matches": len(matched_sar_ids),
        "inference_available": False,
        "records": records,
    }


METRIC_COLUMNS = {
    "max": "output_msw_ms",
    "p90": "output_p90_ms",
    "mean": "output_mean_ms",
    "core_mean": "output_core_mean_ms",
    "rmw": "output_rmw_km",
    "r64": "output_r64_km",
}


def manifest_storm_ids():
    """Return the sorted unique storm IDs declared by the input manifest."""
    return sorted(
        pd.read_csv(RAW_MANIFEST, usecols=["storm_id"])["storm_id"]
        .dropna()
        .astype(str)
        .unique()
    )


def load_prediction_table(root, storm_id):
    path = root / storm_id / "inference-summary.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path).set_index("observation_id")


def tabular_prediction(table, observation_id, *, include_uncertainty=False):
    if table is None or observation_id not in table.index:
        return None
    row = table.loc[observation_id]
    prediction = {
        metric: finite_number(row.get(column, math.nan))
        for metric, column in METRIC_COLUMNS.items()
    }
    if include_uncertainty:
        prediction["uncertainty"] = {
            "ensemble_size": int(row["ensemble_size"]),
            "pixel_std": {
                statistic: finite_number(
                    row.get(f"ensemble_pixel_std_{statistic}_ms", math.nan)
                )
                for statistic in ("mean", "median", "p90", "max")
            },
            "metrics": {
                metric: {
                    statistic: finite_number(
                        row.get(f"{column}_member_{statistic}", math.nan)
                    )
                    for statistic in ("std", "min", "max", "range", "p10", "p90")
                }
                for metric, column in METRIC_COLUMNS.items()
            },
        }
    return prediction


def export_storm(storm_id):
    storm_dir = VIT_ROOT / storm_id
    summary_path = storm_dir / "inference-summary.csv"
    if not summary_path.is_file():
        raise ValueError(
            f"Missing ViT summary for manifest storm {storm_id}: {summary_path}"
        )
    summary = (
        pd.read_csv(summary_path)
        .sort_values("observation_timestamp")
        .reset_index(drop=True)
    )
    unet = load_prediction_table(UNET_ROOT, storm_id)
    diffusion = load_prediction_table(DIFFUSION_ROOT, storm_id)
    for label, table in (("UNet", unet), ("Diffusion", diffusion)):
        if table is None:
            continue
        missing = set(summary["observation_id"]) - set(table.index)
        if missing:
            print(
                f"{label} has no result for {len(missing)} stale ViT "
                f"observations in {storm_id}; exporting them as unavailable"
            )
    available_models = ["vit"]
    if unet is not None:
        available_models.append("unet")
    if diffusion is not None:
        available_models.append("diffusion")

    timestamps = pd.to_datetime(summary["observation_timestamp"])
    targets = pd.date_range(timestamps.iloc[0], timestamps.iloc[-1], freq="3h")
    geo_indices = {int((timestamps - target).abs().argmin()) for target in targets}
    records = []
    for index, row in enumerate(summary.itertuples(index=False)):
        bundle = torch.load(
            storm_dir / row.inference_path, map_location="cpu", weights_only=False
        )
        c02_p99 = None
        if "CMI_C02" in bundle["input_channels"]:
            c02 = bundle["input"][bundle["input_channels"].index("CMI_C02")]
            c02_valid = c02[bundle["input_mask"].bool() & torch.isfinite(c02)]
            if c02_valid.numel():
                c02_p99 = finite_number(torch.quantile(c02_valid, 0.99))
        geo_overlay = (
            export_geostat_image(row.observation_id, bundle, GEO_IMAGE_DIR)
            if index in geo_indices
            else None
        )
        vit_prediction = field_metrics(
            bundle["output"], bundle["input_mask"], bundle["distance_to_center"]
        )
        vit_prediction["r64"] = finite_number(row.output_r64_km)
        sar = None
        sar_overlay = None
        if bundle["sar_data"] is not None:
            sar = field_metrics(
                bundle["sar_data"], bundle["sar_mask"], bundle["distance_to_center"]
            )
            sar_overlay = export_sar_image(row.observation_id, bundle)
        meta = bundle["meta"]
        records.append(
            {
                "time": datetime.fromisoformat(
                    str(row.observation_timestamp)
                ).isoformat()
                + "Z",
                "lat": finite_number(meta["ibtracs_center_lat"]),
                "lon": finite_number(meta["ibtracs_center_lon"]),
                "category": int(row.ibtracs_category),
                "ibtracs_msw": finite_number(row.ibtracs_msw_ms),
                "vit_prediction": vit_prediction,
                "unet_prediction": tabular_prediction(unet, row.observation_id),
                "diffusion_prediction": tabular_prediction(
                    diffusion, row.observation_id, include_uncertainty=True
                ),
                "c02_p99": c02_p99,
                "postprocess_excluded": c02_p99 is not None
                and c02_p99 > POSTPROCESS_C02_P99_MAX,
                "geo_overlay": geo_overlay,
                "sar": sar,
                "sar_overlay": sar_overlay,
                "sar_dt_minutes": (
                    finite_number(row.sar_dt_minutes)
                    if pd.notna(row.sar_dt_minutes)
                    else None
                ),
            }
        )
    base_exclusions = [record["postprocess_excluded"] for record in records]
    for index, record in enumerate(records):
        start = max(0, index - POSTPROCESS_EDGE_PADDING_ACQUISITIONS)
        end = min(len(records), index + POSTPROCESS_EDGE_PADDING_ACQUISITIONS + 1)
        record["postprocess_excluded"] = any(base_exclusions[start:end])
    return {
        "id": storm_id,
        "name": "OTIS" if storm_id == "EP182023" else None,
        "basin": "North Atlantic" if storm_id.startswith("AL") else "Eastern Pacific",
        "start": records[0]["time"],
        "end": records[-1]["time"],
        "sar_matches": sum(record["sar"] is not None for record in records),
        "inference_available": True,
        "available_models": available_models,
        "nwp": export_nwp(storm_id),
        "records": records,
    }


def main():
    SAR_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PMW_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    GEO_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for existing in GEO_IMAGE_DIR.glob("*.webp"):
        existing.unlink()
    for existing in SAR_IMAGE_DIR.glob("*.png"):
        existing.unlink()
    for existing in PMW_IMAGE_DIR.glob("*.png"):
        existing.unlink()
    storm_ids = manifest_storm_ids()
    storms = [export_storm(storm_id) for storm_id in storm_ids]
    raw_records = _read_manifest(RAW_MANIFEST, RAW_INPUT_ROOT)
    for storm in storms:
        storm["pmw_observations"] = export_pmw_observations(storm, raw_records)
        storm["pmw_matches"] = len(storm["pmw_observations"])
    payload = {
        "generated_from": [
            "inference/inf_vit",
            "inference/inf_unet",
            "inference/inf_diffusion",
            "inference/NWP",
        ],
        "geo_interval_hours": 3,
        "postprocessing": {
            "enabled_default": True,
            "c02_p99_max": POSTPROCESS_C02_P99_MAX,
            "smoothing_hours": POSTPROCESS_SMOOTHING_HOURS,
            "edge_padding_acquisitions": POSTPROCESS_EDGE_PADDING_ACQUISITIONS,
            "method": "centered median + cubic smoothstep interpolation",
        },
        "geostat_color_scale": {
            "min": GEOSTAT_SCALE_MIN_K,
            "mid": (GEOSTAT_SCALE_MIN_K + GEOSTAT_SCALE_MAX_K) / 2,
            "max": GEOSTAT_SCALE_MAX_K,
            "unit": "K",
            "channel": "CMI_C15",
        },
        "sar_color_scale": {
            "min": SAR_SCALE_MIN_MS,
            "mid": 30.0,
            "max": SAR_SCALE_MAX_MS,
            "unit": "m/s",
        },
        "pmw_color_scale": {
            "min": PMW_SCALE_MIN_K,
            "mid": (PMW_SCALE_MIN_K + PMW_SCALE_MAX_K) / 2,
            "max": PMW_SCALE_MAX_K,
            "unit": "K",
            "channel": "89--92 GHz V",
        },
        "metrics": {
            "max": {"label": "Maximum wind", "unit": "m/s"},
            "p90": {"label": "90th-percentile wind", "unit": "m/s"},
            "mean": {"label": "Mean wind", "unit": "m/s"},
            "core_mean": {
                "label": "Inner-core mean",
                "unit": "m/s",
                "note": "Within 100 km",
            },
            "rmw": {"label": "Radius of maximum wind", "unit": "km"},
            "r64": {"label": "Radius of 64-knot winds", "unit": "km"},
        },
        "models": {
            "vit": {
                "label": "ViT",
                "metrics": ["max", "p90", "mean", "core_mean", "rmw", "r64"],
            },
            "unet": {
                "label": "UNet",
                "metrics": ["max", "p90", "mean", "core_mean", "rmw", "r64"],
            },
            "diffusion": {
                "label": "Diffusion",
                "metrics": ["max", "p90", "mean", "core_mean", "rmw", "r64"],
            },
        },
        "storms": storms,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(
        f"Wrote {OUTPUT_PATH.relative_to(ROOT)} with {sum(len(s['records']) for s in storms)} observations, {sum(s['sar_matches'] for s in storms)} SAR overlays, and {sum(s['pmw_matches'] for s in storms)} PMW overlays"
    )


if __name__ == "__main__":
    main()
