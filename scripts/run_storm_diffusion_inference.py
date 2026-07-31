"""Run deterministic-baseline residual diffusion for the explorer storms.

The legacy ``inference-summary.csv`` contract is retained, but ensemble
diagnostics are computed member-by-member before aggregation. Every run also
writes long-form member metrics and reproducibility metadata. Full member
fields can optionally be retained as compressed NumPy bundles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
import xarray as xr
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (  # noqa: E402
    _normalize,
    _normalized_distance_to_center,
    _solar_time_features,
)
from scripts.export_geo_sar_geotiffs import (  # noqa: E402
    ERA5_CHANNELS,
    GEO_CHANNEL_SETS,
    _append_native_era5_derived_fields,
    _era5_coordinate_values,
    _fix_longitudes,
    _load_geo_channels,
    _make_grid,
    _nearest_time_index,
    _read_manifest,
    _regrid,
    _regrid_continuous,
)
from src.ERA5Residual import ERA5ResidualRegressor  # noqa: E402
from train import build_model, resolve_runtime_config

DEFAULT_DATA_ROOT = ROOT / "inference" / "inf_data"
DEFAULT_REFERENCE_ROOT = ROOT / "inference" / "inf_anna"
DEFAULT_OUTPUT_ROOT = ROOT / "inference" / "inf_model_c"
DEFAULT_STATS = ROOT / "data" / "geotiff" / "geo_sar_10bands_era5" / "stats.json"
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "config_geo_sar_10bands_era5_diffusion_residual_deterministic.yaml"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "logs"
    / "20260730-150623_config_geo_sar_10bands_era5_diffusion_residual_deterministic"
    / "checkpoints"
    / "epoch=055-step=6832.ckpt"
)
GRID_SIZE = 256
GRID_RESOLUTION_DEGREES = 0.027
CROP_SIZE = 192
ROBUST_CLIP = 4.0
R64_THRESHOLD_MS = 64.0 * 0.514444

DEFAULT_ENSEMBLE_SIZE = 10
DEFAULT_MEMBER_QUANTILES = (0.1, 0.9)
METRIC_COLUMNS = {
    "msw": ("output_msw", "ms"),
    "robust_peak": ("output_robust_peak", "ms"),
    "r64": ("output_r64", "km"),
    "p90": ("output_p90", "ms"),
    "core_mean": ("output_core_mean", "ms"),
    "rmw": ("output_rmw", "km"),
    "mean": ("output_mean", "ms"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_DATA_ROOT / "index-files" / "observation_manifest_v6.csv",
    )
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--storms", nargs="+", default=["AL082025", "EP112025"])
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=DEFAULT_ENSEMBLE_SIZE,
        help=(
            "Number of reproducible diffusion members. Metrics are calculated "
            "per member before aggregation (default: 10)."
        ),
    )
    guidance = parser.add_mutually_exclusive_group()
    guidance.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override the checkpoint/config classifier-free guidance scale.",
    )
    guidance.add_argument(
        "--guidance-scales",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Run a paired guidance ablation. Multiple values are written below "
            "<output-root>/guidance_<value>/."
        ),
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Override the configured base seed used to derive member latents.",
    )
    parser.add_argument(
        "--summary-aggregation",
        choices=("median", "mean", "medoid"),
        default="median",
        help=(
            "Statistic assigned to the legacy output_* columns. The default "
            "median is the median of member-wise metrics."
        ),
    )
    parser.add_argument(
        "--member-quantiles",
        type=float,
        nargs="+",
        default=list(DEFAULT_MEMBER_QUANTILES),
        metavar="Q",
        help="Member-metric quantiles to retain (default: 0.1 0.9).",
    )
    parser.add_argument(
        "--save-member-fields",
        action="store_true",
        help=(
            "Save members, mean/median fields, medoid field, mask, and distance "
            "as one compressed NPZ per observation."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Debug: rows per storm."
    )
    args = parser.parse_args()
    if args.ensemble_size < 1:
        parser.error("--ensemble-size must be at least 1")
    guidance_values = (
        args.guidance_scales
        if args.guidance_scales is not None
        else ([] if args.guidance_scale is None else [args.guidance_scale])
    )
    if any(value < 0 or not math.isfinite(value) for value in guidance_values):
        parser.error("guidance scales must be finite and non-negative")
    if not args.member_quantiles:
        parser.error("--member-quantiles requires at least one value")
    if any(value < 0 or value > 1 for value in args.member_quantiles):
        parser.error("--member-quantiles values must be between 0 and 1")
    if len(set(args.member_quantiles)) != len(args.member_quantiles):
        parser.error("--member-quantiles values must be unique")
    return args


def _bounds(center_lat: float, center_lon: float) -> torch.Tensor:
    half_extent = GRID_SIZE * GRID_RESOLUTION_DEGREES / 2.0
    full = torch.tensor(
        [
            center_lon - half_extent,
            center_lon + half_extent,
            center_lat - half_extent,
            center_lat + half_extent,
        ],
        dtype=torch.float64,
    )
    crop_offset = (GRID_SIZE - CROP_SIZE) // 2 * GRID_RESOLUTION_DEGREES
    return full + torch.tensor(
        [crop_offset, -crop_offset, crop_offset, -crop_offset],
        dtype=torch.float64,
    )


def _center_crop(tensor: torch.Tensor) -> torch.Tensor:
    offset = (tensor.shape[-1] - CROP_SIZE) // 2
    return tensor[..., offset : offset + CROP_SIZE, offset : offset + CROP_SIZE]


def _physical_distance_km(
    bounds: torch.Tensor, center: tuple[float, float]
) -> torch.Tensor:
    normalized = _normalized_distance_to_center(
        bounds, (CROP_SIZE, CROP_SIZE), torch.tensor(center)
    ).squeeze(0)
    left, right, bottom, top = bounds.tolist()
    corners = [
        (bottom, left),
        (bottom, right),
        (top, left),
        (top, right),
    ]

    def haversine_km(point: tuple[float, float]) -> float:
        lat1, lon1 = map(math.radians, center)
        lat2, lon2 = map(math.radians, point)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        value = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        return 6371.0 * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))

    return normalized * max(haversine_km(corner) for corner in corners)


def _output_metrics(
    output: torch.Tensor, valid: torch.Tensor, distance_km: torch.Tensor
) -> dict[str, float]:
    """Calculate physical wind metrics over valid pixels only."""
    field = output.squeeze().cpu()
    valid = valid.squeeze().cpu().bool() & torch.isfinite(field)
    if int(valid.sum()) < math.ceil(0.05 * field.numel()):
        return {key: math.nan for key in METRIC_COLUMNS}
    values = field[valid]
    robust_peak_count = max(1, math.ceil(0.005 * values.numel()))
    robust_peak = float(
        torch.topk(values, robust_peak_count, sorted=False).values.mean()
    )
    core = field[valid & (distance_km <= 100.0)]
    bin_width = float(
        torch.nanmedian(torch.abs(torch.diff(distance_km[:, CROP_SIZE // 2])))
    )
    bin_width = bin_width if math.isfinite(bin_width) and bin_width > 0 else 3.0
    radii = torch.arange(0.0, float(distance_km[valid].max()) + bin_width, bin_width)
    radial_distance, radial_wind = [], []
    for radius in radii:
        annulus = valid & (distance_km >= radius) & (distance_km < radius + bin_width)
        if annulus.any():
            radial_distance.append(float(radius))
            radial_wind.append(float(field[annulus].mean()))
    r64 = [
        radius
        for radius, wind in zip(radial_distance, radial_wind)
        if wind >= R64_THRESHOLD_MS
    ]
    rmw = (
        radial_distance[radial_wind.index(max(radial_wind))]
        if radial_wind
        else math.nan
    )
    return {
        "msw": float(values.max()),
        "robust_peak": robust_peak,
        "r64": max(r64) if r64 else math.nan,
        "p90": float(torch.quantile(values, 0.9, interpolation="linear")),
        "core_mean": float(core.mean()) if core.numel() else math.nan,
        "rmw": rmw,
        "mean": float(values.mean()),
    }


def _member_latent_seed(
    base_seed: int, observation_id: str, ensemble_index: int
) -> int:
    """Mirror PixelDiffusion's stable sample-ID/member seed derivation."""
    latent_identifier = (
        observation_id
        if ensemble_index == 0
        else f"{observation_id}:ensemble={ensemble_index}"
    )
    digest = hashlib.sha256(
        f"{int(base_seed)}:{latent_identifier}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _metric_column(metric: str, statistic: str | None = None) -> str:
    base, unit = METRIC_COLUMNS[metric]
    return f"{base}_{unit}" if statistic is None else f"{base}_{unit}_{statistic}"


def _member_metric_column(metric: str) -> str:
    _, unit = METRIC_COLUMNS[metric]
    return f"{metric}_{unit}"


def _quantile_label(quantile: float) -> str:
    percentage = quantile * 100.0
    if math.isclose(percentage, round(percentage), abs_tol=1e-9):
        return f"p{int(round(percentage)):02d}"
    compact = f"{percentage:.4f}".rstrip("0").rstrip(".").replace(".", "_")
    return f"p{compact}"


def _finite_statistic(values: list[float], statistic: str) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan
    if statistic == "mean":
        return float(np.mean(finite))
    if statistic == "median":
        return float(np.median(finite))
    raise ValueError(f"Unsupported finite statistic: {statistic}")


def _select_medoid_member(
    member_fields: torch.Tensor, valid: torch.Tensor
) -> tuple[int, list[float], torch.Tensor]:
    """Select the complete member closest to the pixel-wise median field."""
    if member_fields.ndim != 3:
        raise ValueError("member_fields must have shape [member, height, width]")
    mask = valid.squeeze().cpu().bool()
    fields = member_fields.detach().float().cpu()
    common = mask & torch.isfinite(fields).all(dim=0)
    # torch.median chooses the lower middle value for even member counts;
    # linear q=0.5 matches the conventional scalar ensemble median.
    median_field = torch.quantile(fields, 0.5, dim=0, interpolation="linear")
    if not common.any():
        return 0, [math.nan] * fields.shape[0], median_field
    consensus = median_field[common]
    distances = torch.sqrt(
        torch.mean((fields[:, common] - consensus.unsqueeze(0)).square(), dim=1)
    )
    return (
        int(torch.argmin(distances)),
        [float(value) for value in distances],
        median_field,
    )


def _aggregate_member_metrics(
    member_metrics: list[dict[str, float]],
    medoid_index: int,
    mean_field_metrics: dict[str, float],
    median_field_metrics: dict[str, float],
    *,
    quantiles: list[float],
    summary_aggregation: str,
) -> dict[str, float]:
    """Create scalar summary columns without ever taking max(mean(field))."""
    summary: dict[str, float] = {}
    for metric in METRIC_COLUMNS:
        values = [row[metric] for row in member_metrics]
        member_mean = _finite_statistic(values, "mean")
        member_median = _finite_statistic(values, "median")
        medoid_value = float(member_metrics[medoid_index][metric])
        selected = {
            "mean": member_mean,
            "median": member_median,
            "medoid": medoid_value,
        }[summary_aggregation]
        summary[_metric_column(metric)] = selected
        summary[_metric_column(metric, "member_mean")] = member_mean
        summary[_metric_column(metric, "member_median")] = member_median
        summary[_metric_column(metric, "medoid")] = medoid_value
        summary[_metric_column(metric, "mean_field")] = float(
            mean_field_metrics[metric]
        )
        summary[_metric_column(metric, "median_field")] = float(
            median_field_metrics[metric]
        )
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        for quantile in quantiles:
            value = (
                float(np.quantile(finite, quantile, method="linear"))
                if finite.size
                else math.nan
            )
            label = f"member_{_quantile_label(quantile)}"
            summary[_metric_column(metric, label)] = value
    return summary


def _summarize_ensemble(
    member_fields: torch.Tensor,
    valid: torch.Tensor,
    distance_km: torch.Tensor,
    *,
    quantiles: list[float],
    summary_aggregation: str,
) -> tuple[
    dict[str, float],
    list[dict[str, float]],
    int,
    list[float],
    torch.Tensor,
    torch.Tensor,
]:
    """Return member-first scalar diagnostics and coherent field summaries."""
    if member_fields.ndim != 3 or member_fields.shape[0] < 1:
        raise ValueError(
            "member_fields must have non-empty [member, height, width] shape"
        )
    medoid_index, medoid_distances, median_field = _select_medoid_member(
        member_fields, valid
    )
    mean_field = torch.mean(member_fields.detach().float().cpu(), dim=0)
    member_metrics = [
        _output_metrics(field, valid, distance_km) for field in member_fields
    ]
    summary = _aggregate_member_metrics(
        member_metrics,
        medoid_index,
        _output_metrics(mean_field, valid, distance_km),
        _output_metrics(median_field, valid, distance_km),
        quantiles=quantiles,
        summary_aggregation=summary_aggregation,
    )
    finite_distances = [value for value in medoid_distances if math.isfinite(value)]
    summary.update(
        {
            "medoid_member_index": medoid_index,
            "medoid_consensus_rmse_ms": medoid_distances[medoid_index],
            "ensemble_consensus_rmse_mean_ms": (
                float(np.mean(finite_distances)) if finite_distances else math.nan
            ),
            "ensemble_pixel_std_mean_ms": float(
                torch.std(member_fields.float(), dim=0, correction=0)[
                    valid.squeeze().bool()
                ].mean()
            ),
        }
    )
    return (
        summary,
        member_metrics,
        medoid_index,
        medoid_distances,
        mean_field,
        median_field,
    )


def _atomic_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_source_metadata() -> dict[str, Any]:
    """Hash the exact tracked and untracked source inputs used for inference."""

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    commit_result = run("rev-parse", "HEAD")
    status_result = run("status", "--porcelain=v1")
    files_result = run("ls-files", "--cached", "--others", "--exclude-standard")
    status = status_result.stdout if status_result.returncode == 0 else ""
    tree_digest = hashlib.sha256()
    file_count = 0
    if files_result.returncode == 0:
        for relative in sorted(filter(None, files_result.stdout.splitlines())):
            source = ROOT / relative
            if not source.is_file() or source.suffix.lower() not in {
                ".py",
                ".yaml",
                ".yml",
                ".sh",
            }:
                continue
            tree_digest.update(relative.encode("utf-8") + b"\0")
            tree_digest.update(bytes.fromhex(_sha256(source)))
            file_count += 1
    return {
        "git_commit": (
            commit_result.stdout.strip() if commit_result.returncode == 0 else None
        ),
        "git_dirty": bool(status.strip()),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_tree_sha256": tree_digest.hexdigest(),
        "source_file_count": file_count,
    }


def _file_metadata(path: Path, *, sha256: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    payload: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if sha256:
        payload["sha256"] = _sha256(resolved)
    return payload


def _guidance_label(value: float) -> str:
    return format(value, ".8g").replace("-", "m").replace(".", "p")


def _safe_component(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
    return safe[:180] or "observation"


def _era5_fields(
    dataset: xr.Dataset, timestamp: pd.Timestamp
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    times = pd.DatetimeIndex(dataset["time"].values)
    times = times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
    index = _nearest_time_index(times, timestamp)
    if index is None:
        raise ValueError("ERA5 dataset has no time steps")
    selected_time = times[index]
    lat = _era5_coordinate_values(dataset, "latitude", "lat", index)
    lon = _fix_longitudes(_era5_coordinate_values(dataset, "longitude", "lon", index))
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    return {
        channel: (
            np.asarray(dataset[channel].isel(time=index).values, dtype=np.float32),
            lat,
            lon,
        )
        for channel in ERA5_CHANNELS
    }


def _prepare_sample(
    geo, era5_dataset: xr.Dataset, stats: dict
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if geo.center is None or geo.ibtracs_center is None:
        raise ValueError(f"{geo.observation_id} has no finite center")
    grid_lat, grid_lon = _make_grid(
        geo.center[0],
        geo.center[1],
        GRID_SIZE,
        GRID_RESOLUTION_DEGREES,
    )
    geo_channels = list(GEO_CHANNEL_SETS["common10"][geo.sensor])
    geo_fields = _load_geo_channels(geo, geo_channels)
    geo_regridded = [
        _regrid(*geo_fields[channel], grid_lat, grid_lon) for channel in geo_channels
    ]
    geo_array = np.stack([item[0] for item in geo_regridded]).astype(np.float32)
    geo_mask = np.logical_and.reduce(
        [item[1] & np.isfinite(item[0]) for item in geo_regridded]
    )

    era5_fields = _era5_fields(era5_dataset, geo.timestamp)
    era5_fields = _append_native_era5_derived_fields(era5_fields)
    era5_channels = list(era5_fields)
    era5_regridded = [
        _regrid_continuous(*era5_fields[channel], grid_lat, grid_lon)
        for channel in era5_channels
    ]
    era5_array = np.stack([item[0] for item in era5_regridded]).astype(np.float32)
    era5_mask = np.logical_and.reduce(
        [item[1] & np.isfinite(item[0]) for item in era5_regridded]
    )
    valid = torch.from_numpy(geo_mask & era5_mask).unsqueeze(0)
    geo_tensor = torch.from_numpy(geo_array)
    era5_tensor = torch.from_numpy(era5_array)
    wind_index = era5_channels.index("wind_speed_10m")
    era5_wind_physical = era5_tensor[wind_index : wind_index + 1].clone()

    geo_tensor = _normalize(
        geo_tensor,
        "geo",
        geo_channels,
        stats,
        normalization="robust-zscore",
        robust_clip=ROBUST_CLIP,
    )
    era5_tensor = _normalize(
        era5_tensor,
        "era5",
        [f"era5_{channel}" for channel in era5_channels],
        stats,
        normalization="robust-zscore",
        robust_clip=ROBUST_CLIP,
    )
    era5_wind = _normalize(
        era5_wind_physical,
        "sar",
        ["wind_speed"],
        stats,
        normalization="min-max",
    )
    condition = torch.cat([geo_tensor, era5_tensor], dim=0)
    condition = _center_crop(torch.nan_to_num(condition) * valid)

    era5_wind = _center_crop(torch.nan_to_num(era5_wind) * valid)
    era5_wind_physical = _center_crop(torch.nan_to_num(era5_wind_physical) * valid)
    valid = _center_crop(valid)
    bounds = _bounds(*geo.center)
    distance = _normalized_distance_to_center(
        bounds, (CROP_SIZE, CROP_SIZE), torch.tensor(geo.ibtracs_center)
    )
    solar = _solar_time_features(bounds, (CROP_SIZE, CROP_SIZE), geo.timestamp)
    condition = torch.cat([condition, distance, solar], dim=0)
    batch = {
        "condition": condition.unsqueeze(0),
        "condition_mask": valid.unsqueeze(0),
        "era5_wind_speed": era5_wind.unsqueeze(0),
        "era5_wind_speed_physical": era5_wind_physical.unsqueeze(0),
        "era5_wind_speed_mask": valid.unsqueeze(0),
    }
    return batch, _physical_distance_km(bounds, geo.ibtracs_center)


def main() -> None:
    args = parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    records = _read_manifest(args.manifest, args.data_root)
    by_id = {record.observation_id: record for record in records}
    era5_records = {
        storm: next(
            record
            for record in records
            if record.storm_id == storm and record.source_type == "era5"
        )
        for storm in args.storms
    }
    era5_by_storm = {}
    for storm, record in era5_records.items():
        with xr.open_dataset(
            record.path,
            group="rectilinear",
            engine="h5netcdf",
            decode_times=True,
        ) as source:
            era5_by_storm[storm] = source[list(ERA5_CHANNELS)].load()

    config = resolve_runtime_config(
        yaml.safe_load(args.config.read_text(encoding="utf-8"))
    )
    baseline_checkpoint_path = (
        config.get("model", {})
        .get("residual", {})
        .get("baseline", {})
        .get("checkpoint_path")
    )
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if args.sampling_seed is not None:
        model.validation_seed = int(args.sampling_seed)
    model.eval().to(args.device)

    configured_guidance = float(model.guidance_scale)
    guidance_values = (
        list(args.guidance_scales)
        if args.guidance_scales is not None
        else [
            (
                configured_guidance
                if args.guidance_scale is None
                else float(args.guidance_scale)
            )
        ]
    )
    if len(set(guidance_values)) != len(guidance_values):
        raise ValueError("guidance scales must be unique")
    multiple_guidance = len(guidance_values) > 1
    output_roots = {
        guidance: (
            args.output_root / f"guidance_{_guidance_label(guidance)}"
            if multiple_guidance
            else args.output_root
        )
        for guidance in guidance_values
    }
    for output_root in output_roots.values():
        output_root.mkdir(parents=True, exist_ok=True)

    target_stats = stats["channels"]["sar"]["wind_speed"]
    target_offset = float(target_stats["min"])
    target_scale = float(target_stats["max"]) - target_offset
    base_seed = int(model.validation_seed)
    checkpoint_metadata = _file_metadata(args.checkpoint)
    checkpoint_metadata.update(
        {
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
        }
    )
    common_metadata: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "started_utc": started_utc,
        "command": [str(value) for value in sys.argv],
        "storms": list(args.storms),
        "device": str(args.device),
        "model_class": model.__class__.__name__,
        "ensemble_kind": "stochastic_members_single_checkpoint",
        "ensemble_size": args.ensemble_size,
        "checkpoint": checkpoint_metadata,
        "deterministic_baseline_checkpoint": (
            _file_metadata(Path(baseline_checkpoint_path))
            if baseline_checkpoint_path
            else None
        ),
        "config": _file_metadata(args.config),
        "stats": _file_metadata(args.stats),
        "manifest": _file_metadata(args.manifest, sha256=False),
        "configured_guidance_scale": configured_guidance,
        "sampling": {
            "method": getattr(model, "sampling_method", None),
            "timesteps": getattr(model, "sampling_timesteps", None),
            "eta": getattr(model, "sampling_eta", None),
        },
        "seeding": {
            "base_seed": base_seed,
            "algorithm": "sha256(base_seed:observation_id[:ensemble=index])",
            "member_indices": list(range(args.ensemble_size)),
            "exact_seed_location": "per-member-metrics.csv:latent_seed",
            "paired_across_guidance_scales": True,
        },
        "aggregation": {
            "legacy_output_columns": args.summary_aggregation,
            "member_statistics": [
                "mean",
                "median",
                *[_quantile_label(value) for value in args.member_quantiles],
            ],
            "medoid": "minimum valid-pixel RMSE to pixel-wise member median",
            "mean_field_metrics_retained_for_legacy_comparison": True,
            "robust_peak": "mean of highest 0.5% of valid pixels per field",
        },
        "save_member_fields": bool(args.save_member_fields),
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "xarray": xr.__version__,
        },
        "include_test_in_train": config.get("data", {}).get("include_test_in_train"),
        "source": _git_source_metadata(),
        "outputs": {
            "summary": "<storm>/inference-summary.csv",
            "member_metrics": "<storm>/per-member-metrics.csv",
            "member_fields_manifest": (
                "<storm>/member-fields-manifest.csv"
                if args.save_member_fields
                else None
            ),
            "member_fields": (
                "<storm>/member-fields/<observation>.npz"
                if args.save_member_fields
                else None
            ),
        },
    }
    run_metadata = {}
    for guidance, output_root in output_roots.items():
        payload = {
            **common_metadata,
            "guidance_scale": guidance,
            "output_root": str(output_root.resolve()),
        }
        run_metadata[guidance] = payload
        _atomic_json(payload, output_root / "run-metadata.json")
    if multiple_guidance:
        _atomic_json(
            {
                "schema_version": 1,
                "started_utc": started_utc,
                "guidance_scales": guidance_values,
                "paired_latents": True,
                "runs": [
                    {
                        "guidance_scale": guidance,
                        "metadata": str(
                            (output_roots[guidance] / "run-metadata.json").relative_to(
                                args.output_root
                            )
                        ),
                    }
                    for guidance in guidance_values
                ],
            },
            args.output_root / "guidance-sweep-metadata.json",
        )

    record_counts = {guidance: {} for guidance in guidance_values}
    member_counts = {guidance: 0 for guidance in guidance_values}
    with torch.inference_mode():
        for storm in args.storms:
            reference_table = pd.read_csv(
                args.reference_root / storm / "inference-summary.csv"
            )
            if args.limit is not None:
                reference_table = reference_table.head(args.limit).copy()
            rows = list(reference_table.itertuples(index=False))
            states: dict[float, dict[str, Any]] = {}
            for guidance in guidance_values:
                storm_dir = output_roots[guidance] / storm
                storm_dir.mkdir(parents=True, exist_ok=True)
                states[guidance] = {
                    "summary_rows": [],
                    "member_rows": [],
                    "field_rows": [],
                    "local_paths": [],
                    "storm_dir": storm_dir,
                }

            starts = range(0, len(rows), args.batch_size)
            iterator = tqdm(
                starts,
                total=math.ceil(len(rows) / args.batch_size),
                desc=storm,
            )
            for start in iterator:
                chunk = rows[start : start + args.batch_size]
                prepared = [
                    _prepare_sample(
                        by_id[row.observation_id], era5_by_storm[storm], stats
                    )
                    for row in chunk
                ]
                batch = {
                    key: torch.cat([sample[0][key] for sample in prepared], dim=0).to(
                        args.device
                    )
                    for key in prepared[0][0]
                }
                batch["sample_id"] = [row.observation_id for row in chunk]
                batch["target"] = torch.zeros_like(batch["era5_wind_speed"])
                batch["target_mask"] = batch["condition_mask"]
                batch["target_norm_offset"] = torch.full(
                    (len(chunk), 1), target_offset, device=args.device
                )
                batch["target_norm_scale"] = torch.full(
                    (len(chunk), 1), target_scale, device=args.device
                )

                for guidance in guidance_values:
                    model.guidance_scale = float(guidance)
                    physical_members = []
                    for ensemble_index in range(args.ensemble_size):
                        _, normalized = model._predict_batch(
                            batch,
                            start // args.batch_size,
                            ensemble_index=ensemble_index,
                        )
                        physical_members.append(
                            (normalized * target_scale + target_offset)
                            .detach()
                            .float()
                            .cpu()
                        )
                    members = torch.stack(physical_members, dim=0)
                    if members.shape[2] != 1:
                        raise ValueError(
                            "Storm inference expects one generated wind channel; "
                            f"received {members.shape[2]}"
                        )
                    state = states[guidance]
                    for index, (row, (_, distance_km)) in enumerate(
                        zip(chunk, prepared)
                    ):
                        observation_id = str(row.observation_id)
                        member_fields = members[:, index, 0]
                        valid = batch["condition_mask"][index].detach().cpu()
                        (
                            summary,
                            metrics_by_member,
                            medoid_index,
                            medoid_distances,
                            mean_field,
                            median_field,
                        ) = _summarize_ensemble(
                            member_fields,
                            valid,
                            distance_km,
                            quantiles=list(args.member_quantiles),
                            summary_aggregation=args.summary_aggregation,
                        )
                        seeds = [
                            _member_latent_seed(base_seed, observation_id, member_index)
                            for member_index in range(args.ensemble_size)
                        ]
                        summary.update(
                            {
                                "ensemble_size": args.ensemble_size,
                                "ensemble_sampling_seed": base_seed,
                                "guidance_scale": guidance,
                                "summary_aggregation": args.summary_aggregation,
                            }
                        )
                        state["summary_rows"].append(summary)
                        local_path = str(by_id[observation_id].path)
                        state["local_paths"].append(local_path)

                        field_relative_path = None
                        if args.save_member_fields:
                            filename = f"{_safe_component(observation_id)}.npz"
                            field_path = state["storm_dir"] / "member-fields" / filename
                            mask = valid.squeeze().bool()
                            masked_members = torch.where(
                                mask.unsqueeze(0),
                                member_fields,
                                torch.full_like(member_fields, torch.nan),
                            )
                            masked_mean = torch.where(
                                mask, mean_field, torch.full_like(mean_field, torch.nan)
                            )
                            masked_median = torch.where(
                                mask,
                                median_field,
                                torch.full_like(median_field, torch.nan),
                            )
                            _atomic_npz(
                                field_path,
                                observation_id=np.asarray(observation_id),
                                guidance_scale=np.asarray(guidance, dtype=np.float32),
                                base_seed=np.asarray(base_seed, dtype=np.int64),
                                member_indices=np.arange(
                                    args.ensemble_size, dtype=np.int32
                                ),
                                latent_seeds=np.asarray(seeds, dtype=np.int64),
                                member_fields_ms=masked_members.numpy().astype(
                                    np.float32, copy=False
                                ),
                                mean_field_ms=masked_mean.numpy().astype(
                                    np.float32, copy=False
                                ),
                                median_field_ms=masked_median.numpy().astype(
                                    np.float32, copy=False
                                ),
                                medoid_field_ms=masked_members[medoid_index]
                                .numpy()
                                .astype(np.float32, copy=False),
                                medoid_member_index=np.asarray(
                                    medoid_index, dtype=np.int32
                                ),
                                valid_mask=mask.numpy().astype(np.uint8, copy=False),
                                distance_km=distance_km.numpy().astype(
                                    np.float32, copy=False
                                ),
                            )
                            field_relative_path = str(
                                field_path.relative_to(output_roots[guidance])
                            )
                            state["field_rows"].append(
                                {
                                    "storm_id": storm,
                                    "observation_id": observation_id,
                                    "observation_timestamp": getattr(
                                        row, "observation_timestamp", None
                                    ),
                                    "guidance_scale": guidance,
                                    "ensemble_size": args.ensemble_size,
                                    "medoid_member_index": medoid_index,
                                    "npz_path": field_relative_path,
                                    "member_array": "member_fields_ms",
                                    "member_shape": "x".join(
                                        str(value) for value in member_fields.shape
                                    ),
                                    "dtype": "float32",
                                }
                            )

                        for member_index, metrics in enumerate(metrics_by_member):
                            member_row = {
                                "storm_id": storm,
                                "observation_id": observation_id,
                                "observation_timestamp": getattr(
                                    row, "observation_timestamp", None
                                ),
                                "sensor": getattr(row, "sensor", None),
                                "original_input_path": local_path,
                                "ibtracs_msw_ms": getattr(
                                    row, "ibtracs_msw_ms", math.nan
                                ),
                                "ibtracs_category": getattr(
                                    row, "ibtracs_category", None
                                ),
                                "ibtracs_r64_mean_km": getattr(
                                    row, "ibtracs_r64_mean_km", math.nan
                                ),
                                "guidance_scale": guidance,
                                "ensemble_size": args.ensemble_size,
                                "ensemble_member_index": member_index,
                                "latent_seed": seeds[member_index],
                                "is_medoid_member": member_index == medoid_index,
                                "consensus_rmse_ms": medoid_distances[member_index],
                                "member_fields_npz": field_relative_path,
                            }
                            member_row.update(
                                {
                                    _member_metric_column(metric): value
                                    for metric, value in metrics.items()
                                }
                            )
                            state["member_rows"].append(member_row)

            for guidance in guidance_values:
                state = states[guidance]
                output_table = reference_table.copy()
                output_table["original_input_path"] = state["local_paths"]
                if state["summary_rows"]:
                    for column in state["summary_rows"][0]:
                        output_table[column] = [
                            summary[column] for summary in state["summary_rows"]
                        ]
                storm_dir = state["storm_dir"]
                summary_path = storm_dir / "inference-summary.csv"
                member_path = storm_dir / "per-member-metrics.csv"
                _atomic_csv(output_table, summary_path)
                _atomic_csv(pd.DataFrame(state["member_rows"]), member_path)
                if args.save_member_fields:
                    _atomic_csv(
                        pd.DataFrame(state["field_rows"]),
                        storm_dir / "member-fields-manifest.csv",
                    )
                record_counts[guidance][storm] = len(output_table)
                member_counts[guidance] += len(state["member_rows"])
                try:
                    display_path = summary_path.relative_to(ROOT)
                except ValueError:
                    display_path = summary_path
                print(f"Wrote {display_path} ({len(output_table)} rows)")

    completed_utc = datetime.now(timezone.utc).isoformat()
    for guidance, output_root in output_roots.items():
        payload = run_metadata[guidance]
        payload.update(
            {
                "status": "complete",
                "completed_utc": completed_utc,
                "records_by_storm": record_counts[guidance],
                "member_records": member_counts[guidance],
            }
        )
        _atomic_json(payload, output_root / "run-metadata.json")
    if multiple_guidance:
        sweep_path = args.output_root / "guidance-sweep-metadata.json"
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        sweep.update({"status": "complete", "completed_utc": completed_utc})
        _atomic_json(sweep, sweep_path)


if __name__ == "__main__":
    main()
