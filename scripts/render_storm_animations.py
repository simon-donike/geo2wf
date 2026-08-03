"""Render all explorer CMI_C15 inputs and Model B wind fields as GIFs.

The Model B fields are regenerated from the saved deterministic checkpoint because
its published inference directory intentionally stores only tabular summaries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import xarray as xr
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export_geostat_images import GEOSTAT_SCALE_MAX_K, GEOSTAT_SCALE_MIN_K
from run_storm_unet_inference import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_ROOT,
    DEFAULT_REFERENCE_ROOT,
    DEFAULT_STATS,
    ERA5_CHANNELS,
    _prepare_sample,
)
from scripts.export_geo_sar_geotiffs import _read_manifest
from geo2wf.models.deterministic_residual import ERA5ResidualRegressor


OUTPUT_DIR = ROOT / "docs" / "assets" / "images"
STORMS = ("AL082025", "EP112025")
FRAME_STEP = 2
FRAME_DURATION_MS = 40  # 180 selected frames / 25 fps = 7.2 seconds.
SIZE = 256


def _caption(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (SIZE, SIZE + 22), "white")
    canvas.paste(image, (0, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), text, fill=(25, 25, 25), font=ImageFont.load_default())
    return canvas


def _geo_frame(bundle: dict, label: str) -> np.ndarray:
    index = bundle["input_channels"].index("CMI_C15")
    field = bundle["input"][index].float().numpy()
    valid = bundle["input_mask"].bool().numpy() & np.isfinite(field)
    value = np.clip(
        (field - GEOSTAT_SCALE_MIN_K) / (GEOSTAT_SCALE_MAX_K - GEOSTAT_SCALE_MIN_K),
        0,
        1,
    )
    rgb = np.empty((*field.shape, 3), dtype=np.uint8)
    # Fixed explorer ramp: cold clouds are white and warmer pixels blue.
    rgb[:] = np.array([255, 255, 255], dtype=np.uint8)
    rgb[valid] = (
        np.array([255, 255, 255])
        + value[valid, None] * (np.array([22, 82, 180]) - np.array([255, 255, 255]))
    ).astype(np.uint8)
    return np.asarray(
        _caption(
            Image.fromarray(rgb).resize((SIZE, SIZE), Image.Resampling.LANCZOS), label
        )
    )


def _prediction_frame(
    field: np.ndarray, valid: np.ndarray, label: str, vmin: float, vmax: float
) -> np.ndarray:
    value = np.clip((field - vmin) / (vmax - vmin), 0, 1)
    red, yellow, green = (
        np.array([215, 48, 39]),
        np.array([255, 235, 59]),
        np.array([26, 150, 65]),
    )
    rgb = np.full((*field.shape, 3), 255, dtype=np.uint8)
    lower = green + (value[..., None] * 2) * (yellow - green)
    upper = yellow + ((value[..., None] - 0.5) * 2) * (red - yellow)
    ramp = np.where((value <= 0.5)[..., None], lower, upper).astype(np.uint8)
    rgb[valid] = ramp[valid]
    return np.asarray(
        _caption(
            Image.fromarray(rgb).resize((SIZE, SIZE), Image.Resampling.LANCZOS), label
        )
    )


def main() -> None:
    import json

    stats = json.loads(DEFAULT_STATS.read_text())
    records = _read_manifest(
        DEFAULT_DATA_ROOT / "index-files" / "observation_manifest_v6.csv",
        DEFAULT_DATA_ROOT,
    )
    by_id = {record.observation_id: record for record in records}
    era5 = {}
    for storm in STORMS:
        record = next(
            item
            for item in records
            if item.storm_id == storm and item.source_type == "era5"
        )
        with xr.open_dataset(
            record.path, group="rectilinear", engine="h5netcdf", decode_times=True
        ) as source:
            era5[storm] = source[list(ERA5_CHANNELS)].load()

    model = (
        ERA5ResidualRegressor.load_from_checkpoint(
            DEFAULT_CHECKPOINT, map_location="cpu"
        )
        .eval()
        .cuda()
    )
    geo_frames, fields, masks, labels = [], [], [], []
    with torch.inference_mode():
        for storm in STORMS:
            table = pd.read_csv(
                DEFAULT_REFERENCE_ROOT / storm / "inference-summary.csv"
            ).sort_values("observation_timestamp")
            for row in table.itertuples(index=False):
                bundle = torch.load(
                    DEFAULT_REFERENCE_ROOT / storm / row.inference_path,
                    map_location="cpu",
                    weights_only=False,
                )
                labels.append(
                    f"{storm}  {pd.Timestamp(row.observation_timestamp):%Y-%m-%d %H:%M UTC}"
                )
                geo_frames.append(_geo_frame(bundle, labels[-1]))
                batch, _ = _prepare_sample(
                    by_id[row.observation_id], era5[storm], stats
                )
                valid = batch["condition_mask"].squeeze().numpy().astype(bool)
                prediction = (
                    model.predict_physical(
                        {key: value.cuda() for key, value in batch.items()}
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                fields.append(prediction)
                masks.append(valid & np.isfinite(prediction))

    # Centered 3-frame weighted moving average softens frame-to-frame noise while
    # retaining the temporal evolution and leaves the first/last frame anchored.
    smoothed = []
    for index, field in enumerate(fields):
        start, end = max(0, index - 1), min(len(fields), index + 2)
        weights = np.array(
            [1.0 if item == index else 0.5 for item in range(start, end)],
            dtype=np.float32,
        )
        stacked = np.stack(fields[start:end])
        valid = np.stack(masks[start:end])
        weighted = np.where(valid, stacked * weights[:, None, None], 0.0)
        total = (valid * weights[:, None, None]).sum(axis=0)
        smoothed.append(
            np.divide(
                weighted.sum(axis=0), total, out=np.zeros_like(field), where=total > 0
            )
        )
    valid_values = np.concatenate([field[mask] for field, mask in zip(smoothed, masks)])
    vmin, vmax = np.percentile(valid_values, [0.5, 99.5])
    geo_frames = geo_frames[::FRAME_STEP]
    prediction_frames = [
        _prediction_frame(field, mask, label, float(vmin), float(vmax))
        for field, mask, label in zip(
            smoothed[::FRAME_STEP], masks[::FRAME_STEP], labels[::FRAME_STEP]
        )
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        OUTPUT_DIR / "all-cmi-c15.gif",
        geo_frames,
        format="GIF",
        duration=FRAME_DURATION_MS,
        loop=0,
        palettesize=256,
        subrectangles=True,
    )
    imageio.mimsave(
        OUTPUT_DIR / "all-model-b-predictions.gif",
        prediction_frames,
        format="GIF",
        duration=FRAME_DURATION_MS,
        loop=0,
        palettesize=256,
        subrectangles=True,
    )
    print(
        f"Rendered {len(geo_frames)} frames (7.2 s); Model B scale: {vmin:.2f}–{vmax:.2f} m/s"
    )


if __name__ == "__main__":
    main()
