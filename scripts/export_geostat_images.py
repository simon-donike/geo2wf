"""Create fixed-scale CMI_C15 visualizations for the web explorer."""

import re
from pathlib import Path

import numpy as np
from PIL import Image


GEOSTAT_SCALE_MIN_K = 190.0
GEOSTAT_SCALE_MAX_K = 292.0
GEOSTAT_IMAGE_SIZE = 256


def export_geostat_image(observation_id, bundle, output_dir: Path):
    channel_index = bundle["input_channels"].index("CMI_C15")
    channel = bundle["input"][channel_index].float().numpy()
    valid = bundle["input_mask"].bool().numpy() & np.isfinite(channel)
    scaled = np.clip(
        (channel - GEOSTAT_SCALE_MIN_K) / (GEOSTAT_SCALE_MAX_K - GEOSTAT_SCALE_MIN_K),
        0,
        1,
    )
    scaled = np.nan_to_num(scaled)

    # Requested fixed ramp: white at the low-temperature end, blue at the high end.
    white = np.array([255, 255, 255], dtype=np.float32)
    blue = np.array([22, 82, 180], dtype=np.float32)
    rgb = white + scaled[..., None] * (blue - white)
    rgb[~valid] = white

    image = Image.fromarray(rgb.astype(np.uint8)).resize(
        (GEOSTAT_IMAGE_SIZE, GEOSTAT_IMAGE_SIZE),
        resample=Image.Resampling.LANCZOS,
    )
    filename = re.sub(r"[^a-zA-Z0-9_-]+", "_", observation_id) + ".webp"
    image.save(output_dir / filename, "WEBP", lossless=True, method=6)
    lat, lon = bundle["grid_lat"].numpy(), bundle["grid_lon"].numpy()
    return {
        "image": f"geo/{filename}",
        "bounds": [
            [round(float(np.nanmin(lat)), 3), round(float(np.nanmin(lon)), 3)],
            [round(float(np.nanmax(lat)), 3), round(float(np.nanmax(lon)), 3)],
        ],
        "kind": "Geostationary",
        "channel": "CMI_C15",
        "size": GEOSTAT_IMAGE_SIZE,
    }
