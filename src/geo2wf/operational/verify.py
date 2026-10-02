"""Finite real-data acceptance probe. Retains only metrics and diagnostics."""

from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
import torch

from .common import file_hash, iso, utc, write_json
from .models import MANIFEST, Models, checkpoint, prepare, resolve_file
from .satellite import BANDS, acquire
from .tracks import historical_center


def preprocessing_parity(array, valid, center, at, stats):
    """Exercise the real training loader using GDAL's in-memory filesystem."""
    from geo2wf.data.datasets.paired_geotiff import PairedImageDataset

    lat, lon = center["lat"], center["lon"]
    transform = from_origin(lon - 128 * 0.027, lat + 128 * 0.027, 0.027, 0.027)
    with ExitStack() as stack:
        root = Path(
            stack.enter_context(TemporaryDirectory(prefix="stormsense-parity-"))
        )
        (root / "test").mkdir()
        paths = {}
        for name, values in [
            ("geo", array),
            ("sar", np.zeros((1, 256, 256), dtype="float32")),
        ]:
            memory = stack.enter_context(MemoryFile())
            with memory.open(
                driver="GTiff",
                height=256,
                width=256,
                count=len(values),
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
                nodata=np.nan,
            ) as ds:
                ds.write(values)
            paths[name] = memory.name
        (root / "stats.json").write_text(json.dumps(stats))
        pd.DataFrame(
            [
                dict(
                    sample_id="parity",
                    storm_id="verification",
                    condition_path=paths["geo"],
                    target_path=paths["sar"],
                    condition_channels=json.dumps(BANDS),
                    target_channels='["wind_speed"]',
                    condition_timestamp=at,
                    dt_minutes=0,
                    ibtracs_center_lat=lat,
                    ibtracs_center_lon=lon,
                )
            ]
        ).to_csv(root / "test/manifest.csv", index=False)
        reference = PairedImageDataset(
            root,
            "test",
            center_crop_size=(192, 192),
            use_era5=False,
            normalization="robust-zscore",
            target_normalization="min-max",
            robust_clip=4,
        )[0]
        batch = prepare(array, valid, lat, lon, at, stats)
        errors = {}
        for key, value in batch.items():
            torch.testing.assert_close(value[0], reference[key], rtol=1e-5, atol=2e-5)
            errors[key] = float((value[0].float() - reference[key].float()).abs().max())
        return errors


def verify(store, model_root, device=None):
    models = Models(model_root, device)
    # Select actual recorded centers, one in each requested viewing regime.
    selections = {}
    for storm in reversed(store.storms()):
        for fix in reversed(storm.get("track", [])):
            if utc(fix["time"]) > utc():
                continue
            regime = (
                "GOES-East"
                if storm["id"].startswith("AL")
                else ("Central Pacific" if fix["lon"] < -140 else "GOES-West")
            )
            if regime not in selections:
                selections[regime] = (storm, fix)
    if len(selections) != 3:
        raise ValueError(
            "Discover tracks covering Atlantic, eastern and central Pacific first"
        )
    rows = []
    for regime, (storm, fix) in sorted(selections.items()):
        started = time.monotonic()
        center = historical_center(storm["track"], fix["time"])
        array, valid, source = acquire(fix["time"], center)
        parity = preprocessing_parity(array, valid, center, source["end"], models.stats)
        inference_started = time.monotonic()
        metrics = models.infer(
            array, valid, center["lat"], center["lon"], source["end"]
        )
        rows.append(
            dict(
                regime=regime,
                storm_id=storm["id"],
                time=fix["time"],
                center=center,
                imagery=source,
                metrics=metrics,
                channel_kelvin={
                    band: {
                        "min": float(np.nanmin(array[i])),
                        "median": float(np.nanmedian(array[i])),
                        "max": float(np.nanmax(array[i])),
                    }
                    for i, band in enumerate(BANDS)
                },
                preprocessing_max_abs_error=parity,
                inference_seconds=time.monotonic() - inference_started,
                total_seconds=time.monotonic() - started,
            )
        )
    return {
        "schema_version": 1,
        "generated_at": iso(),
        "device": models.device,
        "checkpoint_hashes": {
            role: file_hash(checkpoint(role, model_root)) for role in MANIFEST["models"]
        },
        "samples": rows,
        "imagery_retained": False,
        "note": "Observed byte counts and runtimes for these probes; throughput varies with network and shared compute.",
    }
