"""Export StormSense's fixed, real method-page example (no live data writes).

Run from the repository root: .venv/bin/python scripts/export_method_example.py
The test raster is reprojected onto the operational storm-centered grid before
the shared operational preprocessing is applied to both pinned checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geo2wf.data.joint_intensity import _interpolate_ibtracs_wind, _load_ibtracs_tracks
from geo2wf.historical.training import ScalarAdapter
from geo2wf.models.bottleneck_unet_mlp import BottleneckUNetMLPRegressor
from geo2wf.models.bottleneck_unet_mlp.module import _masked_huber_loss
from geo2wf.operational.models import prepare
from geo2wf.operational.satellite import BANDS

PALETTE = ["#13233c", "#245e86", "#32a6ab", "#a3d9b1", "#efce7a", "#eb895c", "#b54961"]


def digest(path):
    with Path(path).open("rb") as f:
        return (
            hashlib.file_digest(f, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else _digest_stream(f)
        )


def _digest_stream(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def pinned(item, key):
    candidates = [
        ROOT / item.get(f"local_{key}", item[key]),
        ROOT / "downloads/models" / item[key],
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(candidates)
    expected = item["sha256" if key == "checkpoint" else f"{key}_sha256"]
    if digest(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return path


def colorize(values, low, high, palette):
    colors = np.array([[int(c[i : i + 2], 16) for i in (1, 3, 5)] for c in palette])
    v = np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0, 1)
    return np.stack(
        [np.interp(v, np.linspace(0, 1, len(colors)), colors[:, i]) for i in range(3)],
        axis=-1,
    ).astype(np.uint8)


def select_example(root, ibtracs):
    tracks = _load_ibtracs_tracks(ibtracs, {"AL092022"})
    positions = pd.read_csv(
        ibtracs, usecols=["USA_ATCF_ID", "ISO_TIME", "LAT", "LON"], low_memory=False
    )
    positions = positions[positions.USA_ATCF_ID.str.strip() == "AL092022"].copy()
    positions["time"] = pd.to_datetime(positions.ISO_TIME, utc=True)
    positions = positions.sort_values("time").drop_duplicates("time")
    times = positions.time.astype("int64").to_numpy()
    latitudes = pd.to_numeric(positions.LAT).to_numpy()
    longitudes = pd.to_numeric(positions.LON).to_numpy()
    with (root / "test/manifest.csv").open() as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["storm_id"] == "AL092022" and r["geo_sensor"] == "ABI"
        ]
    eligible = []
    for row in rows:
        at = pd.Timestamp(row["geo_timestamp"])
        label = _interpolate_ibtracs_wind(tracks["AL092022"], at, max_bracket_hours=3)
        if label is None:
            continue
        lat = float(np.interp(at.value, times, latitudes))
        lon = float(np.interp(at.value, times, longitudes))
        transform = from_origin(lon - 128 * 0.027, lat + 128 * 0.027, 0.027, 0.027)
        with rasterio.open(root / row["geo_path"]) as ds:
            if list(ds.descriptions) != list(BANDS):
                raise ValueError("Unexpected satellite channel order")
            source = ds.read(masked=True).filled(np.nan)
            array = np.full((10, 256, 256), np.nan, dtype=np.float32)
            reproject(
                source,
                array,
                src_transform=ds.transform,
                src_crs=ds.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
        valid = np.isfinite(array).all(axis=0)
        fraction = float(valid[32:224, 32:224].mean())
        if fraction < 0.9 or not valid[126:130, 126:130].all():
            continue
        eligible.append(
            (
                label["target_wind_ms"],
                at,
                row["sample_id"],
                row,
                array,
                valid,
                lat,
                lon,
                fraction,
            )
        )
    if not eligible:
        raise ValueError(
            "No eligible, storm-centered Ian crop; do not substitute fabricated imagery"
        )
    chosen = min(eligible, key=lambda x: (-x[0], x[1], x[2]))
    track = [
        {"time": t.isoformat(), "lat": float(lat), "lon": float(lon)}
        for t, lat, lon in zip(positions.time, latitudes, longitudes)
        if chosen[1] - pd.Timedelta(hours=72) <= t <= chosen[1]
    ]
    track.append({"time": chosen[1].isoformat(), "lat": chosen[6], "lon": chosen[7]})
    return chosen, track


def scalars(intensity, structure):
    radii = structure[0].cpu().tolist()
    result = dict(
        zip(
            ["vmax_ms", "rmw_km", "r34_km", "r50_km", "r64_km"],
            [float(intensity[0]), *radii[1:]],
        )
    )
    if not all(math.isfinite(v) and v >= 0 for v in result.values()):
        raise ValueError("Invalid scalar output")
    return result


def export_training_pair(output, root, row, field, condition_mask, lat, lon, delta):
    """Align the matched SAR target to the exact inference grid; no model update."""
    source_path = root / row["sar_path"]
    transform = from_origin(lon - 96 * 0.027, lat + 96 * 0.027, 0.027, 0.027)
    sar = np.full((192, 192), np.nan, dtype=np.float32)
    with rasterio.open(source_path) as ds:
        assert ds.count == 1 and ds.descriptions == ("wind_speed",)
        source = ds.read(1, masked=True).filled(np.nan)
        source[~np.isfinite(source)] = np.nan
        reproject(
            source,
            sar,
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
        # Independent pixel-center lookup verifies orientation, location and mask.
        yy, xx = np.indices(sar.shape)
        xs, ys = rasterio.transform.xy(transform, yy.ravel(), xx.ravel())
        sr, sc = rasterio.transform.rowcol(ds.transform, xs, ys)
        sr, sc = np.asarray(sr), np.asarray(sc)
        inside = (sr >= 0) & (sr < ds.height) & (sc >= 0) & (sc < ds.width)
        check = np.full(sar.size, np.nan, dtype=np.float32)
        check[inside] = source[sr[inside], sc[inside]]
        np.testing.assert_allclose(sar, check.reshape(sar.shape), equal_nan=True)
    mask = np.isfinite(sar) & condition_mask.astype(bool)
    assert mask.any() and (sar[mask] >= 0).all()
    predicted = torch.tensor(field, requires_grad=True)
    target = torch.from_numpy(np.nan_to_num(sar, nan=0.0))
    loss = _masked_huber_loss(predicted, target, torch.from_numpy(mask), delta)
    gradient = torch.autograd.grad(loss, predicted)[0].numpy()
    expected = np.where(
        mask, np.clip(field - np.nan_to_num(sar), -delta, delta) / mask.sum(), 0
    )
    np.testing.assert_allclose(gradient, expected, rtol=1e-6, atol=1e-10)
    assert not gradient[~mask].any()
    high = math.ceil(max(float(field.max()), float(sar[mask].max())) / 10) * 10
    alpha = mask.astype(np.uint8) * 255
    Image.fromarray(np.dstack([colorize(sar, 0, high, PALETTE), alpha])).save(
        output / "sar-target.webp", lossless=True
    )
    Image.fromarray(colorize(field, 0, high, PALETTE)).save(
        output / "training-prediction.webp", lossless=True
    )
    limit = delta / int(mask.sum())
    gradient_palette = ["#63d8d2", "#182437", "#f1ad80"]
    Image.fromarray(
        np.dstack([colorize(gradient, -limit, limit, gradient_palette), alpha])
    ).save(output / "field-gradient.webp", lossless=True)
    np.save(output / "sar-target.npy", sar, allow_pickle=False)
    np.save(output / "field-loss-mask.npy", mask, allow_pickle=False)
    np.save(output / "field-gradient.npy", gradient, allow_pickle=False)
    files = [
        "sar-target.webp",
        "training-prediction.webp",
        "field-gradient.webp",
        "sar-target.npy",
        "field-loss-mask.npy",
        "field-gradient.npy",
    ]
    return {
        "sar_image": "/method/sar-target.webp",
        "prediction_image": "/method/training-prediction.webp",
        "gradient_image": "/method/field-gradient.webp",
        "sar_values": "/method/sar-target.npy",
        "mask_values": "/method/field-loss-mask.npy",
        "gradient_values": "/method/field-gradient.npy",
        "sar_time": row["sar_timestamp"],
        "sar_sensor": row["sar_sensor"],
        "source": row["sar_path"],
        "source_sha256": digest(source_path),
        "sha256": {name: digest(output / name) for name in files},
        "min_ms": 0,
        "max_ms": high,
        "palette": PALETTE,
        "valid_pixels": int(mask.sum()),
        "total_pixels": int(mask.size),
        "field_loss": float(loss.detach()),
        "huber_delta_ms": delta,
        "gradient_display_limit": limit,
        "gradient_palette": gradient_palette,
        "alignment": "Nearest-neighbor SAR reprojection to the example's 192x192 EPSG:4326 grid. Pixel-center lookup independently verified. North up; exact same extent as predicted field.",
        "mask": "Finite SAR target intersected with satellite condition mask. Transparent comparison/gradient pixels are excluded from field loss.",
        "gradient": "Actual autograd derivative of the masked mean physical-space Huber field loss with respect to predicted wind speed: clip(prediction-target, -delta, delta) / valid_pixel_count; zero outside mask. Display uses symmetric +/- delta / valid_pixel_count.",
        "usage": "Held-out test observation used to illustrate the training objective, not to update model weights.",
    }


def export(output):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    root = ROOT / "data/geotiff/geo_sar_10bands_era5_v2_pmw"
    ibtracs = ROOT / "data/IBTrACs/ibtracs.ALL.list.v04r01.csv"
    manifests = {
        key: json.loads((ROOT / path).read_text())["models"]["nowcast"]
        for key, path in {
            "joint": "src/geo2wf/operational/models.json",
            "encoder": "src/geo2wf/operational/models-finetuned.json",
        }.items()
    }
    checkpoints = {key: pinned(item, "checkpoint") for key, item in manifests.items()}
    for item in manifests.values():
        pinned(item, "config")
    stats_path = pinned(manifests["joint"], "stats")
    assert digest(pinned(manifests["encoder"], "stats")) == digest(stats_path)
    (reference_wind, at, sample_id, row, array, valid, lat, lon, fraction), track = (
        select_example(root, ibtracs)
    )
    batch = prepare(
        array, valid, lat, lon, at.isoformat(), json.loads(stats_path.read_text())
    )
    assert batch["condition"].shape == (1, 14, 192, 192)
    joint = BottleneckUNetMLPRegressor.load_from_checkpoint(
        checkpoints["joint"], map_location="cpu"
    ).eval()
    encoder = ScalarAdapter.load_from_checkpoint(
        checkpoints["encoder"],
        map_location="cpu",
        architecture=manifests["encoder"]["architecture"],
    ).eval()
    with torch.inference_mode():
        prediction = joint.predict_joint(batch)
        intensity, structure = encoder(batch)
        # Independent forward + physical conversion verifies the published field.
        raw = joint.predict_normalized(batch)
        expected = (
            raw.reconstruction_normalized * batch["target_norm_scale"]
            + batch["target_norm_offset"]
        )
        torch.testing.assert_close(prediction.central_physical, expected)
        torch.testing.assert_close(
            prediction.intensity_prediction_ms, raw.intensity_prediction_ms
        )
        torch.testing.assert_close(
            prediction.structure_prediction_km, raw.structure_prediction_km
        )
        encoder_output = encoder.model(
            torch.cat(
                [batch["condition"] * batch["condition_mask"], batch["condition_mask"]],
                dim=1,
            )
        )
        torch.testing.assert_close(intensity, encoder_output.intensity_prediction_ms)
        torch.testing.assert_close(structure, encoder_output.structure_prediction_km)
    field = prediction.central_physical[0, 0].numpy()
    assert field.shape == (192, 192) and np.isfinite(field).all() and (field >= 0).all()
    output.mkdir(parents=True, exist_ok=True)
    # Small contextual basemap, derived from the site's existing Natural Earth map.
    south = min(p["lat"] for p in track) - 2
    north = max(p["lat"] for p in track) + 4
    midpoint = (min(p["lon"] for p in track) + max(p["lon"] for p in track)) / 2
    span = max(
        max(p["lon"] for p in track) - min(p["lon"] for p in track) + 6,
        (north - south) * 280 / 180 / math.cos(math.radians((north + south) / 2)),
    )
    west, east = midpoint - span / 2, midpoint + span / 2
    land = json.loads((ROOT / "apps/stormsense/public/land.geojson").read_text())
    paths = []
    for feature in land["features"]:
        geometry = feature["geometry"]
        polygons = (
            geometry["coordinates"]
            if geometry["type"] == "MultiPolygon"
            else [geometry["coordinates"]]
        )
        for polygon in polygons:
            ring = polygon[0]
            if not any(west <= x <= east and south <= y <= north for x, y, *_ in ring):
                continue
            points = [
                (
                    round((x - west) / (east - west) * 280, 1),
                    round((north - y) / (north - south) * 180, 1),
                )
                for x, y, *_ in ring
            ]
            paths.append("M" + "L".join(f"{x},{y}" for x, y in points) + "Z")
    (output / "coast.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 280 180"><g fill="#27394b" stroke="#466172" stroke-width="0.7">'
        + "".join(f'<path d="{path}"/>' for path in paths)
        + "</g></svg>"
    )
    channels = []
    for i, band in enumerate(BANDS):
        image = colorize(
            array[i, 32:224, 32:224],
            190,
            300,
            ["#f3f3e9", "#abbec8", "#47657b", "#121c30"],
        )
        image[~valid[32:224, 32:224]] = [17, 20, 39]
        name = f"{band}.webp"
        Image.fromarray(image).save(output / name, lossless=True)
        channels.append(
            {"id": band, "image": f"/method/{name}", "sha256": digest(output / name)}
        )
    # The second stack shows the exact five additional network input planes.
    context_values = torch.cat(
        [
            batch["condition"][0, 10:] * batch["condition_mask"][0],
            batch["condition_mask"][0],
        ],
        dim=0,
    ).numpy()
    assert context_values.shape == (5, 192, 192)
    np.save(output / "context-tensors.npy", context_values, allow_pickle=False)
    context_channels = []
    for values, (identifier, label, short) in zip(
        context_values,
        [
            ("distance", "Center distance", "Dist"),
            ("solar-sin", "Solar time · sine", "Sin"),
            ("solar-cos", "Solar time · cosine", "Cos"),
            ("solar-zenith", "Solar zenith angle", "Sun"),
            ("validity", "Validity mask", "Mask"),
        ],
    ):
        visible = values[valid[32:224, 32:224]]
        low, high = (
            (0.0, 1.0)
            if identifier == "validity"
            else (float(visible.min()), float(visible.max()))
        )
        if high <= low:
            high = low + 1
        pixels = colorize(
            values, low, high, ["#132536", "#286a79", "#70c9ba", "#e7dda7"]
        )
        pixels[~valid[32:224, 32:224]] = [17, 20, 39]
        name = f"context-{identifier}.webp"
        Image.fromarray(pixels).save(output / name, lossless=True)
        context_channels.append(
            {
                "id": identifier,
                "label": label,
                "short_label": short,
                "image": f"/method/{name}",
                "sha256": digest(output / name),
                "display_min": low,
                "display_max": high,
            }
        )
    high = math.ceil(float(field.max()) / 10) * 10
    Image.fromarray(colorize(field, 0, high, PALETTE)).save(
        output / "wind-field.webp", lossless=True
    )
    # Preserve physical values for reproducibility; the UI only requests the WebP.
    np.save(output / "wind-field.npy", field, allow_pickle=False)
    training = export_training_pair(
        output,
        root,
        row,
        field,
        batch["condition_mask"][0, 0].numpy(),
        lat,
        lon,
        joint.image_huber_delta_ms,
    )
    models = {
        key: {"id": item["id"], "sha256": item["sha256"], "scalars": values}
        for (key, item), values in zip(
            manifests.items(),
            [
                scalars(
                    prediction.intensity_prediction_ms,
                    prediction.structure_prediction_km,
                ),
                scalars(intensity, structure),
            ],
        )
    }
    example = {
        "schema_version": 2,
        "storm": {"id": "AL092022", "name": "Ian"},
        "time": at.isoformat(),
        "sample_id": sample_id,
        "center": {"lat": lat, "lon": lon},
        "track": track,
        "map_bounds": [west, south, east, north],
        "channels": channels,
        "context_channels": context_channels,
        "context_tensor": {
            "values": "/method/context-tensors.npy",
            "sha256": digest(output / "context-tensors.npy"),
            "display": "Per-channel color scaling on valid pixels; masked pixels are dark. Values are the exact normalized network inputs.",
        },
        "models": models,
        "training": training,
        "field": {
            "image": "/method/wind-field.webp",
            "values": "/method/wind-field.npy",
            "sha256": digest(output / "wind-field.webp"),
            "values_sha256": digest(output / "wind-field.npy"),
            "min_ms": 0,
            "max_ms": high,
            "palette": PALETTE,
        },
        "grid": {
            "width": 192,
            "height": 192,
            "resolution_degrees": 0.027,
            "bounds": [
                lon - 96 * 0.027,
                lat - 96 * 0.027,
                lon + 96 * 0.027,
                lat + 96 * 0.027,
            ],
            "crs": "EPSG:4326",
            "valid_fraction": fraction,
        },
        "provenance": {
            "source": row["geo_path"],
            "source_sha256": digest(root / row["geo_path"]),
            "stats_sha256": digest(stats_path),
            "ibtracs_sha256": digest(ibtracs),
            "selection": "Highest matched USA_WIND among ABI Ian test rasters with >=90% valid storm-centered crop and valid central 4x4; ties by earliest scan then sample ID.",
            "matched_reference_wind_ms": reference_wind,
            "preprocessing": "Nearest-neighbor reproject to storm-centered 256x256 at 0.027 degrees; operational prepare: central 192x192, robust-zscore clip 4, distance and three solar channels, validity mask.",
            "units": {"wind": "m/s", "radii": "km", "brightness_temperature": "K"},
        },
    }
    (output / "example.json").write_text(
        json.dumps(example, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "sample": sample_id,
                "time": str(at),
                "valid_fraction": fraction,
                "models": models,
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "apps/stormsense/public/method"
    )
    export(parser.parse_args().output)
