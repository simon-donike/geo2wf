"""Pinned GEO-only nowcast and history-only forecast inference."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from geo2wf.data.features import normalized_distance_to_center, solar_time_features
from geo2wf.data.normalization import normalize, normalization_affine_parameters

from .common import file_hash
from .satellite import BANDS
from . import PIPELINE_VERSION

MANIFEST = json.loads(Path(__file__).with_name("models.json").read_text())
ROOT = Path(__file__).resolve().parents[3]


def version(role):
    item = MANIFEST["models"][role]
    return f"{PIPELINE_VERSION}:{item['id']}:{item['sha256'][:12]}"


def resolve_file(name, model_root):
    candidates = [Path(model_root) / name, ROOT / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {name}; run geo2wf-operational bootstrap")


def bootstrap(model_root):
    names = set()
    for item in MANIFEST["models"].values():
        names.update(
            item[k]
            for k in ("checkpoint", "config", "stats", "training_manifest")
            if k in item
        )
    subprocess.run(
        [
            "hf",
            "download",
            MANIFEST["repository"],
            *sorted(names),
            "--revision",
            MANIFEST["revision"],
            "--local-dir",
            str(model_root),
        ],
        check=True,
    )


def checkpoint(role, model_root):
    item = MANIFEST["models"][role]
    local = ROOT / item["local_checkpoint"]
    path = local if local.is_file() else resolve_file(item["checkpoint"], model_root)
    if file_hash(path) != item["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {role}")
    return path


def prepare(array, valid, lat, lon, at, stats):
    values = torch.from_numpy(np.asarray(array, dtype=np.float32))
    normalized = normalize(
        values,
        "geo",
        list(BANDS),
        stats,
        normalization="robust-zscore",
        robust_clip=4.0,
    )
    mask = torch.from_numpy(valid).unsqueeze(0)
    condition = (torch.nan_to_num(normalized) * mask)[:, 32:224, 32:224]
    mask = mask[:, 32:224, 32:224]
    half = 192 * 0.027 / 2
    bounds = torch.tensor(
        [lon - half, lon + half, lat - half, lat + half], dtype=torch.float64
    )
    distance = normalized_distance_to_center(
        bounds, (192, 192), torch.tensor([lat, lon])
    )
    solar = solar_time_features(bounds, (192, 192), at)
    condition = torch.cat([condition, distance, solar], dim=0)
    offset, scale = normalization_affine_parameters(
        "sar", ["wind_speed"], stats, normalization="min-max"
    )
    return {
        "condition": condition.unsqueeze(0),
        "condition_mask": mask.unsqueeze(0),
        "target_norm_offset": offset.unsqueeze(0),
        "target_norm_scale": scale.unsqueeze(0),
    }


class Models:
    def __init__(self, model_root="downloads/models", device=None):
        from geo2wf.models.bottleneck_unet_mlp import BottleneckUNetMLPRegressor
        from geo2wf.models.intensity_forecast import IntensityForecastMLP

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(2)
        for item in MANIFEST["models"].values():
            if (
                file_hash(resolve_file(item["config"], model_root))
                != item["config_sha256"]
            ):
                raise ValueError("Pinned model configuration checksum mismatch")
        self.nowcast = (
            BottleneckUNetMLPRegressor.load_from_checkpoint(
                checkpoint("nowcast", model_root), map_location="cpu"
            )
            .eval()
            .to(self.device)
        )
        self.forecast = (
            IntensityForecastMLP.load_from_checkpoint(
                checkpoint("forecast", model_root), map_location="cpu"
            )
            .eval()
            .to(self.device)
        )
        item = MANIFEST["models"]["nowcast"]
        stats_path = resolve_file(item["stats"], model_root)
        if file_hash(stats_path) != item["stats_sha256"]:
            raise ValueError("Normalization statistics checksum mismatch")
        self.stats = json.loads(stats_path.read_text())

    def infer(self, array, valid, lat, lon, at):
        batch = {
            k: v.to(self.device)
            for k, v in prepare(array, valid, lat, lon, at, self.stats).items()
        }
        with torch.inference_mode():
            prediction = self.nowcast.predict_joint(batch)
        radii = prediction.structure_prediction_km[0].cpu().tolist()
        values = {
            "vmax_ms": float(prediction.intensity_prediction_ms[0]),
            "rmw_km": radii[1],
            "r34_km": radii[2],
            "r50_km": radii[3],
            "r64_km": radii[4],
        }
        if not all(np.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError("Model returned invalid scalar values")
        return values

    def predict_future(self, current, minus6, minus12):
        values = [
            torch.tensor([x], dtype=torch.float32, device=self.device)
            for x in (current, minus6, minus12)
        ]
        with torch.inference_mode():
            predictions = self.forecast.predict_two_steps(*values)
        result = [float(x[0]) for x in predictions]
        if not all(np.isfinite(x) and x >= 0 for x in result):
            raise ValueError("Forecast model returned invalid intensity")
        return result
