"""Pinned GEO-only nowcast and history-only forecast inference."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from geo2wf.data.features import normalized_distance_to_center, solar_time_features
from geo2wf.data.normalization import normalize, normalization_affine_parameters

from .common import file_hash
from .satellite import BANDS
from . import PIPELINE_VERSION

MANIFEST_PATH = Path(
    os.environ.get("STORMSENSE_MODEL_MANIFEST", Path(__file__).with_name("models.json"))
).resolve()
MANIFEST = json.loads(MANIFEST_PATH.read_text())
ROOT = Path(__file__).resolve().parents[3]


def select_manifest(path):
    """Select once per CLI/worker process, preserving imported manifest references."""
    global MANIFEST_PATH
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    policy = manifest.get("prediction_policy", {})
    if policy.get("cadence_hours", 1) not in (1, 2):
        raise ValueError("Prediction cadence must be one or two hours")
    for role in ("nowcast", "forecast"):
        if role not in manifest.get("models", {}):
            raise ValueError(f"Missing model role: {role}")
    MANIFEST.clear()
    MANIFEST.update(manifest)
    MANIFEST_PATH = path
    os.environ["STORMSENSE_MODEL_MANIFEST"] = str(path)


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
        if item.get("loader") == "scalar_adapter":
            # This locally trained checkpoint is not in the pinned upstream
            # Hub release. Never try to fetch it from that repository.
            checkpoint("nowcast", model_root)
        names.update(
            item[k]
            for k in ("checkpoint", "config", "stats", "training_manifest")
            if k in item
            and not (
                item.get("loader") == "scalar_adapter" and k in ("checkpoint", "config")
            )
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
        self.scalar_adapter = (
            MANIFEST["models"]["nowcast"].get("loader") == "scalar_adapter"
        )
        if self.scalar_adapter:
            from geo2wf.historical.training import ScalarAdapter

            item = MANIFEST["models"]["nowcast"]
            if item["bands"] != list(BANDS):
                raise ValueError(
                    "Pinned input band order differs from operational inputs"
                )
            self.nowcast = (
                ScalarAdapter.load_from_checkpoint(
                    checkpoint("nowcast", model_root),
                    map_location="cpu",
                    architecture=item["architecture"],
                )
                .eval()
                .to(self.device)
            )
        else:
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
            if self.scalar_adapter:
                intensity, structure = self.nowcast(batch)
            else:
                prediction = self.nowcast.predict_joint(batch)
                intensity, structure = (
                    prediction.intensity_prediction_ms,
                    prediction.structure_prediction_km,
                )
        radii = structure[0].cpu().tolist()
        values = {
            "vmax_ms": float(intensity[0]),
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
