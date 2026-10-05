"""Scalar-only adaptation of the pinned nowcast; no SAR inputs or decoder."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from geo2wf.models.bottleneck_unet_mlp import BottleneckUNetMLPRegressor
from geo2wf.models.bottleneck_unet_mlp.module import BottleneckEncoderMLP, _huber_values
from geo2wf.operational.common import file_hash, KNOT
from geo2wf.operational.models import checkpoint, prepare, MANIFEST
from .dataset import TARGETS, coverage, verify, write_json


class HistoricalDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if file_hash(self.root / "stats.json") != self.manifest["stats_sha256"]:
            raise ValueError("Normalization checksum mismatch")
        self.stats = json.loads((self.root / "stats.json").read_text())
        with sqlite3.connect(
            f'file:{(self.root/"collection.sqlite").resolve()}?mode=ro', uri=True
        ) as db:
            ready = {
                r[0]: json.loads(r[1])
                for r in db.execute(
                    "SELECT id,body FROM observations WHERE status='ready'"
                )
            }
        self.rows = [
            {**r, "artifact": ready[r["sample_id"]]}
            for r in self.manifest["samples"]
            if r["split"] == split and r["sample_id"] in ready
        ]
        if not self.rows:
            raise ValueError(f"No ready observations for {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.root / row["artifact"]["path"], allow_pickle=False) as data:
            batch = prepare(
                data["array"],
                data["valid"],
                row["lat"],
                row["lon"],
                row["time"],
                self.stats,
            )
        # Operational prepare also supplies field normalization metadata, which this
        # scalar-only interface intentionally never exposes to the model.
        return {
            "condition": batch["condition"][0],
            "condition_mask": batch["condition_mask"][0],
            "intensity": torch.tensor(row["labels"]["vmax"], dtype=torch.float32),
            "structure": torch.tensor(
                [row["labels"][k] if row["label_valid"][k] else 0.0 for k in TARGETS]
            ),
            "structure_valid": torch.tensor([row["label_valid"][k] for k in TARGETS]),
            "index": index,
        }


def load_encoder(model_root):
    source = BottleneckUNetMLPRegressor.load_from_checkpoint(
        checkpoint("nowcast", model_root), map_location="cpu"
    )
    hp = source.hparams
    model = BottleneckEncoderMLP(
        hp.condition_channels + 1,
        hp.base_channels,
        hp.channel_mults,
        hp.intensity_hidden_features,
        hp.intensity_dropout,
        hp.initial_intensity_ms,
        5,
    )
    weights = {
        k: v
        for k, v in source.model.state_dict().items()
        if not k.startswith(
            ("decoder.", "decoder_projections.", "reconstruction_head.")
        )
    }
    model.load_state_dict(weights, strict=True)
    return model


class ScalarAdapter(pl.LightningModule):
    def __init__(
        self,
        mode="heads",
        model_root="downloads/models",
        head_lr=2e-4,
        encoder_lr=2e-5,
        weight_decay=1e-4,
        architecture=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        if mode not in ("heads", "encoder"):
            raise ValueError("mode must be heads or encoder")
        self.model = (
            load_encoder(model_root)
            if architecture is None
            else BottleneckEncoderMLP(**architecture)
        )
        self.mode = mode
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(mode == "encoder" or self.is_head(name))
        self.validation_totals = None

    @staticmethod
    def is_head(name):
        return name.startswith(("intensity_mlp.", "intensity_head.", "structure_head."))

    def forward(self, batch):
        condition, mask = batch["condition"], batch["condition_mask"].to(
            batch["condition"]
        )
        inputs = torch.cat([condition * mask, mask], dim=1)
        if self.mode == "heads":
            with torch.no_grad():
                bottleneck, _ = self.model.encode(inputs)
            _, intensity, structure = self.model.intensity_features(bottleneck)
        else:
            output = self.model(inputs)
            intensity, structure = (
                output.intensity_prediction_ms,
                output.structure_prediction_km,
            )
        return intensity, structure

    def loss_terms(self, batch):
        intensity, structure = self(batch)
        iv = _huber_values(intensity - batch["intensity"], 5.0)
        valid = batch["structure_valid"]
        target = torch.where(valid, batch["structure"], structure.detach())
        sv = F.smooth_l1_loss(structure, target, reduction="none", beta=20.0) * valid
        return iv.sum(), iv.numel(), sv.sum(), valid.sum()

    def training_step(self, batch, batch_idx):
        isum, n, ssum, sn = self.loss_terms(batch)
        loss = isum / n + 0.25 * ssum / sn.clamp_min(1)
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=n)
        return loss

    def on_validation_epoch_start(self):
        self.validation_totals = torch.zeros(4, device=self.device, dtype=torch.float64)

    def validation_step(self, batch, batch_idx):
        isum, n, ssum, sn = self.loss_terms(batch)
        self.validation_totals += torch.stack(
            [
                isum.detach().double(),
                isum.new_tensor(n).double(),
                ssum.detach().double(),
                sn.double(),
            ]
        )

    def on_validation_epoch_end(self):
        isum, n, ssum, sn = self.validation_totals
        self.log(
            "val/loss",
            isum / n.clamp_min(1) + 0.25 * ssum / sn.clamp_min(1),
            prog_bar=True,
        )

    def configure_optimizers(self):
        heads, encoder = [], []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                (heads if self.is_head(name) else encoder).append(p)
        groups = [{"params": heads, "lr": self.hparams.head_lr}]
        if encoder:
            groups.append({"params": encoder, "lr": self.hparams.encoder_lr})
        return torch.optim.AdamW(groups, weight_decay=self.hparams.weight_decay)


def metrics(rows):
    result = {}
    for key in ("vmax", "rmw", "r34", "r50", "r64"):
        usable = [r for r in rows if r["target"][key] is not None]
        if not usable:
            result[key] = {"n": 0}
            continue
        errors = np.array([r["prediction"][key] - r["target"][key] for r in usable])
        storms = defaultdict(list)
        for r, e in zip(usable, errors):
            storms[r["storm_id"]].append(abs(float(e)))
        result[key] = {
            "n": len(usable),
            "storms": len(storms),
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "bias": float(errors.mean()),
            "storm_macro_mae": float(np.mean([np.mean(v) for v in storms.values()])),
        }
    return result


def evaluate_model(model, dataset, device, batch_size):
    model.to(device).eval()
    rows = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            indices = batch.pop("index").tolist()
            intensity, structure = model({k: v.to(device) for k, v in batch.items()})
            for index, v, s in zip(
                indices, intensity.cpu().tolist(), structure.cpu().tolist()
            ):
                source = dataset.rows[index]
                rows.append(
                    {
                        "sample_id": source["sample_id"],
                        "storm_id": source["storm_id"],
                        "basin": source["basin"],
                        "time": source["time"],
                        "is_ri": source["is_ri"],
                        "pretraining_membership": source["pretraining_membership"],
                        "target": source["labels"],
                        "prediction": dict(zip(("vmax", *TARGETS), (v, *s))),
                    }
                )
    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups["basin/" + r["basin"]].append(r)
        groups["pretraining/" + r["pretraining_membership"]].append(r)
        wind = r["target"]["vmax"] / KNOT
        groups[
            "intensity/"
            + (
                "major"
                if wind >= 96
                else (
                    "hurricane"
                    if wind >= 64
                    else "storm" if wind >= 34 else "depression"
                )
            )
        ].append(r)
        groups["ri/" + str(r["is_ri"])].append(r)
    return rows, {k: metrics(v) for k, v in groups.items()}


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root, output = args.root, args.output
    manifest = json.loads((root / "manifest.json").read_text())
    verification = verify(
        root, manifest, "historical" if args.command == "train" else "test"
    )
    if verification["errors"] or verification["checked"] == 0:
        raise ValueError(
            "Dataset verification must pass with ready observations before running"
        )
    report = coverage(root, manifest)
    # Transient acquisition failures must be resolved before freezing the cohort.
    relevant = ("train/", "val/") if args.command == "train" else ("test/",)
    if any(
        count
        for key, count in report["counts"].items()
        if key.endswith("/failed") and key.startswith(relevant)
    ):
        raise ValueError("Retry acquisition failures before training/evaluation")
    if manifest["model"] != MANIFEST:
        raise ValueError("Pinned model manifest differs from dataset provenance")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pl.seed_everything(42, workers=True)
    torch.set_num_threads(4)
    if args.command == "train":
        if (output / "config.json").exists() and not args.resume:
            raise ValueError("Output exists; use --resume or a new output directory")
        output.mkdir(parents=True, exist_ok=True)
        datasets = {split: HistoricalDataset(root, split) for split in ("train", "val")}
        config = {
            "mode": args.mode,
            "manifest_sha256": file_hash(root / "manifest.json"),
            "stats_sha256": manifest["stats_sha256"],
            "checkpoint_sha256": MANIFEST["models"]["nowcast"]["sha256"],
            "seed": 42,
            "code_sha256": {
                path.name: file_hash(path)
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
            "head_lr": 2e-4,
            "encoder_lr": 2e-5,
            "weight_decay": 1e-4,
            "effective_batch_size": 32,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "patience": 15,
            "structure_loss_weight": 0.25,
            "counts": {k: len(v) for k, v in datasets.items()},
            "sample_ids": {
                k: [r["sample_id"] for r in v.rows] for k, v in datasets.items()
            },
        }
        if args.resume and json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Resume configuration/cohort mismatch")
        write_json(output / "config.json", config)
        write_json(output / "dataset-coverage.json", report)
        model = ScalarAdapter(args.mode, args.model_root)
        loaders = {
            k: DataLoader(
                v,
                batch_size=args.batch_size,
                shuffle=k == "train",
                num_workers=args.workers,
                pin_memory=device.startswith("cuda"),
            )
            for k, v in datasets.items()
        }
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=output / "checkpoints",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best-{epoch:03d}",
        )
        trainer = pl.Trainer(
            accelerator="gpu" if device.startswith("cuda") else "cpu",
            devices=1,
            max_epochs=args.epochs,
            accumulate_grad_batches=32 // args.batch_size,
            callbacks=[
                checkpoint_callback,
                pl.callbacks.EarlyStopping(monitor="val/loss", patience=15, mode="min"),
            ],
            logger=pl.loggers.CSVLogger(str(output), name="curves"),
            deterministic=True,
            enable_progress_bar=False,
            log_every_n_steps=10,
        )
        trainer.fit(
            model,
            loaders["train"],
            loaders["val"],
            ckpt_path=str(output / "checkpoints/last.ckpt") if args.resume else None,
        )
        write_json(
            output / "result.json",
            {
                "best_checkpoint": checkpoint_callback.best_model_path,
                "checkpoint_sha256": file_hash(checkpoint_callback.best_model_path),
                "best_validation_loss": float(checkpoint_callback.best_model_score),
                "completed_epochs": trainer.current_epoch,
            },
        )
    else:
        configs = [
            json.loads((path / "config.json").read_text())
            for path in (args.heads_run, args.encoder_run)
        ]
        for key in (
            "manifest_sha256",
            "sample_ids",
            "checkpoint_sha256",
            "seed",
            "max_epochs",
            "batch_size",
        ):
            if configs[0][key] != configs[1][key]:
                raise ValueError("Experiments differ in " + key)
        if configs[0]["mode"] != "heads" or configs[1]["mode"] != "encoder":
            raise ValueError("Incorrect experiment modes")
        if configs[0]["manifest_sha256"] != file_hash(root / "manifest.json"):
            raise ValueError("Evaluation dataset differs from training manifest")
        dataset = HistoricalDataset(root, "test")
        all_metrics = {}
        for name, path in [
            ("baseline", None),
            ("heads", args.heads_run),
            ("encoder", args.encoder_run),
        ]:
            if path is None:
                model = ScalarAdapter("heads", args.model_root)
            else:
                result = json.loads((path / "result.json").read_text())
                if file_hash(result["best_checkpoint"]) != result["checkpoint_sha256"]:
                    raise ValueError("Adapted checkpoint checksum mismatch")
                model = ScalarAdapter.load_from_checkpoint(
                    result["best_checkpoint"], map_location="cpu"
                )
            rows, summary = evaluate_model(model, dataset, device, args.batch_size)
            write_json(output / f"{name}-predictions.json", rows)
            all_metrics[name] = summary
            del model
        write_json(
            output / "comparison.json",
            {
                "manifest_sha256": file_hash(root / "manifest.json"),
                "units": {"vmax": "m/s", "radii": "km"},
                "metrics": all_metrics,
            },
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "evaluate"])
    p.add_argument("--root", type=Path, default=Path("data/historical_storms_v1"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["heads", "encoder"], default="heads")
    p.add_argument("--model-root", default="downloads/models")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--heads-run", type=Path, default=Path("logs/historical/heads"))
    p.add_argument("--encoder-run", type=Path, default=Path("logs/historical/encoder"))
    args = p.parse_args()
    if args.batch_size < 1 or 32 % args.batch_size:
        p.error("batch-size must divide effective batch size 32")
    run(args)


if __name__ == "__main__":
    main()
