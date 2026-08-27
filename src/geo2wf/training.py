from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from geo2wf.config.local_environment import load_local_env

load_local_env()

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dif_img_rec_matplotlib")

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from geo2wf.config import (
    compose_config,
    instantiate_datamodule,
    instantiate_model,
    load_config_file,
)
from geo2wf.tracking.callbacks import ReconstructionLoggingCallback
from geo2wf.tracking.run_manifest import (
    MachineReadableRunCallback,
    initialize_run_manifest,
    record_run_failure,
)


def load_config(config_path: str) -> dict:
    """Load a resolved YAML file or compose a Hydra defaults file."""
    return load_config_file(config_path)


def create_run_directory(config_path: str, parent: str | Path) -> Path:
    """Create one timestamped run directory before Lightning launches DDP ranks."""
    inherited = os.environ.get("GEO2WF_RUN_DIR")
    if inherited:
        return Path(inherited)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = Path(config_path).stem
    run_dir = (Path(parent) / f"{timestamp}_{config_name}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    # Lightning local DDP workers inherit this when they re-execute the script,
    # so every rank reuses the directory without attempting to create it again.
    os.environ["GEO2WF_RUN_DIR"] = str(run_dir)
    return run_dir


_ACTIVE_RUN_DIR: Path | None = None
_ACTIVE_FAILURE_PHASE: str | None = None


def _absolute_checkpoint_path(path: str | None) -> str | None:
    if not path:
        return None
    return str(Path(path).expanduser().resolve())


def resolve_runtime_config(config: dict) -> dict:
    """Return the composed active config ready to be snapshotted."""
    return config


def _configured_baseline_checkpoint(config: dict) -> str | None:
    """Compatibility field retained in the run-manifest schema."""
    return None


def _is_global_zero_environment() -> bool:
    """Best-effort rank check for failures raised before Trainer is available."""
    for name in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return int(value) == 0
        except ValueError:
            continue
    return True


def build_model(config: dict) -> pl.LightningModule:
    """Instantiate one active model from its local Hydra target."""
    return instantiate_model(config)


def main() -> None:
    """Entry point for training with PyTorch Lightning.

    Lightning flow in this script:
    1. Build data module (encapsulates DataLoaders).
    2. Build the configured LightningModule.
    3. Build `pl.Trainer` with runtime options.
    4. Call `trainer.fit(model, datamodule=datamodule)` to start the full train/val loop.
    """
    global _ACTIVE_FAILURE_PHASE, _ACTIVE_RUN_DIR

    _ACTIVE_FAILURE_PHASE = "argument_parsing"
    parser = argparse.ArgumentParser()
    # Path to the YAML config used for all runtime settings.
    parser.add_argument("--config", type=str, default="configs/modular.yaml")
    # Optional override for Lightning's validation batch fraction/count.
    parser.add_argument("--limit-val-batches", type=float, default=None)
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Optional Lightning checkpoint to resume from.",
    )
    parser.add_argument(
        "--weights-only-path",
        type=str,
        default=None,
        help=(
            "Optional Lightning checkpoint used only to initialize model weights. "
            "Optimizer, scheduler, callback, epoch, and global-step state are ignored."
        ),
    )
    args, overrides = parser.parse_known_args()
    if args.ckpt_path and args.weights_only_path:
        parser.error("--ckpt-path and --weights-only-path are mutually exclusive")

    args.ckpt_path = _absolute_checkpoint_path(args.ckpt_path)
    args.weights_only_path = _absolute_checkpoint_path(args.weights_only_path)
    if overrides:
        if args.config != "configs/modular.yaml":
            parser.error("Hydra overrides cannot be combined with --config")
        config = resolve_runtime_config(compose_config(overrides))
        args.config = "configs/modular.yaml"
    else:
        config = resolve_runtime_config(load_config(args.config))
    trainer_cfg = config.get("trainer", {})
    # Ampere and newer NVIDIA GPUs can accelerate float32 matrix products with
    # Tensor Cores. "high" retains more mantissa accuracy than "medium".
    torch.set_float32_matmul_precision(
        trainer_cfg.get("float32_matmul_precision", "high")
    )
    run_dir = create_run_directory(
        args.config, trainer_cfg.get("default_root_dir", "logs")
    )
    if not (run_dir / "run-manifest.json").exists():
        _ACTIVE_FAILURE_PHASE = "manifest_initialization"
        initialize_run_manifest(
            run_dir,
            config_path=args.config,
            config=config,
            checkpoint_paths={
                "deterministic_baseline": _configured_baseline_checkpoint(config),
                "resume": args.ckpt_path,
                "weights_only": args.weights_only_path,
            },
        )
    _ACTIVE_RUN_DIR = run_dir
    _ACTIVE_FAILURE_PHASE = "runtime_setup"
    wandb_dir = run_dir / "wandb"
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ["WANDB_CACHE_DIR"] = str(wandb_dir / "cache")
    os.environ["WANDB_CONFIG_DIR"] = str(wandb_dir / "config")

    # Ensures deterministic random behavior where possible.
    _ACTIVE_FAILURE_PHASE = "random_seed_setup"
    pl.seed_everything(config.get("seed", 42), workers=True)

    # DataModule centralizes loader construction and setup for Lightning.
    _ACTIVE_FAILURE_PHASE = "datamodule_build"
    datamodule = instantiate_datamodule(config)
    # Split config sections for clarity.
    wandb_cfg = config.get("logging", {}).get("wandb", {})

    # This is the Lightning model used for training and validation.
    _ACTIVE_FAILURE_PHASE = "model_build"
    model = build_model(config)
    validate_data_spec = getattr(model, "validate_data_spec", None)
    if callable(validate_data_spec):
        datamodule.setup("fit")
        validate_data_spec(datamodule.data_spec)
    if args.weights_only_path:
        _ACTIVE_FAILURE_PHASE = "weights_only_checkpoint_load"
        checkpoint = torch.load(
            args.weights_only_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(
            "Initialized model weights only from "
            f"{args.weights_only_path}; optimizer and scheduler start fresh."
        )

    # CLI override has priority over config file value.
    limit_val_batches = (
        args.limit_val_batches
        if args.limit_val_batches is not None
        else trainer_cfg.get("limit_val_batches", 1.0)
    )
    _ACTIVE_FAILURE_PHASE = "logger_setup"
    wandb_disabled = str(os.environ.get("WANDB_DISABLED", "")).lower() in {
        "1",
        "true",
        "yes",
    }
    wandb_enabled = wandb_cfg.get("enabled", True) and not wandb_disabled
    wandb_logger = (
        WandbLogger(
            project=os.environ.get(
                "WANDB_PROJECT", wandb_cfg.get("project", "dif_img_rec")
            ),
            name=os.environ.get("WANDB_NAME", wandb_cfg.get("name")),
            group=os.environ.get("WANDB_GROUP", wandb_cfg.get("group")),
            job_type=wandb_cfg.get("job_type", "training"),
            tags=list(wandb_cfg.get("tags", [])),
            save_dir=str(run_dir),
            log_model=wandb_cfg.get("log_model", False),
        )
        if wandb_enabled
        else False
    )
    csv_logger = CSVLogger(
        save_dir=str(run_dir),
        name="metrics",
        version="",
    )
    loggers = [csv_logger]
    if wandb_logger:
        loggers.append(wandb_logger)

    checkpoint_cfg = trainer_cfg.get("checkpoint", {})
    checkpoint_monitor = checkpoint_cfg.get("monitor") or getattr(
        model, "checkpoint_monitor", "val/loss"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename=checkpoint_cfg.get("filename", "epoch={epoch:03d}-step={step}"),
        monitor=checkpoint_monitor,
        mode=checkpoint_cfg.get("mode", getattr(model, "checkpoint_mode", "min")),
        save_top_k=checkpoint_cfg.get("save_top_k", 2),
        save_last=checkpoint_cfg.get("save_last", False),
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpointing_enabled = bool(trainer_cfg.get("enable_checkpointing", True))
    callbacks = [
        MachineReadableRunCallback(run_dir),
        ReconstructionLoggingCallback(),
        lr_monitor,
    ]
    if checkpointing_enabled:
        callbacks.append(checkpoint_callback)
    early_stopping_cfg = trainer_cfg.get("early_stopping", {})
    if early_stopping_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping_cfg.get("monitor", checkpoint_monitor),
                mode=early_stopping_cfg.get(
                    "mode",
                    checkpoint_cfg.get(
                        "mode", getattr(model, "checkpoint_mode", "min")
                    ),
                ),
                patience=int(early_stopping_cfg.get("patience", 50)),
                min_delta=float(early_stopping_cfg.get("min_delta", 0.0)),
                strict=bool(early_stopping_cfg.get("strict", True)),
                check_finite=bool(early_stopping_cfg.get("check_finite", True)),
            )
        )
    # Trainer controls loop behavior, device placement, precision, and logging cadence.
    trainer_kwargs = {
        "max_epochs": trainer_cfg.get("max_epochs", 1),
        "accelerator": trainer_cfg.get("accelerator", "auto"),
        "devices": trainer_cfg.get("devices", 1),
        "precision": trainer_cfg.get("precision", 32),
        "log_every_n_steps": trainer_cfg.get("log_every_n_steps", 10),
        "enable_checkpointing": checkpointing_enabled,
        "limit_val_batches": limit_val_batches,
        "limit_train_batches": trainer_cfg.get("limit_train_batches", 1.0),
        "logger": loggers,
        "default_root_dir": str(run_dir),
        "callbacks": callbacks,
        "replace_sampler_ddp": not datamodule.intensity_balanced_sampling,
    }
    if trainer_cfg.get("strategy") is not None:
        trainer_kwargs["strategy"] = trainer_cfg["strategy"]
    if trainer_cfg.get("deterministic") is not None:
        trainer_kwargs["deterministic"] = trainer_cfg["deterministic"]

    _ACTIVE_FAILURE_PHASE = "trainer_initialization"
    trainer = pl.Trainer(**trainer_kwargs)

    # Starts the training/validation loop.
    _ACTIVE_FAILURE_PHASE = "trainer_fit"
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    _ACTIVE_FAILURE_PHASE = "completed"


def _entrypoint() -> None:
    try:
        main()
    except BaseException as exception:
        if (
            _ACTIVE_RUN_DIR is not None
            and (_ACTIVE_RUN_DIR / "run-manifest.json").exists()
            and _is_global_zero_environment()
        ):
            record_run_failure(
                _ACTIVE_RUN_DIR,
                exception,
                phase=_ACTIVE_FAILURE_PHASE,
            )
        raise


if __name__ == "__main__":
    _entrypoint()
