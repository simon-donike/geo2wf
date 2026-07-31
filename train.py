from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from scripts.local_env import load_local_env

load_local_env()

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dif_img_rec_matplotlib")

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from data import PairedDataModule
from src.ERA5Residual import ERA5ResidualRegressor
from src.ERA5ResidualDiffusion import (
    ERA5ResidualDiffusion,
    load_frozen_deterministic_baseline,
)
from src.PixelDiffusion import PixelDiffusionConditional
from src.experiment_tracking import (
    MachineReadableRunCallback,
    initialize_run_manifest,
    record_run_failure,
)


def load_config(config_path: str) -> dict:
    """Load the YAML config file used to build data, model, and trainer."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
    """Materialize environment-backed checkpoint paths in the saved config."""
    model_cfg = config.get("model", {})
    if str(model_cfg.get("type", "diffusion")).lower() != "diffusion_residual":
        return config
    residual_cfg = model_cfg.setdefault("residual", {})
    baseline_cfg = residual_cfg.setdefault("baseline", {})
    if str(baseline_cfg.get("source", "era5")).lower() != "deterministic":
        return config
    checkpoint_path = baseline_cfg.get("checkpoint_path") or os.environ.get(
        "GEO2WF_BASELINE_CKPT"
    )
    if checkpoint_path:
        baseline_cfg["checkpoint_path"] = _absolute_checkpoint_path(
            str(checkpoint_path)
        )
    return config


def _configured_baseline_checkpoint(config: dict) -> str | None:
    model_cfg = config.get("model", {})
    if str(model_cfg.get("type", "")).lower() != "diffusion_residual":
        return None
    baseline_cfg = model_cfg.get("residual", {}).get("baseline", {})
    if str(baseline_cfg.get("source", "era5")).lower() != "deterministic":
        return None
    return baseline_cfg.get("checkpoint_path")


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
    """Build the configured diffusion or deterministic residual model."""
    model_cfg = config.get("model", {})
    opt_cfg = config.get("optimization", {})
    lr_sched_cfg = opt_cfg.get("reduce_lr_on_plateau", {})
    validation_cfg = config.get("validation", {})
    model_type = str(model_cfg.get("type", "diffusion")).lower()

    if model_type in {"diffusion", "diffusion_residual"}:
        unet_cfg = model_cfg.get("unet", {})
        sampling_cfg = model_cfg.get("sampling", {})
        sparse_cfg = model_cfg.get("sparse_target", {})
        guidance_cfg = model_cfg.get("classifier_free_guidance", {})
        ema_cfg = opt_cfg.get("ema", {})
        ema_decay = (
            ema_cfg.get("decay", 0.999) if ema_cfg.get("enabled", False) else None
        )
        diffusion_kwargs = {
            "generated_channels": model_cfg.get("out_channels", 2),
            "num_timesteps": model_cfg.get("num_timesteps", 1000),
            "schedule": model_cfg.get("schedule", "linear"),
            "model_dim": unet_cfg.get("dim", 64),
            "model_dim_mults": tuple(unet_cfg.get("dim_mults", [1, 2, 4, 8])),
            "model_channels": unet_cfg.get("channels"),
            "model_out_dim": unet_cfg.get("out_dim"),
            "lr": opt_cfg.get("lr", 1e-3),
            "lr_scheduler_factor": lr_sched_cfg.get("factor", 0.5),
            "lr_scheduler_patience": lr_sched_cfg.get("patience", 25),
            "lr_scheduler_monitor": lr_sched_cfg.get(
                "monitor", "val/eye_structure_score"
            ),
            "lr_scheduler_cooldown": lr_sched_cfg.get("cooldown", 0),
            "lr_scheduler_min_lr": lr_sched_cfg.get("min_lr", 0.0),
            "sampling_method": sampling_cfg.get("method", "ddpm"),
            "sampling_timesteps": sampling_cfg.get("timesteps"),
            "sampling_eta": sampling_cfg.get("eta", 0.0),
            "guidance_scale": sampling_cfg.get("guidance_scale", 1.0),
            "clip_sample": sampling_cfg.get("clip_sample", True),
            "sparse_target_fill": sparse_cfg.get("fill"),
            "unobserved_loss_weight": sparse_cfg.get("unobserved_loss_weight", 0.0),
            "validation_reconstruction_batches": validation_cfg.get(
                "reconstruction_batches", 1
            ),
            "log_reconstruction_images": validation_cfg.get(
                "log_reconstruction_images", True
            ),
            "validation_seed": validation_cfg.get(
                "sampling_seed", config.get("seed", 42)
            ),
            "validation_ensemble_size": validation_cfg.get("ensemble_size", 1),
            "validation_ensemble_batches": validation_cfg.get("ensemble_batches", 1),
            "probabilistic_score_sharpness_weight": validation_cfg.get(
                "probabilistic_score_sharpness_weight", 2.0
            ),
            "probabilistic_score_target_sharpness_ratio": validation_cfg.get(
                "probabilistic_score_target_sharpness_ratio", 1.0
            ),
            "probabilistic_score_peak_weight": validation_cfg.get(
                "probabilistic_score_peak_weight", 0.0
            ),
            "probabilistic_peak_fraction": validation_cfg.get(
                "probabilistic_peak_fraction", 0.005
            ),
            "min_snr_gamma": opt_cfg.get("min_snr_gamma"),
            "condition_dropout_probability": guidance_cfg.get(
                "condition_dropout_probability", 0.0
            ),
            "ema_decay": ema_decay,
            "ema_update_after_step": ema_cfg.get("update_after_step", 0),
            "ema_use_for_eval": ema_cfg.get("use_for_eval", True),
        }
        if model_type == "diffusion":
            return PixelDiffusionConditional(
                condition_channels=model_cfg.get("in_channels", 3),
                **diffusion_kwargs,
            )

        residual_cfg = model_cfg.get("residual", {})
        residual_loss_cfg = residual_cfg.get("loss", {})
        baseline_cfg = residual_cfg.get("baseline", {})
        baseline_source = str(baseline_cfg.get("source", "era5")).lower()
        baseline_model = None
        if baseline_source == "deterministic":
            checkpoint_path = baseline_cfg.get("checkpoint_path") or os.environ.get(
                "GEO2WF_BASELINE_CKPT"
            )
            if not checkpoint_path:
                raise ValueError(
                    "diffusion_residual with a deterministic baseline requires "
                    "model.residual.baseline.checkpoint_path or the "
                    "GEO2WF_BASELINE_CKPT environment variable"
                )
            baseline_model = load_frozen_deterministic_baseline(checkpoint_path)
        return ERA5ResidualDiffusion(
            base_condition_channels=model_cfg.get("in_channels", 21),
            baseline_source=baseline_source,
            baseline_model=baseline_model,
            residual_transform=residual_cfg.get("transform", "asinh"),
            residual_soft_scale_ms=residual_cfg.get("soft_scale_ms", 5.0),
            residual_clip_ms=residual_cfg.get("clip_ms", 80.0),
            prediction_min_ms=residual_cfg.get("prediction_min_ms", 0.0),
            prediction_max_ms=residual_cfg.get("prediction_max_ms", 80.0),
            preserve_baseline_condition_on_dropout=guidance_cfg.get(
                "preserve_baseline_condition_on_dropout", False
            ),
            gradient_loss_weight=residual_loss_cfg.get("gradient_weight", 0.0),
            spectrum_loss_weight=residual_loss_cfg.get("spectrum_weight", 0.0),
            low_frequency_loss_weight=residual_loss_cfg.get(
                "low_frequency_weight", 0.0
            ),
            smoothness_loss_weight=residual_loss_cfg.get("smoothness_weight", 0.0),
            robust_peak_loss_weight=residual_loss_cfg.get("robust_peak_weight", 0.0),
            robust_peak_fraction=residual_loss_cfg.get("robust_peak_fraction", 0.005),
            radial_profile_loss_weight=residual_loss_cfg.get(
                "radial_profile_weight", 0.0
            ),
            radial_profile_bins=residual_loss_cfg.get("radial_profile_bins", 16),
            radial_profile_max_radius_km=residual_loss_cfg.get(
                "radial_profile_max_radius_km"
            ),
            exceedance_area_loss_weight=residual_loss_cfg.get(
                "exceedance_area_weight", 0.0
            ),
            exceedance_thresholds_ms=tuple(
                residual_loss_cfg.get("exceedance_thresholds_ms", [17.0, 33.0, 43.0])
            ),
            exceedance_temperature_ms=residual_loss_cfg.get(
                "exceedance_temperature_ms", 1.0
            ),
            multiscale_field_loss_weight=residual_loss_cfg.get(
                "multiscale_field_weight", 0.0
            ),
            multiscale_field_kernel_sizes=tuple(
                residual_loss_cfg.get("multiscale_kernel_sizes", [1, 3, 9])
            ),
            annular_residual_loss_weight=residual_loss_cfg.get(
                "annular_residual_weight", 0.0
            ),
            auxiliary_max_timestep_fraction=residual_loss_cfg.get(
                "auxiliary_max_timestep_fraction", 0.5
            ),
            high_wind_threshold_ms=residual_loss_cfg.get(
                "high_wind_threshold_ms", 17.0
            ),
            high_wind_loss_weight=residual_loss_cfg.get("high_wind_weight", 1.0),
            inner_core_radius_km=residual_loss_cfg.get("inner_core_radius_km", 100.0),
            inner_core_loss_weight=residual_loss_cfg.get("inner_core_weight", 1.0),
            high_gradient_threshold_ms=residual_loss_cfg.get(
                "high_gradient_threshold_ms", 2.0
            ),
            high_gradient_loss_weight=residual_loss_cfg.get(
                "high_gradient_weight", 1.0
            ),
            low_frequency_kernel_size=residual_loss_cfg.get(
                "low_frequency_kernel_size", 9
            ),
            **diffusion_kwargs,
        )

    if model_type == "deterministic_residual":
        residual_cfg = model_cfg.get("residual", {})
        residual_loss_cfg = residual_cfg.get("loss", {})
        return ERA5ResidualRegressor(
            condition_channels=model_cfg.get(
                "condition_channels",
                model_cfg.get("in_channels", 20),
            ),
            base_channels=residual_cfg.get("base_channels", 32),
            channel_mults=tuple(residual_cfg.get("channel_mults", [1, 2, 4, 8])),
            huber_delta_ms=opt_cfg.get(
                "huber_delta_ms",
                residual_cfg.get("huber_delta_ms", 2.0),
            ),
            off_swath_anchor_weight=opt_cfg.get(
                "off_swath_anchor_weight",
                residual_cfg.get("off_swath_anchor_weight", 0.05),
            ),
            high_wind_threshold_ms=residual_cfg.get("high_wind_threshold_ms", 17.0),
            high_wind_weight_max=residual_loss_cfg.get("high_wind_weight_max", 1.0),
            high_wind_weight_start_ms=residual_loss_cfg.get(
                "high_wind_weight_start_ms", 25.0
            ),
            high_wind_weight_full_ms=residual_loss_cfg.get(
                "high_wind_weight_full_ms", 43.0
            ),
            peak_loss_weight=residual_loss_cfg.get("peak_weight", 0.0),
            peak_top_fraction=residual_loss_cfg.get("peak_top_fraction", 0.005),
            peak_min_pixels=residual_loss_cfg.get("peak_min_pixels", 16),
            peak_inner_core_radius_km=residual_loss_cfg.get(
                "peak_inner_core_radius_km", 100.0
            ),
            radial_profile_loss_weight=residual_loss_cfg.get(
                "radial_profile_weight", 0.0
            ),
            radial_profile_max_radius_km=residual_loss_cfg.get(
                "radial_profile_max_radius_km", 200.0
            ),
            radial_profile_bin_km=residual_loss_cfg.get("radial_profile_bin_km", 10.0),
            radial_profile_min_bin_pixels=residual_loss_cfg.get(
                "radial_profile_min_bin_pixels", 4
            ),
            exceedance_area_loss_weight=residual_loss_cfg.get(
                "exceedance_area_weight", 0.0
            ),
            exceedance_area_thresholds_ms=tuple(
                residual_loss_cfg.get("exceedance_thresholds_ms", [17.0, 33.0, 43.0])
            ),
            exceedance_area_temperature_ms=residual_loss_cfg.get(
                "exceedance_temperature_ms", 1.0
            ),
            peak_structure_radial_weight=residual_loss_cfg.get(
                "peak_structure_radial_weight", 0.25
            ),
            peak_structure_area_weight=residual_loss_cfg.get(
                "peak_structure_area_weight", 10.0
            ),
            prediction_min_ms=residual_cfg.get("prediction_min_ms", 0.0),
            prediction_max_ms=residual_cfg.get("prediction_max_ms"),
            psnr_data_range_ms=residual_cfg.get("psnr_data_range_ms", 79.8),
            lr=opt_cfg.get("lr", 3e-4),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
            lr_scheduler_factor=lr_sched_cfg.get("factor", 0.5),
            lr_scheduler_patience=lr_sched_cfg.get("patience", 10),
            lr_scheduler_monitor=lr_sched_cfg.get("monitor", "val/eye_structure_score"),
            lr_scheduler_cooldown=lr_sched_cfg.get("cooldown", 0),
            lr_scheduler_min_lr=lr_sched_cfg.get("min_lr", 0.0),
            validation_reconstruction_batches=validation_cfg.get(
                "reconstruction_batches", 1
            ),
            log_reconstruction_images=validation_cfg.get(
                "log_reconstruction_images", True
            ),
        )

    raise ValueError(
        f"Unsupported model.type {model_type!r}; expected 'diffusion', "
        "'diffusion_residual', or 'deterministic_residual'"
    )


def main() -> None:
    """Entry point for training with PyTorch Lightning.

    Lightning flow in this script:
    1. Build data module (encapsulates DataLoaders).
    2. Build LightningModule (`PixelDiffusionConditional`).
    3. Build `pl.Trainer` with runtime options.
    4. Call `trainer.fit(model, datamodule=datamodule)` to start the full train/val loop.
    """
    global _ACTIVE_FAILURE_PHASE, _ACTIVE_RUN_DIR

    _ACTIVE_FAILURE_PHASE = "argument_parsing"
    parser = argparse.ArgumentParser()
    # Path to the YAML config used for all runtime settings.
    parser.add_argument("--config", type=str, default="configs/config.yaml")
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
    args = parser.parse_args()
    if args.ckpt_path and args.weights_only_path:
        parser.error("--ckpt-path and --weights-only-path are mutually exclusive")

    args.ckpt_path = _absolute_checkpoint_path(args.ckpt_path)
    args.weights_only_path = _absolute_checkpoint_path(args.weights_only_path)
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
    datamodule = PairedDataModule.from_config(config)
    # Split config sections for clarity.
    wandb_cfg = config.get("logging", {}).get("wandb", {})

    # This is the Lightning model used for training and validation.
    _ACTIVE_FAILURE_PHASE = "model_build"
    model = build_model(config)
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
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename=checkpoint_cfg.get("filename", "epoch={epoch:03d}-step={step}"),
        monitor=checkpoint_cfg.get(
            "monitor", getattr(model, "checkpoint_monitor", "val/loss")
        ),
        mode=checkpoint_cfg.get("mode", getattr(model, "checkpoint_mode", "min")),
        save_top_k=checkpoint_cfg.get("save_top_k", 2),
        save_last=checkpoint_cfg.get("save_last", False),
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpointing_enabled = bool(trainer_cfg.get("enable_checkpointing", True))
    callbacks = [MachineReadableRunCallback(run_dir), lr_monitor]
    if checkpointing_enabled:
        callbacks.append(checkpoint_callback)
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
