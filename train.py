from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import WandbLogger

from data import PairedDataModule
from src.PixelDiffusion import PixelDiffusionConditional


def load_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    pl.seed_everything(config.get("seed", 42), workers=True)

    datamodule = PairedDataModule.from_config(config)
    datamodule.setup("fit")
    train_ds = datamodule.train_dataset
    val_ds = datamodule.val_dataset

    model_cfg = config.get("model", {})
    opt_cfg = config.get("optimization", {})
    lr_sched_cfg = opt_cfg.get("reduce_lr_on_plateau", {})
    loader_cfg = config.get("data", {}).get("loader", {})
    unet_cfg = model_cfg.get("unet", {})
    wandb_cfg = config.get("logging", {}).get("wandb", {})

    model = PixelDiffusionConditional(
        train_dataset=train_ds,
        valid_dataset=val_ds,
        condition_channels=model_cfg.get("in_channels", 2),
        generated_channels=model_cfg.get("out_channels", 2),
        num_timesteps=model_cfg.get("num_timesteps", 1000),
        schedule=model_cfg.get("schedule", "linear"),
        model_dim=unet_cfg.get("dim", 64),
        model_dim_mults=tuple(unet_cfg.get("dim_mults", [1, 2, 4, 8])),
        model_channels=unet_cfg.get("channels"),
        model_out_dim=unet_cfg.get("out_dim"),
        batch_size=loader_cfg.get("batch_size", 4),
        lr=opt_cfg.get("lr", 1e-3),
        lr_scheduler_factor=lr_sched_cfg.get("factor", 0.5),
        lr_scheduler_patience=lr_sched_cfg.get("patience", 10),
    )

    trainer_cfg = config.get("trainer", {})
    wandb_logger = WandbLogger(
        project=wandb_cfg.get("project", "dif_img_rec"),
        name=wandb_cfg.get("name"),
        save_dir=wandb_cfg.get("save_dir", "logs"),
        log_model=wandb_cfg.get("log_model", False),
    )
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 1),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", 32),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
        enable_checkpointing=trainer_cfg.get("enable_checkpointing", False),
        logger=wandb_logger,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
