"""Structured configuration schemas for shared runtime concerns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckpointConfig:
    monitor: str | None = None
    mode: str = "min"
    save_top_k: int = 2
    save_last: bool = False
    filename: str = "epoch={epoch:03d}-step={step}"


@dataclass
class TrainerConfig:
    max_epochs: int = 1
    accelerator: str = "auto"
    devices: Any = 1
    precision: Any = 32
    log_every_n_steps: int = 10
    enable_checkpointing: bool = True
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    deterministic: bool | None = None
    strategy: str | None = None
    default_root_dir: str = "logs"
    float32_matmul_precision: str = "high"
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "geo2wf"
    name: str | None = None
    group: str | None = None
    job_type: str = "training"
    tags: list[str] = field(default_factory=list)
    log_model: bool = False


@dataclass
class LoggingConfig:
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class WorkspaceConfig:
    seed: int = 42
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
