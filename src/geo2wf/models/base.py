"""Shared extension contract for wind-field Lightning modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import pytorch_lightning as pl
import torch

from geo2wf.data.contracts import DataSpec, WindFieldBatch, validate_batch


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    components: Mapping[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionRequest:
    ensemble_size: int = 1
    seed: int = 42
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be positive")


@dataclass(frozen=True)
class PredictionBatch:
    """Physical predictions with a consistent [B, E, C, H, W] member axis."""

    samples_physical: torch.Tensor
    central_physical: torch.Tensor
    baseline_physical: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.samples_physical.ndim != 5:
            raise ValueError("samples_physical must have shape [B, E, C, H, W]")
        expected = (self.samples_physical.shape[0], *self.samples_physical.shape[2:])
        if tuple(self.central_physical.shape) != expected:
            raise ValueError("central_physical must have shape [B, C, H, W]")


class WindFieldLightningModule(pl.LightningModule, ABC):
    """Minimal interface a newly added wind-field model must implement."""

    checkpoint_monitor = "val/loss"
    checkpoint_mode = "min"

    @abstractmethod
    def compute_training_objective(self, batch: WindFieldBatch) -> LossOutput:
        raise NotImplementedError

    @abstractmethod
    def predict_batch(
        self, batch: WindFieldBatch, request: PredictionRequest
    ) -> PredictionBatch:
        raise NotImplementedError

    def validate_data_spec(self, spec: DataSpec) -> None:
        expected = getattr(
            self,
            "expected_data_condition_channels",
            getattr(self, "condition_channels", None),
        )
        if expected is not None and int(expected) != spec.condition_channel_count:
            raise ValueError(
                f"model expects {expected} condition channels but data provides "
                f"{spec.condition_channel_count}: {spec.condition_channels}"
            )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint.setdefault(
            "geo2wf",
            {
                "schema_version": 1,
                "model_target": f"{type(self).__module__}.{type(self).__name__}",
                "batch_schema_version": 1,
            },
        )

    def queue_reconstruction_event(self, event: Any) -> None:
        self.__dict__.setdefault("_reconstruction_events", []).append(event)

    def drain_reconstruction_events(self) -> list[Any]:
        events = self.__dict__.get("_reconstruction_events", [])
        self.__dict__["_reconstruction_events"] = []
        return events

    def training_step(self, batch: WindFieldBatch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        validate_batch(batch)
        output = self.compute_training_objective(batch)
        batch_size = int(batch["target"].shape[0])
        self.log(
            "train/loss",
            output.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        for name, value in output.components.items():
            self.log(
                f"train/{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        return output.loss

    def predict_step(
        self, batch: WindFieldBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> PredictionBatch:
        del batch_idx, dataloader_idx
        validate_batch(batch)
        return self.predict_batch(batch, PredictionRequest())
