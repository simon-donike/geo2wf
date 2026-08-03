"""Model-agnostic tracking callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytorch_lightning as pl

from .reconstruction_media import log_wandb_reconstruction
from .run_manifest import MachineReadableRunCallback


@dataclass
class ReconstructionEvent:
    batch: Any
    prediction: Any
    wandb_key: str
    options: dict[str, Any]


class ReconstructionLoggingCallback(pl.Callback):
    """Drain optional standardized reconstruction events emitted by modules."""

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, outputs, batch, batch_idx, dataloader_idx
        drain = getattr(pl_module, "drain_reconstruction_events", None)
        if not callable(drain):
            return
        for event in drain():
            log_wandb_reconstruction(
                pl_module,
                event.batch,
                event.prediction,
                wandb_key=event.wandb_key,
                **event.options,
            )


__all__ = [
    "MachineReadableRunCallback",
    "ReconstructionEvent",
    "ReconstructionLoggingCallback",
]
