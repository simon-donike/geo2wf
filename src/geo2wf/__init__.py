"""Modular training and inference tools for tropical-cyclone wind fields."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data.contracts import DataSpec, WindFieldBatch
    from .models.base import (
        LossOutput,
        PredictionBatch,
        PredictionRequest,
        WindFieldLightningModule,
    )

__all__ = [
    "DataSpec",
    "LossOutput",
    "PredictionBatch",
    "PredictionRequest",
    "WindFieldBatch",
    "WindFieldLightningModule",
]


def __getattr__(name):
    # Acquisition workers import this package on spawn. Keep the training stack
    # out of their startup path unless a caller requests a training contract.
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = (
        ".data.contracts" if name in {"DataSpec", "WindFieldBatch"} else ".models.base"
    )
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
