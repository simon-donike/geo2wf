"""Deprecated compatibility entry point for geo2wf training."""

from importlib import import_module
import sys
from warnings import warn

warn(
    "train.py is deprecated as an import; use geo2wf.training or geo2wf-train",
    DeprecationWarning,
    stacklevel=2,
)
_module = import_module("geo2wf.training")
sys.modules[__name__] = _module
globals().update(
    {name: getattr(_module, name) for name in dir(_module) if not name.startswith("__")}
)

if __name__ == "__main__":
    _module._entrypoint()
