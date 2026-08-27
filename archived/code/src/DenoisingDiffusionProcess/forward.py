"""Deprecated compatibility import."""

from importlib import import_module
from warnings import warn

warn("Use geo2wf.diffusion.forward_process", DeprecationWarning, stacklevel=2)
_module = import_module("geo2wf.diffusion.forward_process")
import sys
sys.modules[__name__] = _module
globals().update(
    {
        name: getattr(_module, name)
        for name in dir(_module)
        if not name.startswith("__")
    }
)
