"""Deprecated compatibility import."""

from importlib import import_module
from warnings import warn

warn("src.PixelDiffusion is deprecated; use geo2wf.models.conditional_diffusion", DeprecationWarning, stacklevel=2)
_module = import_module("geo2wf.models.conditional_diffusion.module")
import sys
sys.modules[__name__] = _module
globals().update(
    {
        name: getattr(_module, name)
        for name in dir(_module)
        if not name.startswith("__")
    }
)
