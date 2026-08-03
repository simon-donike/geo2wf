"""Deprecated compatibility import."""

from importlib import import_module
from warnings import warn

warn("src.ERA5Residual is deprecated; use geo2wf.models.deterministic_residual", DeprecationWarning, stacklevel=2)
_module = import_module("geo2wf.models.deterministic_residual.module")
import sys
sys.modules[__name__] = _module
globals().update(
    {
        name: getattr(_module, name)
        for name in dir(_module)
        if not name.startswith("__")
    }
)
