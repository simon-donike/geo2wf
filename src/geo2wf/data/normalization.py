"""Normalization policies shared by datasets and raw-data inference."""

from .datasets.paired_geotiff import (
    DEFAULT_ROBUST_CLIP,
    NORMALIZATION_MIN_MAX,
    NORMALIZATION_ROBUST_ZSCORE,
    _denormalize as denormalize,
    _normalization_affine_parameters as normalization_affine_parameters,
    _normalize as normalize,
    _normalized_physical_zero as normalized_physical_zero,
)

__all__ = [
    "DEFAULT_ROBUST_CLIP",
    "NORMALIZATION_MIN_MAX",
    "NORMALIZATION_ROBUST_ZSCORE",
    "denormalize",
    "normalization_affine_parameters",
    "normalize",
    "normalized_physical_zero",
]
