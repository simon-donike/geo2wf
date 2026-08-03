"""GeoTIFF tensor reads and mask-preserving resizing."""

from .datasets.paired_geotiff import (
    _read_geotiff as read_geotiff,
    _resize_target as resize_masked_target,
)

__all__ = ["read_geotiff", "resize_masked_target"]
