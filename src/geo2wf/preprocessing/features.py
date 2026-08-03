"""Shared normalization, geometry, and solar feature entry points."""

from geo2wf.data.datasets.paired_geotiff import (
    _normalize as normalize_channels,
    _normalized_distance_to_center as normalized_distance_to_center,
    _solar_time_features as solar_time_features,
)

__all__ = ["normalize_channels", "normalized_distance_to_center", "solar_time_features"]
