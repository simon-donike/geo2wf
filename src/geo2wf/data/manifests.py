"""Manifest predicates and metadata extraction."""

from .datasets.paired_geotiff import (
    _manifest_era5_time_gap_hours as era5_time_gap_hours,
    _manifest_has_era5 as has_era5,
    _manifest_has_pmw as has_pmw,
    _manifest_pmw_time_gap_hours as pmw_time_gap_hours,
)

__all__ = ["era5_time_gap_hours", "has_era5", "has_pmw", "pmw_time_gap_hours"]
