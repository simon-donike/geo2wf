"""Dataset implementations and stable data contracts."""

from .contracts import DataSpec, SampleMetadata, WindFieldBatch, validate_batch
from .intensity import IntensityDataSpec, UNetIntensityDataModule, UNetIntensityDataset
from .intensity_forecast import (
    IntensityForecastDataModule,
    IntensityForecastDataSpec,
    IntensityForecastDataset,
)
from .joint_intensity import (
    JointPairedIntensityDataModule,
    JointPairedIntensityDataset,
)

__all__ = [
    "DataSpec",
    "IntensityDataSpec",
    "IntensityForecastDataModule",
    "IntensityForecastDataSpec",
    "IntensityForecastDataset",
    "JointPairedIntensityDataModule",
    "JointPairedIntensityDataset",
    "SampleMetadata",
    "UNetIntensityDataModule",
    "UNetIntensityDataset",
    "WindFieldBatch",
    "validate_batch",
]
