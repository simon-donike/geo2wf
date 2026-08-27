from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from geo2wf.data.datamodule import PairedDataModule
from geo2wf.data.datasets.paired_geotiff import (
    EARTH_RADIUS_M,
    ERA5_RELATIVE_VORTICITY_10M,
    ERA5_WIND_SPEED_10M,
    PairedImageDataset,
    _append_era5_derived_channels,
    _cell_centers,
    _manifest_era5_time_gap_hours,
    _normalize,
    _relative_vorticity_10m,
)


def test_append_era5_derived_channels_adds_wind_speed_and_vorticity() -> None:
    bounds = torch.tensor([-1.5, 1.5, -1.5, 1.5], dtype=torch.float64)
    u10 = torch.full((4, 4), 3.0)
    v10 = torch.full((4, 4), 4.0)
    tensor = torch.stack([u10, v10])
    channels = ["era5_u_wind_10m", "era5_v_wind_10m"]

    derived, derived_channels = _append_era5_derived_channels(
        tensor, channels, bounds
    )

    assert derived.shape == (4, 4, 4)
    assert derived_channels == [
        "era5_u_wind_10m",
        "era5_v_wind_10m",
        ERA5_WIND_SPEED_10M,
        ERA5_RELATIVE_VORTICITY_10M,
    ]
    assert torch.allclose(derived[2], torch.full((4, 4), 5.0))
    assert torch.allclose(derived[3], torch.zeros((4, 4)), atol=1e-7)


def test_relative_vorticity_10m_uses_meter_spacing() -> None:
    omega = 1.0e-4
    bounds = torch.tensor([-0.25, 0.25, -0.25, 0.25], dtype=torch.float64)
    lat = np.deg2rad(_cell_centers(0.25, -0.25, 5))
    lon = np.deg2rad(_cell_centers(-0.25, 0.25, 5))
    x = EARTH_RADIUS_M * lon[None, :]
    y = EARTH_RADIUS_M * lat[:, None]
    u10 = torch.from_numpy((-omega * y).astype(np.float32)).expand(5, 5)
    v10 = torch.from_numpy((omega * x).astype(np.float32)).expand(5, 5)

    vorticity = _relative_vorticity_10m(u10, v10, bounds)

    interior = vorticity[1:-1, 1:-1]
    assert torch.allclose(
        interior, torch.full_like(interior, 2.0 * omega), atol=2e-8, rtol=1e-3
    )


def test_normalize_uses_default_stats_for_derived_era5_channels() -> None:
    tensor = torch.tensor([[[42.5]], [[0.0]]])

    normalized = _normalize(
        tensor,
        "era5",
        [ERA5_WIND_SPEED_10M, ERA5_RELATIVE_VORTICITY_10M],
        {"channels": {"era5": {}}},
    )

    assert torch.allclose(normalized, torch.full_like(normalized, 0.5))


def test_dataset_filters_samples_without_era5_when_required(tmp_path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    pd.DataFrame(
        {
            "sample_id": ["modern", "legacy", "other-context", "missing"],
            "context_path": [
                "train/modern_era5.tif",
                "",
                "train/other_context.tif",
                "",
            ],
            "context_source_type": ["era5", "", "ocean", ""],
            "era5_path": ["", "train/legacy_era5.tif", "", ""],
        }
    ).to_csv(split_dir / "manifest.csv", index=False)
    (tmp_path / "stats.json").write_text("{}", encoding="utf-8")

    dataset = PairedImageDataset(tmp_path, "train", require_era5=True)

    assert dataset.samples["sample_id"].tolist() == ["modern", "legacy"]
    assert dataset.manifest_sample_count == 4
    assert dataset.filtered_missing_era5_count == 2


def test_dataset_keeps_samples_without_era5_when_not_required(tmp_path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    pd.DataFrame(
        {
            "sample_id": ["with-era5", "without-era5"],
            "context_path": ["train/context.tif", ""],
        }
    ).to_csv(split_dir / "manifest.csv", index=False)
    (tmp_path / "stats.json").write_text("{}", encoding="utf-8")

    dataset = PairedImageDataset(tmp_path, "train")

    assert dataset.samples["sample_id"].tolist() == [
        "with-era5",
        "without-era5",
    ]
    assert dataset.filtered_missing_era5_count == 0


def test_datamodule_enables_era5_filter_from_export_config() -> None:
    datamodule = PairedDataModule.from_config(
        {
            "export": {"include_era5": True},
            "data": {"root": "custom/data"},
        }
    )

    assert datamodule.require_era5 is True


def test_dataset_filters_stale_or_unverifiable_era5_contexts(tmp_path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    pd.DataFrame(
        {
            "sample_id": ["fresh", "stale", "unknown"],
            "context_path": ["era5.tif", "era5.tif", "era5.tif"],
            "context_source_type": ["era5", "era5", "era5"],
            "era5_timestamp": [
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "",
            ],
            "target_timestamp": [
                "2025-01-01T02:00:00Z",
                "2025-01-01T10:00:00Z",
                "2025-01-01T01:00:00Z",
            ],
        }
    ).to_csv(split_dir / "manifest.csv", index=False)
    (tmp_path / "stats.json").write_text("{}", encoding="utf-8")

    dataset = PairedImageDataset(
        tmp_path,
        "train",
        require_era5=True,
        max_era5_time_gap_hours=3.1,
    )

    assert dataset.samples["sample_id"].tolist() == ["fresh"]
    assert dataset.filtered_stale_era5_count == 2


def test_era5_age_uses_condition_timestamp_before_target_timestamp() -> None:
    samples = pd.DataFrame(
        {
            "era5_timestamp": ["2025-01-01T00:00:00Z"],
            "condition_timestamp": ["2025-01-01T02:00:00Z"],
            "target_timestamp": ["2025-01-01T10:00:00Z"],
        }
    )

    assert _manifest_era5_time_gap_hours(samples).iloc[0] == 2.0
