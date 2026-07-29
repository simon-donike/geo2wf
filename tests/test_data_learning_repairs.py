from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from torch.utils.data import SequentialSampler, TensorDataset

from data.datamodule import PairedDataModule, _storm_stratified_indices
from data.dataset import (
    ERA5_RELATIVE_VORTICITY_10M,
    ERA5_WIND_SPEED_10M,
    PairedImageDataset,
    _append_era5_derived_channels,
    _denormalize,
    _normalize,
    _normalization_affine_parameters,
    _paired_random_flips,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_geo_sar_geotiffs import (  # noqa: E402
    EARTH_RADIUS_M,
    StatsAccumulator,
    _native_relative_vorticity_10m,
    _parse_args as _parse_export_args,
    _regrid_continuous,
)


@pytest.mark.parametrize(
    ("flip_state", "negated_indices"),
    [
        ((True, False), {0, 3}),
        ((False, True), {1, 3}),
        ((True, True), {0, 1}),
    ],
)
def test_era5_flip_physics_preserves_vectors_and_vorticity_parity(
    flip_state: tuple[bool, bool], negated_indices: set[int]
) -> None:
    channels = [
        "era5_u_wind_10m",
        "era5_v_wind_10m",
        ERA5_WIND_SPEED_10M,
        ERA5_RELATIVE_VORTICITY_10M,
        "era5_temperature_2m",
    ]
    condition = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.2, 0.3], [0.4, 0.5]],
            [[0.3, 0.4], [0.5, 0.6]],
            [[0.4, 0.5], [0.6, 0.7]],
            [[0.5, 0.6], [0.7, 0.8]],
        ]
    )
    target = torch.arange(4, dtype=torch.float32).view(1, 2, 2)
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    dims = ([-1] if flip_state[0] else []) + ([-2] if flip_state[1] else [])

    flipped, flipped_target, _, _ = _paired_random_flips(
        condition,
        target,
        mask,
        mask,
        condition_channels=channels,
        condition_zero_values=torch.full((len(channels),), 0.5),
        flip_state=flip_state,
    )

    expected = torch.flip(condition, dims=dims)
    for index in negated_indices:
        expected[index] = 1.0 - expected[index]
    assert torch.allclose(flipped, expected)
    assert torch.equal(flipped_target, torch.flip(target, dims=dims))


def test_era5_flip_uses_channel_validity_not_aggregate_geo_mask() -> None:
    condition = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.7, 0.8], [0.9, 0.6]],
        ]
    )
    aggregate_mask = torch.zeros(1, 2, 2, dtype=torch.bool)
    channel_mask = torch.stack(
        [aggregate_mask[0], torch.ones(2, 2, dtype=torch.bool)]
    )

    flipped, _, _, _ = _paired_random_flips(
        condition,
        torch.zeros(1, 2, 2),
        aggregate_mask,
        aggregate_mask,
        condition_channels=["CMI_C13", "era5_u_wind_10m"],
        condition_zero_values=torch.tensor([0.5, 0.5]),
        condition_channel_mask=channel_mask,
        flip_state=(True, False),
    )

    assert torch.allclose(flipped[1], 1.0 - torch.flip(condition[1], dims=[-1]))


def test_legacy_era5_vorticity_is_neutral_not_a_block_edge_derivative() -> None:
    u10 = torch.tensor([[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]])
    v10 = torch.zeros_like(u10)

    derived, channels = _append_era5_derived_channels(
        torch.stack([u10, v10]),
        ["era5_u_wind_10m", "era5_v_wind_10m"],
        torch.tensor([0.0, 3.0, 0.0, 2.0], dtype=torch.float64),
    )

    assert channels[-1] == ERA5_RELATIVE_VORTICITY_10M
    assert torch.count_nonzero(derived[-1]) == 0


def test_robust_normalization_uses_train_stats_and_round_trips() -> None:
    stats = {
        "normalization": "min-max",
        "channels": {
            "sar": {
                "wind_speed": {
                    "min": 0.0,
                    "max": 80.0,
                    "mean": 20.0,
                    "std": 5.0,
                }
            }
        },
    }
    physical = torch.tensor([[[0.0, 20.0, 40.0]]])

    normalized = _normalize(
        physical,
        "sar",
        ["wind_speed"],
        stats,
        normalization="robust-zscore",
        robust_clip=4.0,
    )
    reconstructed = _denormalize(
        normalized,
        "sar",
        ["wind_speed"],
        stats,
        normalization="robust-zscore",
        robust_clip=4.0,
    )
    offset, scale = _normalization_affine_parameters(
        "sar",
        ["wind_speed"],
        stats,
        normalization="robust-zscore",
        robust_clip=4.0,
    )

    assert torch.allclose(offset, torch.tensor([0.0]))
    assert torch.allclose(scale, torch.tensor([40.0]))
    assert torch.allclose(normalized, torch.tensor([[[0.0, 0.5, 1.0]]]))
    assert torch.allclose(reconstructed, physical)


def test_robust_derived_era5_speed_uses_sar_target_scale() -> None:
    stats = {
        "channels": {
            "era5": {
                ERA5_WIND_SPEED_10M: {"median": 8.0, "robust_scale": 2.0}
            },
            "sar": {"wind_speed": {"median": 20.0, "robust_scale": 5.0}},
        }
    }

    normalized = _normalize(
        torch.tensor([[[20.0]]]),
        "era5",
        [ERA5_WIND_SPEED_10M],
        stats,
        normalization="robust-zscore",
    )

    assert torch.allclose(normalized, torch.tensor([[[0.5]]]))


def test_continuous_regrid_is_bilinear_for_a_planar_field() -> None:
    lat, lon = np.meshgrid([0.0, 1.0], [0.0, 1.0], indexing="ij")
    values = 2.0 * lat + 3.0 * lon
    grid_lat = np.array([[0.25, 0.5, 0.75]])
    grid_lon = np.array([[0.25, 0.5, 0.75]])

    regridded, mask = _regrid_continuous(
        values, lat, lon, grid_lat, grid_lon
    )

    assert mask.all()
    assert np.allclose(regridded, 2.0 * grid_lat + 3.0 * grid_lon)


def test_native_grid_vorticity_recovers_solid_body_rotation() -> None:
    omega = 1.0e-4
    lat_axis = np.linspace(0.25, -0.25, 7)
    lon_axis = np.linspace(-0.25, 0.25, 7)
    lon, lat = np.meshgrid(lon_axis, lat_axis)
    x = EARTH_RADIUS_M * np.deg2rad(lon)
    y = EARTH_RADIUS_M * np.deg2rad(lat)
    u10 = -omega * y
    v10 = omega * x

    vorticity = _native_relative_vorticity_10m(u10, v10, lat, lon)

    assert np.allclose(
        vorticity[1:-1, 1:-1], 2.0 * omega, atol=2e-8, rtol=1e-3
    )


def test_export_stats_include_bounded_robust_train_statistics() -> None:
    accumulator = StatsAccumulator.create(robust_sample_size=2_000, seed=3)
    values = np.arange(1_000, dtype=np.float32).reshape(20, 50)
    accumulator.update("sar", "wind_speed", values, np.ones_like(values, bool))

    channel_stats = accumulator.to_jsonable()["channels"]["sar"]["wind_speed"]

    assert channel_stats["median"] == pytest.approx(499.5)
    assert channel_stats["q25"] == pytest.approx(249.75)
    assert channel_stats["q75"] == pytest.approx(749.25)
    assert channel_stats["robust_scale"] > 0


def test_dataset_exposes_physical_target_and_target_scaled_era5_anchor(
    tmp_path,
) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    transform = from_origin(0.0, 2.0, 1.0, 1.0)
    _write_test_tiff(split_dir / "condition.tif", np.full((1, 2, 2), 250.0), transform)
    _write_test_tiff(
        split_dir / "era5.tif",
        np.stack([np.full((2, 2), 3.0), np.full((2, 2), 4.0)]),
        transform,
    )
    _write_test_tiff(
        split_dir / "target.tif",
        np.array([[[1.0, 2.0], [3.0, np.nan]]]),
        transform,
    )
    pd.DataFrame(
        [
            {
                "sample_id": "sample",
                "storm_id": "storm",
                "condition_path": "train/condition.tif",
                "context_path": "train/era5.tif",
                "target_path": "train/target.tif",
                "condition_source_type": "geo",
                "context_source_type": "era5",
                "target_source_type": "sar",
                "condition_channels": json.dumps(["CMI_C13"]),
                "context_channels": json.dumps(
                    ["era5_u_wind_10m", "era5_v_wind_10m"]
                ),
                "target_channels": json.dumps(["wind_speed"]),
                "condition_sensor": "ABI",
                "target_sensor": "SAR",
                "dt_minutes": 0.0,
                "ibtracs_center_lat": 1.0,
                "ibtracs_center_lon": 1.0,
            }
        ]
    ).to_csv(split_dir / "manifest.csv", index=False)
    stats = {
        "normalization": "min-max",
        "channels": {
            "geo": {
                "CMI_C13": {
                    "min": 200.0,
                    "max": 300.0,
                    "mean": 250.0,
                    "std": 10.0,
                }
            },
            "era5": {
                "era5_u_wind_10m": {
                    "min": -10.0,
                    "max": 10.0,
                    "mean": 0.0,
                    "std": 5.0,
                },
                "era5_v_wind_10m": {
                    "min": -10.0,
                    "max": 10.0,
                    "mean": 0.0,
                    "std": 5.0,
                },
            },
            "sar": {
                "wind_speed": {
                    "min": 0.0,
                    "max": 10.0,
                    "mean": 5.0,
                    "std": 2.0,
                }
            },
        },
    }
    (tmp_path / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

    sample = PairedImageDataset(tmp_path, "train", target_size=(2, 2))[0]

    assert sample["condition"].shape == (5, 2, 2)
    assert torch.allclose(sample["condition"][3], torch.full((2, 2), 5.0 / 85.0))
    assert torch.allclose(sample["condition"][4], torch.full((2, 2), 0.5))
    assert torch.allclose(sample["target_norm_offset"], torch.tensor([0.0]))
    assert torch.allclose(sample["target_norm_scale"], torch.tensor([10.0]))
    assert torch.allclose(
        sample["target_physical"], torch.tensor([[[1.0, 2.0], [3.0, 0.0]]])
    )
    assert torch.allclose(
        sample["era5_wind_speed_physical"], torch.full((1, 2, 2), 5.0)
    )
    assert torch.allclose(
        sample["era5_wind_speed"], torch.full((1, 2, 2), 0.5)
    )
    assert sample["era5_wind_speed_mask"].dtype == torch.bool
    assert sample["era5_wind_speed_mask"].all()

    separate_target = PairedImageDataset(
        tmp_path,
        "train",
        target_size=(2, 2),
        normalization="robust-zscore",
        target_normalization="min-max",
    )[0]
    assert torch.allclose(
        separate_target["target"],
        torch.tensor([[[0.1, 0.2], [0.3, 0.0]]]),
    )
    assert torch.allclose(
        separate_target["era5_wind_speed"], torch.full((1, 2, 2), 0.5)
    )
    assert torch.allclose(
        separate_target["target_norm_offset"], torch.tensor([0.0])
    )
    assert torch.allclose(
        separate_target["target_norm_scale"], torch.tensor([10.0])
    )


def test_validation_and_fixed_train_preview_loaders_are_deterministic(tmp_path) -> None:
    datamodule = PairedDataModule(root=tmp_path, batch_size=2)
    datamodule.train_dataset = TensorDataset(torch.arange(6))
    datamodule.val_dataset = TensorDataset(torch.arange(4))

    validation, preview = datamodule.val_dataloader()

    assert isinstance(validation.sampler, SequentialSampler)
    assert isinstance(preview.sampler, SequentialSampler)
    assert list(preview.dataset.indices) == [0, 1]
    assert torch.equal(next(iter(preview))[0], torch.tensor([0, 1]))


def test_validation_prefix_round_robins_across_storms() -> None:
    class _Dataset:
        samples = pd.DataFrame(
            {
                "storm_id": ["A", "A", "A", "B", "B", "C"],
            }
        )

        def __len__(self):
            return len(self.samples)

    assert _storm_stratified_indices(_Dataset()) == [0, 3, 5, 1, 4, 2]


def test_datamodule_supports_separate_target_normalization() -> None:
    datamodule = PairedDataModule.from_config(
        {
            "data": {
                "root": "custom/data",
                "normalization": "robust-zscore",
                "robust_clip": 4.0,
                "target_normalization": "min-max",
            }
        }
    )

    assert datamodule.robust_clip == 4.0
    assert datamodule.normalization == "robust-zscore"
    assert datamodule.target_normalization == "min-max"


def test_explicit_export_output_root_wins_machine_default(
    tmp_path, monkeypatch
) -> None:
    desired = tmp_path / "experiment_data"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"export:\n  output_root: {desired}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GEO_SAR_OUTPUT_ROOT", str(tmp_path / "generic_data"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_geo_sar_geotiffs.py", "--config", str(config_file)],
    )

    assert _parse_export_args().output_root == desired


def _write_test_tiff(path, values: np.ndarray, transform) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[1],
        width=values.shape[2],
        count=values.shape[0],
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.write(values.astype(np.float32))
