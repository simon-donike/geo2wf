from __future__ import annotations

import inspect

import pandas as pd
import torch
from matplotlib import pyplot as plt

from data.dataset import (
    _manifest_ibtracs_center,
    _append_era5_derived_channels,
    _cell_centers,
    _json_list,
    _normalize,
    _paired_random_flips,
    _resize_target,
    _row_float,
    _row_value,
)
from utils.plotting import (
    IBTRACS_CENTER_COLUMNS,
    plot_random_geo_sar_pairs,
    plot_validation_reconstruction_batch,
)


def test_resize_target_resizes_values_and_mask_with_expected_modes() -> None:
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.tensor([[[True, False], [True, True]]])

    resized_target, resized_mask = _resize_target(target, mask, (4, 4))

    assert resized_target.shape == (1, 4, 4)
    assert resized_mask.shape == (1, 4, 4)
    assert resized_mask.dtype == torch.bool
    assert torch.all(resized_target[~resized_mask] == 0)


def test_paired_random_flips_applies_same_spatial_flips_to_all_tensors(monkeypatch) -> None:
    values = iter([torch.tensor(0.25), torch.tensor(0.75)])
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: next(values))
    condition = torch.arange(4).view(1, 2, 2)
    target = condition + 10
    condition_mask = torch.tensor([[[True, False], [False, True]]])
    target_mask = ~condition_mask

    flipped_condition, flipped_target, flipped_condition_mask, flipped_target_mask = (
        _paired_random_flips(condition, target, condition_mask, target_mask)
    )

    assert torch.equal(flipped_condition, torch.flip(condition, dims=[-1]))
    assert torch.equal(flipped_target, torch.flip(target, dims=[-1]))
    assert torch.equal(flipped_condition_mask, torch.flip(condition_mask, dims=[-1]))
    assert torch.equal(flipped_target_mask, torch.flip(target_mask, dims=[-1]))


def test_append_era5_derived_channels_is_noop_without_wind_components() -> None:
    tensor = torch.ones(1, 2, 2)
    channels = ["era5_temperature_2m"]

    derived, derived_channels = _append_era5_derived_channels(
        tensor,
        channels,
        torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64),
    )

    assert derived is tensor
    assert derived_channels is channels


def test_normalize_uses_band_fallback_and_clamps_to_unit_interval() -> None:
    tensor = torch.tensor([[[-1.0, 5.0, 11.0]]])

    normalized = _normalize(
        tensor,
        "geo",
        ["unnamed"],
        {"channels": {"geo": {"band_0": {"min": 0.0, "max": 10.0}}}},
    )

    assert torch.allclose(normalized, torch.tensor([[[0.0, 0.5, 1.0]]]))


def test_cell_centers_returns_midpoints_between_bounds() -> None:
    centers = _cell_centers(10.0, 14.0, 4)

    assert centers.tolist() == [10.5, 11.5, 12.5, 13.5]


def test_json_list_accepts_native_lists_and_json_strings() -> None:
    assert _json_list(["a", 2]) == ["a", "2"]
    assert _json_list('["x", "y"]') == ["x", "y"]


def test_row_helpers_handle_primary_fallback_and_missing_floats() -> None:
    row = pd.Series(
        {
            "primary": "",
            "fallback": "value",
            "finite": "3.5",
            "missing": "",
            "not_finite": "nan",
        }
    )

    assert _row_value(row, "primary", row["fallback"]) == "value"
    assert _row_value(row, "present", "fallback") == "fallback"
    assert _row_float(row, "finite") == 3.5
    assert torch.isnan(torch.tensor(_row_float(row, "missing")))
    assert torch.isnan(torch.tensor(_row_float(row, "not_finite")))


def test_manifest_ibtracs_center_ignores_image_center_columns() -> None:
    row = pd.Series(
        {
            "center_lat": "10.0",
            "center_lon": "20.0",
            "ibtracs_center_lat": "11.5",
            "ibtracs_center_lon": "22.5",
        }
    )

    center = _manifest_ibtracs_center(row)

    assert torch.equal(center, torch.tensor([11.5, 22.5], dtype=torch.float64))


def test_random_pair_plot_defaults_to_ibtracs_manifest_columns() -> None:
    default = inspect.signature(plot_random_geo_sar_pairs).parameters[
        "center_columns"
    ].default

    assert default == IBTRACS_CENTER_COLUMNS


def test_validation_plot_adds_era5_wind_map_only_when_available() -> None:
    base_sample = {
        "condition": torch.stack(
            [
                torch.full((4, 4), 0.2),
                torch.full((4, 4), 0.4),
                torch.full((4, 4), 0.6),
            ]
        ),
        "prediction": torch.full((1, 4, 4), 0.3),
        "target": torch.full((1, 4, 4), 0.5),
        "condition_mask": torch.ones((1, 4, 4), dtype=torch.bool),
        "target_mask": torch.ones((1, 4, 4), dtype=torch.bool),
        "condition_channels": ["CMI_C13", "CMI_C14", "CMI_C08"],
        "condition_bounds": [-2.0, 2.0, -2.0, 2.0],
        "target_bounds": [-1.0, 1.0, -1.0, 1.0],
        "center": (0.5, -0.5),
    }

    without_era = plot_validation_reconstruction_batch([base_sample])
    without_titles = [axis.get_title() for axis in without_era.axes]
    assert "ERA5 10 m wind speed" not in without_titles

    with_era_sample = dict(base_sample)
    with_era_sample["condition"] = torch.cat(
        [base_sample["condition"], torch.full((1, 4, 4), 0.5)]
    )
    with_era_sample["condition_channels"] = [
        *base_sample["condition_channels"],
        "era5_wind_speed_10m",
    ]
    with_era = plot_validation_reconstruction_batch([with_era_sample])
    axes_by_title = {axis.get_title(): axis for axis in with_era.axes}
    wind_axis = axes_by_title["ERA5 10 m wind speed"]

    assert float(wind_axis.images[-1].get_array().mean()) == 42.5
    assert any(
        collection.get_label() == "IBTrACS center"
        for collection in wind_axis.collections
    )
    plt.close(without_era)
    plt.close(with_era)


def test_validation_plot_prefers_explicit_physical_era5_wind() -> None:
    sample = {
        "condition": torch.full((1, 4, 4), 0.5),
        "prediction": torch.full((1, 4, 4), 0.3),
        "target": torch.full((1, 4, 4), 0.5),
        "condition_mask": torch.ones((1, 4, 4), dtype=torch.bool),
        "target_mask": torch.ones((1, 4, 4), dtype=torch.bool),
        "condition_channels": ["era5_wind_speed_10m"],
        "condition_bounds": [-2.0, 2.0, -2.0, 2.0],
        "target_bounds": [-2.0, 2.0, -2.0, 2.0],
        "era5_wind_speed_physical": torch.full((1, 4, 4), 20.0),
        "era5_wind_speed_mask": torch.ones((1, 4, 4), dtype=torch.bool),
    }

    figure = plot_validation_reconstruction_batch([sample])
    wind_axis = {
        axis.get_title(): axis for axis in figure.axes
    }["ERA5 10 m wind speed"]

    assert float(wind_axis.images[-1].get_array().mean()) == 20.0
    plt.close(figure)
