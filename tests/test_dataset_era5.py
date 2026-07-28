from __future__ import annotations

import numpy as np
import torch

from data.dataset import (
    EARTH_RADIUS_M,
    ERA5_RELATIVE_VORTICITY_10M,
    ERA5_WIND_SPEED_10M,
    _append_era5_derived_channels,
    _cell_centers,
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
