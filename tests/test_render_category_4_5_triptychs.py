from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts.render_category_4_5_triptychs import (
    choose_render_rows,
    geostationary_channel_index,
    render_triptych,
    select_category_rows,
)


def _category_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["cat5", "cat3", "cat4"],
            "storm_id": ["WP012025", "EP012025", "AL012025"],
            "observation_timestamp": [
                "2025-08-03T00:00:00Z",
                "2025-08-01T00:00:00Z",
                "2025-08-02T00:00:00Z",
            ],
            "target_wind_ms": [72.0, 52.0, 60.0],
            "target_category": [5, 3, 4],
            "center_lat": [0.0, 0.0, 0.0],
            "center_lon": [-140.0, -130.0, -120.0],
        }
    )


def test_select_category_rows_filters_and_sorts_cat_4_5() -> None:
    selected = select_category_rows(_category_frame(), [5, 4], expected_count=2)

    assert selected["sample_id"].tolist() == ["cat4", "cat5"]
    assert selected["target_category"].tolist() == [4, 5]
    assert str(selected["observation_timestamp"].dt.tz) == "UTC"


def test_select_category_rows_enforces_expected_count() -> None:
    with pytest.raises(ValueError, match="yielded 2 rows, expected 25"):
        select_category_rows(_category_frame(), [4, 5], expected_count=25)


def test_select_category_rows_excludes_land_centered_occurrences() -> None:
    frame = _category_frame()
    frame.loc[frame["sample_id"] == "cat4", ["center_lat", "center_lon"]] = [
        40.7128,
        -74.006,
    ]

    selected = select_category_rows(frame, [4, 5], ocean_only=True)

    assert selected["sample_id"].tolist() == ["cat5"]


def test_choose_render_rows_keeps_strongest_occurrence_per_storm() -> None:
    frame = pd.concat(
        [
            _category_frame(),
            pd.DataFrame(
                {
                    "sample_id": ["cat4-stronger"],
                    "storm_id": ["AL012025"],
                    "observation_timestamp": pd.to_datetime(
                        ["2025-08-04T00:00:00Z"], utc=True
                    ),
                    "target_wind_ms": [75.0],
                    "target_category": [5],
                    "center_lat": [0.0],
                    "center_lon": [-120.0],
                }
            ),
        ],
        ignore_index=True,
    )
    frame["observation_timestamp"] = pd.to_datetime(
        frame["observation_timestamp"], utc=True
    )

    selected = choose_render_rows(
        frame,
        strategy="strongest-per-storm",
        limit=2,
        expected_count=2,
    )

    assert selected["sample_id"].tolist() == ["cat4-stronger", "cat3"]


def test_geostationary_channel_index_accepts_common_aliases() -> None:
    channels = ["CMI_C07", "CMI_C13", "era5_sst"]

    assert geostationary_channel_index(channels, "C13") == 1
    assert geostationary_channel_index(channels, "B13") == 1
    assert geostationary_channel_index(channels, "ABI_C13") == 1


def test_render_triptych_is_horizontal_and_has_three_data_panels() -> None:
    condition = np.stack([np.linspace(0.0, 1.0, 16).reshape(4, 4), np.ones((4, 4))])
    target = np.arange(16, dtype=float).reshape(1, 4, 4)
    mask = np.ones((1, 4, 4), dtype=bool)
    figure = render_triptych(
        condition=condition,
        condition_mask=mask,
        condition_channels=["CMI_C13", "distance_to_center"],
        target=target,
        target_mask=mask,
        prediction=target + 1.0,
        bounds=np.array([-2.0, 2.0, -2.0, 2.0]),
        center=np.array([0.0, 0.0]),
        category_row={
            "sample_id": "AL012025_example",
            "storm_id": "AL012025",
            "observation_timestamp": "2025-08-02T00:00:00Z",
            "target_wind_ms": 60.0,
            "target_category": 4,
            "geo_sensor": "ABI",
        },
        geo_channel="C13",
    )
    try:
        width, height = figure.get_size_inches()
        assert width > 2.0 * height
        assert [axis.get_title() for axis in figure.axes[:3]] == [
            "(a) Geostationary ABI C13",
            "(b) SAR observed wind field",
            "(c) Best-model prediction",
        ]
    finally:
        plt.close(figure)
