import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.render_native_storm_gif import (  # noqa: E402
    PMW_HIGH,
    PMW_LOW,
    PMW_MID,
    TMAX,
    TMIN,
    dense_pmw_field,
    dense_pmw_table,
    fit_field_to_target_footprint,
    mark_center,
    pmw_panel,
    recenter_geolocation,
    synthetic_pmw_grid,
)
import scripts.render_native_storm_gif as renderer  # noqa: E402


def test_pmw_panel_uses_purple_to_yellow_palette() -> None:
    field = np.empty((256, 256), dtype=np.float32)
    field[:, :85] = TMIN
    field[:, 85:171] = (TMIN + TMAX) / 2
    field[:, 171:] = TMAX

    image = np.asarray(pmw_panel(field, np.ones_like(field, dtype=bool)))

    assert np.array_equal(image[128, 42], PMW_LOW)
    assert np.array_equal(image[128, 128], PMW_MID)
    assert np.array_equal(image[128, 213], PMW_HIGH)


def test_prediction_can_fill_unchanged_geo_footprint() -> None:
    field = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    target_lat, target_lon = np.meshgrid(
        np.linspace(10.0, 15.0, 6), np.linspace(-60.0, -53.0, 8), indexing="ij"
    )

    resized, valid = fit_field_to_target_footprint(field, target_lat, target_lon)

    assert resized.shape == target_lat.shape
    assert valid.all()
    assert resized[0, 0] == pytest.approx(0.0)
    assert resized[-1, -1] == pytest.approx(3.0)


def test_prediction_center_marker_can_be_white() -> None:
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0.0, 2.0, 11), np.linspace(0.0, 2.0, 11), indexing="ij"
    )

    marked = mark_center(
        Image.new("RGB", (11, 11)), grid_lat, grid_lon, (1.0, 1.0), color=(255, 255, 255)
    )

    assert marked.getpixel((5, 5)) == (255, 255, 255)


def test_dense_pmw_loader_reads_sharded_synthetic_layout(tmp_path: Path) -> None:
    storm = "EP182023"
    tensor_path = tmp_path / "observations" / storm / "frame.pt"
    tensor_path.parent.mkdir(parents=True)
    torch.save(torch.stack([torch.full((2, 3), value) for value in range(4)]), tensor_path)
    manifest_path = tmp_path / "index-files" / "shards" / f"{storm}.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp": "2023-10-25T00:00:00+00:00",
                "path": str(tensor_path.relative_to(tmp_path)),
                "variables": json.dumps(
                    ["TB_36.5H", "TB_36.5V", "TB_A89.0H", "TB_A89.0V"]
                ),
                "resolution": json.dumps([[12, 7], [12, 7], [5, 3], [5, 3]]),
                "ibtracs_center_lat": 10.0,
                "ibtracs_center_lon": -50.0,
            }
        ]
    ).to_csv(manifest_path, index=False)
    sidecar_path = tmp_path / "geolocation" / storm / "frame.npz"
    sidecar_path.parent.mkdir(parents=True)
    expected_lat = np.array([[11.0, 11.1, 11.2], [10.0, 10.1, 10.2]])
    expected_lon = np.array([[-51.0, -50.0, -49.0], [-50.9, -49.9, -48.9]])
    np.savez_compressed(
        sidecar_path,
        grid_lat=expected_lat,
        grid_lon=expected_lon,
        observation_id=np.asarray("synthetic-frame"),
        template_observation_id=np.asarray("template-frame"),
        target_center=np.asarray([10.0, -50.0]),
    )

    table = dense_pmw_table(tmp_path, storm)
    table.loc[0, "observation_id"] = "synthetic-frame"
    field, lat, lon, center = dense_pmw_field(table.iloc[0], tmp_path, storm)

    assert np.array_equal(field, np.full((2, 3), 3, dtype=np.float32))
    assert np.array_equal(lat, expected_lat)
    assert np.array_equal(lon, expected_lon)
    assert center == (10.0, -50.0)


def test_dense_pmw_loader_prefers_explicit_coordinate_tensor(tmp_path: Path) -> None:
    storm = "EP182023"
    tensor_path = tmp_path / "observations" / storm / "frame.pt"
    coordinate_path = tmp_path / "coordinates" / storm / "frame.pt"
    tensor_path.parent.mkdir(parents=True)
    coordinate_path.parent.mkdir(parents=True)
    torch.save(torch.stack([torch.full((2, 3), value) for value in range(4)]), tensor_path)
    expected_lat = np.array([[11.0, 11.0, 11.0], [9.0, 9.0, 9.0]])
    expected_lon = np.array([[-51.0, -50.0, -49.0], [-51.0, -50.0, -49.0]])
    torch.save(
        {"latitude": torch.from_numpy(expected_lat), "longitude": torch.from_numpy(expected_lon)},
        coordinate_path,
    )
    row = pd.Series(
        {
            "observation_id": "synthetic-frame",
            "path": str(tensor_path.relative_to(tmp_path)),
            "coordinates_path": str(coordinate_path.relative_to(tmp_path)),
            "variables": json.dumps(
                ["TB_36.5H", "TB_36.5V", "TB_A89.0H", "TB_A89.0V"]
            ),
            "ibtracs_center_lat": 10.0,
            "ibtracs_center_lon": -50.0,
        }
    )

    field, lat, lon, center = dense_pmw_field(row, tmp_path, storm)

    assert np.array_equal(field, np.full((2, 3), 3, dtype=np.float32))
    assert np.array_equal(lat, expected_lat)
    assert np.array_equal(lon, expected_lon)
    assert center == (10.0, -50.0)


def test_dense_pmw_loader_does_not_invent_geolocation(tmp_path: Path) -> None:
    storm = "EP182023"
    tensor_path = tmp_path / "observations" / storm / "frame.pt"
    tensor_path.parent.mkdir(parents=True)
    torch.save(torch.zeros((4, 2, 3)), tensor_path)
    row = pd.Series(
        {
            "observation_id": "synthetic-frame",
            "path": str(tensor_path.relative_to(tmp_path)),
            "variables": json.dumps(
                ["TB_36.5H", "TB_36.5V", "TB_A89.0H", "TB_A89.0V"]
            ),
            "target_grid_template_observation_id": "template-frame",
            "ibtracs_center_lat": 10.0,
            "ibtracs_center_lon": -50.0,
        }
    )

    with pytest.raises(KeyError, match="template observation is unavailable"):
        dense_pmw_field(row, tmp_path, storm)


def test_template_swath_orientation_is_preserved(monkeypatch) -> None:
    template_lat = np.array([[12.0, 11.4, 10.7], [11.1, 10.2, 9.3]])
    template_lon = np.array([[-52.0, -50.5, -49.0], [-51.6, -50.1, -48.7]])
    template = SimpleNamespace(
        observation_id="template-frame",
        sensor="AMSR2_GCOMW1",
        ibtracs_center=(10.0, -50.0),
    )
    row = SimpleNamespace(
        observation_id="synthetic-frame",
        target_grid_template_observation_id="template-frame",
        ibtracs_center_lat=10.0,
        ibtracs_center_lon=-50.0,
    )

    monkeypatch.setattr(
        renderer,
        "_load_pmw_channels",
        lambda observation, channels: {
            channels[0]: (np.zeros_like(template_lat), template_lat, template_lon)
        },
    )
    lat, lon, center, template_id = synthetic_pmw_grid(
        row, {"template-frame": template}, {}
    )

    assert np.array_equal(lat, template_lat)
    assert np.array_equal(lon, template_lon)
    assert center == (10.0, -50.0)
    assert template_id == "template-frame"


def test_template_center_is_interpolated_from_geostationary_track(monkeypatch) -> None:
    template_lat = np.array([[12.0, 12.0], [10.0, 10.0]])
    template_lon = np.array([[-52.0, -50.0], [-52.0, -50.0]])
    template = SimpleNamespace(
        observation_id="template-frame",
        storm_id="AL082025",
        source_type="pmw",
        sensor="AMSR2_GCOMW1",
        timestamp=pd.Timestamp("2025-09-22T15:00:00Z"),
        ibtracs_center=None,
    )
    geos = [
        SimpleNamespace(
            observation_id=f"geo-{hour}",
            storm_id="AL082025",
            source_type="geo",
            timestamp=pd.Timestamp(f"2025-09-22T{hour}:00:00Z"),
            ibtracs_center=(lat, lon),
            ibtracs_center_lat=lat,
            ibtracs_center_lon=lon,
        )
        for hour, lat, lon in (("14", 10.0, -50.0), ("16", 12.0, -52.0))
    ]
    row = SimpleNamespace(
        observation_id="synthetic-frame",
        target_grid_template_observation_id="template-frame",
        ibtracs_center_lat=20.0,
        ibtracs_center_lon=-70.0,
    )

    monkeypatch.setattr(
        renderer,
        "_load_pmw_channels",
        lambda observation, channels: {
            channels[0]: (np.zeros_like(template_lat), template_lat, template_lon)
        },
    )
    lat, lon, center, _ = synthetic_pmw_grid(
        row,
        {record.observation_id: record for record in [template, *geos]},
        {},
    )

    assert center == (20.0, -70.0)
    assert lat.mean() == pytest.approx(20.0, abs=0.03)
    assert lon.mean() == pytest.approx(-70.0, abs=0.03)


def test_recenter_geolocation_moves_center_without_flipping_grid() -> None:
    lat = np.array([[11.0, 11.0], [9.0, 9.0]])
    lon = np.array([[-51.0, -49.0], [-51.0, -49.0]])

    moved_lat, moved_lon = recenter_geolocation(
        lat, lon, (10.0, -50.0), (20.0, -70.0)
    )

    assert moved_lat[0].mean() > moved_lat[1].mean()
    assert moved_lon[:, 0].mean() < moved_lon[:, 1].mean()
    assert moved_lat.mean() == pytest.approx(20.0, abs=0.02)
    assert moved_lon.mean() == pytest.approx(-70.0, abs=0.02)
