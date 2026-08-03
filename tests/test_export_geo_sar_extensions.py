from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from export_geo_sar_geotiffs import (  # noqa: E402
    Observation,
    _ibtracs_manifest_fields,
    _match_ibtracs_record,
    _write_manifest,
)


def _observation(timestamp: str) -> Observation:
    return Observation(
        observation_id="sar-1",
        storm_id="AL012024",
        split="train",
        source_type="sar",
        source="test",
        sensor="SAR",
        path=Path("sar.nc"),
        timestamp=pd.Timestamp(timestamp),
        center_lat=20.0,
        center_lon=-90.0,
        ibtracs_center_lat=20.0,
        ibtracs_center_lon=-90.0,
        variables=("wind_speed",),
    )


def test_ibtracs_match_retains_full_row_and_builds_intensity_fields() -> None:
    records = pd.DataFrame(
        [
            {
                "ISO_TIME": "2024-06-20 00:00:00",
                "NAME": "ALBERTO",
                "WMO_WIND": "",
                "USA_WIND": "45",
                "WMO_PRES": "",
                "USA_PRES": "992",
                "USA_RMW": "90",
                "_ibtracs_timestamp": pd.Timestamp("2024-06-20T00:00:00Z"),
            },
            {
                "ISO_TIME": "2024-06-20 03:00:00",
                "NAME": "ALBERTO",
                "WMO_WIND": "50",
                "USA_WIND": "48",
                "WMO_PRES": "988",
                "USA_PRES": "990",
                "USA_RMW": "80",
                "_ibtracs_timestamp": pd.Timestamp("2024-06-20T03:00:00Z"),
            },
        ]
    )
    record, dt_minutes = _match_ibtracs_record(
        {"AL012024": records},
        _observation("2024-06-20T00:45:00Z"),
        max_time_gap_hours=6.1,
    )
    fields = _ibtracs_manifest_fields(record, dt_minutes)

    assert dt_minutes == pytest.approx(-45.0)
    assert fields["ibtracs_name"] == "ALBERTO"
    assert fields["ibtracs_usa_rmw"] == "90"
    assert fields["ibtracs_msw_kt"] == pytest.approx(45.0)
    assert fields["ibtracs_msw_ms"] == pytest.approx(45.0 * 0.514444)
    assert fields["ibtracs_msw_source"] == "USA_WIND"
    assert fields["ibtracs_mslp_hpa"] == pytest.approx(992.0)


def test_manifest_writer_accepts_optional_columns_on_later_rows(tmp_path) -> None:
    path = tmp_path / "manifest.csv"
    _write_manifest(
        path,
        [
            {"sample_id": "one", "pmw_path": ""},
            {"sample_id": "two", "pmw_path": "two.tif", "ibtracs_name": "BETA"},
        ],
    )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["ibtracs_name"] == ""
    assert rows[1]["ibtracs_name"] == "BETA"
