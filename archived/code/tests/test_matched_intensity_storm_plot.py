from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pandas as pd

from scripts.build_matched_intensity_storm_plot import (
    _merged_ri_windows,
    _selected_rows,
)


def test_overlapping_ri_history_windows_are_merged() -> None:
    timestamps = pd.to_datetime(
        ["2025-09-27T12:00:00Z", "2025-09-28T00:00:00Z", "2025-09-30T00:00:00Z"],
        utc=True,
    )

    windows = _merged_ri_windows(timestamps)

    assert windows == [
        (
            pd.Timestamp("2025-09-26T12:00:00Z"),
            pd.Timestamp("2025-09-28T00:00:00Z"),
        ),
        (
            pd.Timestamp("2025-09-29T00:00:00Z"),
            pd.Timestamp("2025-09-30T00:00:00Z"),
        ),
    ]


def test_plot_selection_uses_one_model_and_ranks_ri_storms(tmp_path: Path) -> None:
    rows = []
    for storm_id, ri_flags in {
        "storm-a": [True, True, False],
        "storm-b": [True, False, False, False],
        "storm-c": [False, False, False, False, False],
    }.items():
        for index, is_ri in enumerate(ri_flags):
            rows.append(
                {
                    "sample_id": f"{storm_id}-{index}",
                    "storm_id": storm_id,
                    "observation_timestamp": f"2025-09-{index + 1:02d}T00:00:00Z",
                    "era5": "with_era5",
                    "trained_target": "ibtracs",
                    "model_key": "unet_correction",
                    "is_rapid_intensification": is_ri,
                }
            )
            rows.append(
                {
                    **rows[-1],
                    "sample_id": f"unused-{storm_id}-{index}",
                    "model_key": "joint_unet_mlp",
                }
            )
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"prediction_rows": rows}), encoding="utf-8")
    args = Namespace(
        result=result,
        era5="with_era5",
        trained_target="ibtracs",
        model_key="unet_correction",
        storms=2,
    )

    selected = _selected_rows(args)

    assert selected["storm_id"].drop_duplicates().tolist() == ["storm-a", "storm-b"]
    assert selected["sample_id"].is_unique
    assert len(selected) == 7
