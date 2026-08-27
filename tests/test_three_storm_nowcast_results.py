from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_three_storm_nowcast_results import (
    DOCS_END,
    DOCS_START,
    build,
    load_frames,
)
from scripts.run_intensity_comparison_storm_inference import (
    STORMS,
    _regime_paths,
)


def _inference_frame(*, offset: float = 0.0, include_ablations: bool) -> pd.DataFrame:
    rows = []
    for storm_index, storm_id in enumerate(STORMS):
        for observation_index in range(2):
            target = 20.0 + 5.0 * storm_index + observation_index
            row = {
                "observation_id": f"{storm_id}-{observation_index}",
                "storm_id": storm_id,
                "source_split": "val",
                "observation_timestamp": (
                    pd.Timestamp("2025-08-01T00:00:00Z")
                    + pd.Timedelta(days=storm_index, hours=observation_index)
                ).isoformat(),
                "target_ms": target,
                "inference_valid": True,
                "is_rapid_intensification": observation_index == 1,
                "ri_24h_change_ms": 16.0 if observation_index == 1 else 2.0,
                "unet_raw_max_ms": target + 3.0 + offset,
                "unet_correction_ms": target + 2.0 + offset,
                "joint_unet_mlp_ms": target + 1.0 + offset,
            }
            if include_ablations:
                row["ablation_max_wind_only_ms"] = target + 0.75
                row["ablation_max_wind_radii_ms"] = target + 0.5
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_writes_all_series_metrics_figure_and_docs_section(
    tmp_path: Path,
) -> None:
    with_path = tmp_path / "with.csv"
    without_path = tmp_path / "without.csv"
    _inference_frame(include_ablations=True).to_csv(with_path, index=False)
    _inference_frame(offset=0.25, include_ablations=False).to_csv(
        without_path, index=False
    )
    docs_page = tmp_path / "results.md"
    docs_page.write_text(
        f"# Results\n\n{DOCS_START}\nPending\n{DOCS_END}\n", encoding="utf-8"
    )
    args = Namespace(
        with_era5_csv=with_path,
        without_era5_csv=without_path,
        storms=list(STORMS),
        predictions_output=tmp_path / "predictions.csv",
        metrics_output=tmp_path / "metrics.csv",
        metadata_output=tmp_path / "metadata.json",
        png_output=tmp_path / "figure.png",
        pdf_output=tmp_path / "figure.pdf",
        docs_page=docs_page,
        no_docs_update=False,
    )

    payload = build(args)

    predictions = pd.read_csv(args.predictions_output)
    metrics = pd.read_csv(args.metrics_output)
    assert len(payload["series"]) == 8
    assert len(predictions) == len(STORMS) * 2 * 8
    assert len(metrics) == 8 * (2 + 2 * len(STORMS))
    assert set(metrics["scope"]) == {
        "all_three_storms",
        "rapid_intensification_all_storms",
        *STORMS,
        *(f"{storm}_rapid_intensification" for storm in STORMS),
    }
    assert args.png_output.is_file()
    assert args.pdf_output.is_file()
    assert json.loads(args.metadata_output.read_text())["cohort"]["samples"] == 6
    docs = docs_page.read_text(encoding="utf-8")
    assert "Three-storm maximum-wind nowcasts" in docs
    assert "Joint ablation: max wind + radii" in docs
    assert "RI MAE" in docs
    assert "figure.png" in docs


def test_nowcast_builder_rejects_different_regime_targets(tmp_path: Path) -> None:
    with_path = tmp_path / "with.csv"
    without_path = tmp_path / "without.csv"
    with_frame = _inference_frame(include_ablations=False)
    without_frame = _inference_frame(include_ablations=False)
    without_frame.loc[0, "target_ms"] += 1.0
    with_frame.to_csv(with_path, index=False)
    without_frame.to_csv(without_path, index=False)

    with pytest.raises(ValueError, match="different maximum-wind targets"):
        load_frames(with_path, without_path)


def test_storm_inference_resolves_new_completed_workflow(tmp_path: Path) -> None:
    artifacts = {}
    for name in ("unet", "correction", "joint"):
        path = tmp_path / f"{name}.ckpt"
        path.touch()
        artifacts[f"{name}_checkpoint"] = str(path)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "cache-metadata.json").write_text("{}\n", encoding="utf-8")
    artifacts["cache_root"] = str(cache_root)
    workflow = tmp_path / "workflow.json"
    workflow.write_text(
        json.dumps(
            {
                "status": "completed",
                "era5": "with",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        era5="with",
        comparison_run=workflow,
        unet_checkpoint=None,
        correction_checkpoint=None,
        joint_checkpoint=None,
        intensity_cache_metadata=None,
        ablation_max_wind_checkpoint=None,
        ablation_radii_checkpoint=None,
    )

    paths = _regime_paths(args)

    assert paths["unet"] == Path(artifacts["unet_checkpoint"]).resolve()
    assert paths["cache_metadata"] == (cache_root / "cache-metadata.json").resolve()
