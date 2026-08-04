from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.run_storm_diffusion_inference import (
    CROP_SIZE,
    METRIC_COLUMNS,
    _aggregate_member_metrics,
    _member_latent_seed,
    _select_medoid_member,
    _summarize_ensemble,
)


def _complete_metrics(value: float) -> dict[str, float]:
    return {metric: value for metric in METRIC_COLUMNS}


def test_member_seed_derivation_is_stable_and_member_specific() -> None:
    seeds = [_member_latent_seed(42, "observation", index) for index in range(3)]

    assert seeds == [
        _member_latent_seed(42, "observation", index) for index in range(3)
    ]
    assert len(set(seeds)) == 3
    assert all(0 <= seed < 2**63 - 1 for seed in seeds)


def test_medoid_is_complete_member_closest_to_median_consensus() -> None:
    members = torch.stack(
        [torch.zeros(3, 3), torch.ones(3, 3), torch.full((3, 3), 10.0)]
    )

    medoid, distances, median_field = _select_medoid_member(
        members, torch.ones(1, 3, 3, dtype=torch.bool)
    )

    assert medoid == 1
    assert distances == [1.0, 0.0, 9.0]
    assert torch.equal(median_field, members[medoid])


def test_even_member_median_field_averages_middle_members() -> None:
    members = torch.tensor([[[0.0]], [[10.0]]])

    medoid, distances, median_field = _select_medoid_member(
        members, torch.ones(1, 1, 1, dtype=torch.bool)
    )

    assert median_field.item() == 5.0
    assert medoid == 0
    assert distances == [5.0, 5.0]


def test_member_metric_aggregation_retains_quantiles_and_field_diagnostics() -> None:
    summary = _aggregate_member_metrics(
        [_complete_metrics(10.0), _complete_metrics(30.0), _complete_metrics(50.0)],
        1,
        _complete_metrics(12.0),
        _complete_metrics(20.0),
        quantiles=[0.1, 0.9],
        summary_aggregation="median",
    )

    assert summary["output_msw_ms"] == 30.0
    assert summary["output_msw_ms_member_mean"] == 30.0
    assert summary["output_msw_ms_member_median"] == 30.0
    assert summary["output_msw_ms_member_p10"] == 14.0
    assert summary["output_msw_ms_member_p90"] == 46.0
    assert math.isclose(summary["output_msw_ms_member_std"], 16.32993161855452)
    assert summary["output_msw_ms_member_min"] == 10.0
    assert summary["output_msw_ms_member_max"] == 50.0
    assert summary["output_msw_ms_member_range"] == 40.0
    assert summary["output_msw_ms_medoid"] == 30.0
    assert summary["output_msw_ms_mean_field"] == 12.0
    assert summary["output_robust_peak_ms_member_p90"] == 46.0


def test_member_first_msw_avoids_peak_suppression_from_mean_field() -> None:
    members = torch.full((3, CROP_SIZE, CROP_SIZE), 10.0)
    for member_index, column in enumerate((40, 96, 150)):
        members[member_index, 96, column] = 70.0
    coordinates = torch.arange(CROP_SIZE, dtype=torch.float32) - CROP_SIZE / 2
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    distance_km = torch.sqrt(xx.square() + yy.square())
    valid = torch.ones(1, CROP_SIZE, CROP_SIZE, dtype=torch.bool)

    summary, member_metrics, medoid, _, _, _ = _summarize_ensemble(
        members,
        valid,
        distance_km,
        quantiles=[0.1, 0.9],
        summary_aggregation="median",
    )

    assert medoid in range(3)
    assert all(metrics["msw"] == 70.0 for metrics in member_metrics)
    assert summary["output_msw_ms"] == 70.0
    assert math.isclose(summary["output_msw_ms_mean_field"], 30.0)
    assert summary["output_robust_peak_ms"] < summary["output_msw_ms"]
