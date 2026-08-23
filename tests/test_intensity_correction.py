from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from geo2wf.data.intensity import (
    INTENSITY_METADATA_NAMES,
    KNOT_TO_MS,
    UNetIntensityDataModule,
    UNetIntensityDataset,
    _storm_and_category_weights,
    encode_intensity_metadata,
    tropical_category_from_wind_ms,
)
from geo2wf.models.intensity_correction import (
    UNetIntensityCorrection,
    summarize_intensity_rows,
)
from scripts.export_unet_intensity_cache import (
    _eligible_fixes,
    _storm_split_audit,
)
from scripts.run_intensity_correction_inference import inference_rows


def _cache(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "cache-metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "single_timestep": True,
                "unet_checkpoint": {"sha256": "checkpoint-hash"},
            }
        ),
        encoding="utf-8",
    )
    split_storms = {"train": "AL012020", "val": "EP022021", "test": "WP032022"}
    for split, storm_id in split_storms.items():
        fields = root / split / "fields"
        fields.mkdir(parents=True)
        rows = []
        categories = [-1, 0, 1] if split == "train" else [0]
        for index, category in enumerate(categories):
            wind = np.full((32, 32), 10.0 + index, dtype=np.float32)
            wind[4, 5] = 20.0 + index
            mask = np.ones((32, 32), dtype=np.uint8)
            mask[:2] = 0
            distance = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
            sample_id = f"{split}-{index}"
            relative = Path(split) / "fields" / f"{sample_id}.npz"
            np.savez_compressed(
                root / relative,
                wind_speed_ms=wind,
                valid_mask=mask,
                distance_to_center=distance,
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "storm_id": storm_id,
                    "split": split,
                    "field_path": relative.as_posix(),
                    "observation_timestamp": f"2022-08-0{index + 1}T06:00:00+00:00",
                    "target_timestamp": f"2022-08-0{index + 1}T06:00:00+00:00",
                    "center_lat": 20.0,
                    "center_lon": -60.0,
                    "basin": storm_id[:2] if storm_id[:2] in {"EP", "WP"} else "NA",
                    "storm_elapsed_hours": 12.0 * index,
                    "target_wind_ms": (30.0 + 35.0 * index) * KNOT_TO_MS,
                    "target_category": category,
                    "raw_unet_max_wind_ms": 20.0 + index,
                    "valid_fraction": float(mask.mean()),
                }
            )
        pd.DataFrame(rows).to_csv(root / split / "manifest.csv", index=False)
    return root


def _v2_cache(root: Path) -> Path:
    root = _cache(root)
    metadata_path = root / "cache-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "schema_version": 2,
            "target": {"sar_robust_peak_fraction": 0.005},
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    for split in ("train", "val", "test"):
        path = root / split / "manifest.csv"
        frame = pd.read_csv(path)
        robust_values = []
        for relative in frame["field_path"]:
            with np.load(root / relative) as payload:
                wind = payload["wind_speed_ms"]
                valid = payload["valid_mask"].astype(bool) & np.isfinite(wind)
            values = wind[valid]
            count = max(1, math.ceil(values.size * 0.005))
            robust_values.append(float(np.sort(values)[-count:].mean()))
        frame["intensity_target_source"] = "sar_robust_peak"
        frame["anchor_statistic"] = "robust_peak"
        frame["ibtracs_target_ms"] = frame["target_wind_ms"]
        frame["sar_robust_peak_target_ms"] = np.asarray(robust_values) + 2.0
        frame["sar_max_wind_ms"] = frame["raw_unet_max_wind_ms"] + 1.0
        frame["raw_unet_robust_peak_ms"] = robust_values
        frame["target_wind_ms"] = frame["sar_robust_peak_target_ms"]
        frame["target_category"] = frame["target_wind_ms"].map(
            tropical_category_from_wind_ms
        )
        frame["is_rapid_intensification"] = split == "val"
        frame["ri_24h_change_ms"] = 30.0 * KNOT_TO_MS if split == "val" else np.nan
        frame["cohort_retained_count"] = len(frame)
        frame["filtered_unbracketed_count"] = 0
        frame["filtered_invalid_sar_center_count"] = 0
        frame["filtered_unusable_sar_count"] = 0
        frame.to_csv(path, index=False)
    return root


@pytest.mark.parametrize(
    ("wind_kt", "expected"),
    [
        (0.0, -1),
        (33.999, -1),
        (34.0, 0),
        (63.999, 0),
        (64.0, 1),
        (83.0, 2),
        (96.0, 3),
        (113.0, 4),
        (137.0, 5),
    ],
)
def test_category_thresholds_use_continuous_knots_without_rounding(
    wind_kt: float, expected: int
) -> None:
    assert tropical_category_from_wind_ms(wind_kt * KNOT_TO_MS) == expected


def test_metadata_is_current_timestep_only_and_has_stable_width() -> None:
    encoded = encode_intensity_metadata(
        {
            "observation_timestamp": "2022-08-03T06:30:00Z",
            "center_lat": 18.0,
            "center_lon": -179.0,
            "basin": "EP",
            "storm_elapsed_hours": 24.0,
            "valid_fraction": 0.75,
        }
    )
    assert encoded.shape == (len(INTENSITY_METADATA_NAMES),)
    assert torch.isfinite(encoded).all()
    assert not any(
        token in name
        for name in INTENSITY_METADATA_NAMES
        for token in ("previous", "history", "remaining", "storm_id", "wind")
    )


def test_dataset_returns_one_field_and_checks_provenance(tmp_path: Path) -> None:
    root = _cache(tmp_path / "cache")
    dataset = UNetIntensityDataset(
        root, "train", expected_unet_checkpoint_sha256="checkpoint-hash"
    )
    sample = dataset[0]
    assert sample["wind_field"].shape == (32, 32)
    assert sample["valid_mask"].shape == (32, 32)
    assert sample["metadata"].shape == (len(INTENSITY_METADATA_NAMES),)
    assert not any(key in sample for key in ("history", "sequence", "previous_field"))
    assert sample["raw_unet_max_wind_ms"] == 20.0
    assert sample["intensity_target_source"] == "ibtracs"
    assert sample["anchor_statistic"] == "max"
    assert sample["ibtracs_target_ms"] == sample["target_wind_ms"]
    assert torch.isnan(sample["sar_robust_peak_target_ms"])
    assert dataset.data_spec.cache_schema_version == 1
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        UNetIntensityDataset(root, "train", expected_unet_checkpoint_sha256="different")


def test_v2_cache_round_trip_exposes_both_targets_ri_and_robust_anchor(
    tmp_path: Path,
) -> None:
    root = _v2_cache(tmp_path / "cache")
    train = UNetIntensityDataset(root, "train")
    validation = UNetIntensityDataset(root, "val")
    sample = train[0]
    ri_sample = validation[0]

    assert train.data_spec.cache_schema_version == 2
    assert sample["intensity_target_source"] == "sar_robust_peak"
    assert sample["anchor_statistic"] == "robust_peak"
    assert sample["target_wind_ms"] == sample["sar_robust_peak_target_ms"]
    assert sample["raw_unet_robust_peak_ms"] == pytest.approx(12.0)
    assert sample["raw_unet_max_wind_ms"] == 20.0
    assert ri_sample["is_rapid_intensification"]
    assert ri_sample["ri_24h_change_ms"] == pytest.approx(30.0 * KNOT_TO_MS)


def test_storm_and_category_weights_equalize_storm_totals() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d", "e", "f"],
            "storm_id": ["one", "one", "two", "two", "two", "two"],
            "target_category": [0, 5, 0, 0, 0, 0],
        }
    )
    weights = _storm_and_category_weights(samples, 4.0)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[:2].sum() == pytest.approx(weights[2:].sum())
    assert weights[1] > weights[0]


def test_datamodule_rejects_storm_overlap(tmp_path: Path) -> None:
    root = _cache(tmp_path / "cache")
    val = pd.read_csv(root / "val" / "manifest.csv")
    val["storm_id"] = "AL012020"
    val.to_csv(root / "val" / "manifest.csv", index=False)
    with pytest.raises(ValueError, match="not storm-disjoint"):
        UNetIntensityDataModule(root)


def _batch(batch_size: int = 2) -> dict[str, object]:
    wind = torch.full((batch_size, 32, 32), 10.0)
    wind[:, 5, 5] = torch.arange(20.0, 20.0 + batch_size)
    mask = torch.ones_like(wind, dtype=torch.bool)
    mask[:, :2] = False
    return {
        "wind_field": wind,
        "valid_mask": mask,
        "distance_to_center": torch.linspace(0, 1, 32 * 32)
        .reshape(1, 32, 32)
        .expand(batch_size, -1, -1),
        "metadata": torch.zeros(batch_size, len(INTENSITY_METADATA_NAMES)),
        "target_wind_ms": torch.tensor([25.0 + index for index in range(batch_size)]),
        "target_category": torch.zeros(batch_size, dtype=torch.long),
        "sample_weight": torch.ones(batch_size),
        "sample_id": [f"sample-{index}" for index in range(batch_size)],
        "storm_id": [f"storm-{index}" for index in range(batch_size)],
        "observation_timestamp": ["2022-01-01T00:00:00Z"] * batch_size,
    }


def test_zero_initialized_model_reproduces_masked_unet_maximum() -> None:
    torch.manual_seed(1)
    model = UNetIntensityCorrection(field_base_channels=4, field_channel_mults=(1, 2))
    batch = _batch()
    prediction = model.predict_intensity(batch)
    assert torch.equal(prediction.correction_ms, torch.zeros(2))
    assert torch.allclose(prediction.output_msw_ms, torch.tensor([20.0, 21.0]))


def test_robust_peak_anchor_uses_top_half_percent_mean() -> None:
    model = UNetIntensityCorrection(
        field_base_channels=4,
        field_channel_mults=(1, 2),
        anchor_statistic="robust_peak",
        robust_peak_fraction=0.005,
    )
    prediction = model.predict_intensity(_batch(1))

    assert prediction.raw_unet_max_wind_ms.item() == 20.0
    assert prediction.raw_unet_robust_peak_ms.item() == pytest.approx(12.0)
    assert prediction.raw_unet_anchor_ms.item() == pytest.approx(12.0)
    assert prediction.output_msw_ms.item() == pytest.approx(12.0)


def test_invalid_wind_values_cannot_affect_model_or_raw_maximum() -> None:
    torch.manual_seed(2)
    model = UNetIntensityCorrection(
        field_base_channels=4, field_channel_mults=(1, 2)
    ).eval()
    with torch.no_grad():
        model.correction_head.weight.fill_(0.1)
    baseline = _batch(1)
    changed = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in baseline.items()
    }
    changed["wind_field"][:, 0, 0] = 10_000.0
    first = model.predict_intensity(baseline)
    second = model.predict_intensity(changed)
    assert torch.allclose(first.raw_unet_max_wind_ms, second.raw_unet_max_wind_ms)
    assert torch.allclose(first.output_msw_ms, second.output_msw_ms)


def test_correction_is_signed_and_final_wind_is_nonnegative() -> None:
    model = UNetIntensityCorrection(field_base_channels=4, field_channel_mults=(1, 2))
    with torch.no_grad():
        model.correction_head.bias.fill_(-100.0)
    prediction = model.predict_intensity(_batch(1))
    assert prediction.correction_ms.item() < 0
    assert prediction.output_msw_ms.item() == 0.0


def test_training_and_inference_smoke_from_single_field_cache(tmp_path: Path) -> None:
    root = _cache(tmp_path / "cache")
    dataset = UNetIntensityDataset(root, "train")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    model = UNetIntensityCorrection(field_base_channels=4, field_channel_mults=(1, 2))
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    rows = inference_rows(model, loader, device="cpu")
    assert len(rows) == len(dataset)
    assert set(rows[0]) >= {
        "raw_unet_max_wind_ms",
        "correction_ms",
        "output_msw_ms",
        "output_category",
    }


def test_validation_logs_dual_reference_ri_namespace(monkeypatch) -> None:
    model = UNetIntensityCorrection(
        field_base_channels=4,
        field_channel_mults=(1, 2),
        log_wandb_validation_media=False,
    )
    batch = _batch()
    batch.update(
        {
            "ibtracs_target_ms": torch.tensor([25.0, 26.0]),
            "sar_robust_peak_target_ms": torch.tensor([24.0, 27.0]),
            "is_rapid_intensification": torch.tensor([True, False]),
            "intensity_target_source": ["ibtracs", "ibtracs"],
        }
    )
    logged = {}
    monkeypatch.setattr(
        model,
        "log",
        lambda name, value, **kwargs: logged.__setitem__(name, value),
    )

    model.on_validation_epoch_start()
    model.validation_step(batch, 0)
    model.on_validation_epoch_end()

    assert logged["val_ri/samples"] == 1.0
    assert logged["val_ri/storms"] == 1.0
    assert "val_ri/ibtracs_mae_ms" in logged
    assert "val_ri/ibtracs_category_macro_f1" in logged
    assert "val_ri/ibtracs_raw_unet_rmse_ms" in logged
    assert "val_ri/sar_robust_peak_mae_ms" in logged
    assert "val_ri/sar_robust_peak_raw_unet_category_macro_f1" in logged


def test_distributed_validation_rows_are_gathered(monkeypatch) -> None:
    local = [{"sample_id": "local"}]
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(output, rows) -> None:
        output[0] = rows
        output[1] = [{"sample_id": "remote"}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    gathered = UNetIntensityCorrection._distributed_rows(local)

    assert [row["sample_id"] for row in gathered] == ["local", "remote"]


def test_metric_summary_reports_baseline_storm_macro_and_confusion() -> None:
    rows = [
        {
            "storm_id": "a",
            "prediction_ms": 20.0,
            "target_ms": 22.0,
            "raw_unet_ms": 10.0,
            "prediction_category": 0,
            "target_category": 0,
        },
        {
            "storm_id": "b",
            "prediction_ms": 40.0,
            "target_ms": 42.0,
            "raw_unet_ms": 30.0,
            "prediction_category": 1,
            "target_category": 2,
        },
    ]
    summary = summarize_intensity_rows(rows)
    assert summary["regression"]["mae_ms"] == 2.0
    assert summary["raw_unet_baseline"]["mae_ms"] == 12.0
    assert summary["storm_macro_mae_ms"] == 2.0
    assert summary["category"]["within_one_accuracy"] == 1.0
    assert len(summary["category"]["confusion_matrix"]) == 7


def test_ibtracs_filter_keeps_only_tropical_usa_wind_fixes() -> None:
    frame = pd.DataFrame(
        {
            "_ibtracs_timestamp": pd.to_datetime(
                ["2022-01-01T00:00Z", "2022-01-01T06:00Z", "2022-01-01T12:00Z"]
            ),
            "USA_WIND": [40, 50, ""],
            "USA_SSHS": [0, -4, 1],
        }
    )
    eligible = _eligible_fixes(frame)
    assert len(eligible) == 1
    assert eligible.loc[0, "_usa_wind"] == 40


def test_export_split_audit_rejects_same_storm_in_multiple_splits() -> None:
    class Record:
        source_type = "geo"
        storm_id = "AL012020"

        def __init__(self, split: str) -> None:
            self.split = split

    with pytest.raises(ValueError, match="not storm-disjoint"):
        _storm_split_audit([Record("train"), Record("test")], {"train", "test"})
