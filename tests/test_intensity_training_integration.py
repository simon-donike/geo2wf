from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from geo2wf.data.intensity import UNetIntensityDataModule
from geo2wf.models.intensity_correction import UNetIntensityCorrection


def _write_split(root: Path, split: str, storm_id: str) -> None:
    fields = root / split / "fields"
    fields.mkdir(parents=True)
    rows = []
    for index, (wind_ms, category) in enumerate(((20.0, 0), (35.0, 1))):
        wind = np.full((32, 32), wind_ms - 5.0, dtype=np.float32)
        wind[8, 8] = wind_ms
        mask = np.ones_like(wind, dtype=np.uint8)
        distance = np.linspace(0.0, 1.0, wind.size, dtype=np.float32).reshape(
            wind.shape
        )
        relative = Path(split) / "fields" / f"{split}-{index}.npz"
        np.savez_compressed(
            root / relative,
            wind_speed_ms=wind,
            valid_mask=mask,
            distance_to_center=distance,
        )
        rows.append(
            {
                "sample_id": f"{split}-{index}",
                "storm_id": storm_id,
                "split": split,
                "field_path": relative.as_posix(),
                "observation_timestamp": f"2022-08-0{index + 1}T00:00:00Z",
                "target_timestamp": f"2022-08-0{index + 1}T00:00:00Z",
                "center_lat": 15.0,
                "center_lon": -50.0,
                "basin": "NA",
                "storm_elapsed_hours": index * 6.0,
                "target_wind_ms": wind_ms + 2.0,
                "target_category": category,
                "raw_unet_max_wind_ms": wind_ms,
                "valid_fraction": 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(root / split / "manifest.csv", index=False)


def test_lightning_fit_and_checkpoint_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    (root / "cache-metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "single_timestep": True,
                "unet_checkpoint": {"sha256": "synthetic"},
            }
        ),
        encoding="utf-8",
    )
    _write_split(root, "train", "AL012020")
    _write_split(root, "val", "EP022021")
    _write_split(root, "test", "WP032022")
    datamodule = UNetIntensityDataModule(root, batch_size=2, num_workers=0)
    model = UNetIntensityCorrection(
        field_base_channels=4,
        field_channel_mults=(1, 2),
        fusion_hidden_features=16,
        metadata_hidden_features=8,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)
    checkpoint = tmp_path / "intensity.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = UNetIntensityCorrection.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    restored.validate_data_spec(datamodule.data_spec)
    assert restored.metadata_features == model.metadata_features
