# First experiment

This smoke test exercises export, configuration composition, `DataSpec`
validation, one Lightning training step, CSV logging, and physical prediction.

## 1. Export two GEO–SAR pairs per split

```bash
uv run geo2wf-export geo-sar \
  --config configs/config.yaml \
  --limit 2
```

The command retains the exporter's established full-YAML/flag interface. It
writes raw values and masks, then derives `stats.json` from the training split
only.

```text
data/geotiff/geo_sar/
├── stats.json
├── train/manifest.csv
├── val/manifest.csv
└── test/manifest.csv
```

## 2. Inspect the composed data contract

```bash
uv run python - <<'PY'
from geo2wf.config import compose_config, instantiate_datamodule

config = compose_config([
    "data=geo_sar_common10_era5",
    "model=deterministic_residual",
])
datamodule = instantiate_datamodule(config)
datamodule.setup("fit")
print(datamodule.data_spec)

batch = next(iter(datamodule.train_dataloader()))
for key in ("condition", "target", "condition_mask", "target_mask"):
    print(key, tuple(batch[key].shape), batch[key].dtype)
print("sample ids:", batch["sample_id"])
print("metadata records:", len(batch["meta"]))
PY
```

The default ERA5 data/model pair contains compatible channel widths. Its
`DataSpec` lists the ordered inputs and required ERA5 companion before training.

## 3. Run one training and validation batch

```bash
WANDB_DISABLED=true uv run geo2wf-train \
  data=geo_sar_common10_era5 \
  model=deterministic_residual \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=1 \
  trainer.enable_checkpointing=false \
  data.loader.num_workers=0
```

Hydra overrides apply only to this invocation. `WANDB_DISABLED=true` disables
W&B while retaining CSV logging and the run manifest.

## 4. Inspect the run directory

The command prints the timestamped directory created under `logs/`. It contains:

```text
logs/<timestamp>_modular/
├── resolved-config.yaml
├── run-manifest.json
├── source-diff.patch
├── metrics/metrics.csv
└── source-snapshot/              # relevant untracked source, when present
```

A successful smoke run has finite `train/loss` and `val/loss`.

## Next

- [Configuration groups and override rules](../experiments/configuration.md)
- [Training, resume, and transfer](../experiments/training.md)
- [Dataset contract](../data/dataset-contract.md)
- [ERA5-residual field model](../models/era5-residual.md)
