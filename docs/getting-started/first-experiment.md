# First experiment

Start with an existing [paired raster export](../data/dataset-contract.md)
containing ten GEO bands, ERA5 context, SAR targets, split manifests, and
`stats.json`. The [dataset guide](../data/index.md) explains data access. If you
have source observations instead, [export them first](../data/export-geo-sar.md).

Replace `/path/to/geo_sar` below with the export root. A Hugging Face metadata
catalog alone is not a training root.

## Inspect a batch

```python
from geo2wf.config import compose_config, instantiate_datamodule

config = compose_config([
    "data=geo_sar_common10_era5",
    "model=deterministic_residual",
    "data.root=/path/to/geo_sar",
    "data.stats_file=/path/to/geo_sar/stats.json",
    "data.loader.num_workers=0",
])
datamodule = instantiate_datamodule(config)
datamodule.setup("fit")
print(datamodule.data_spec)
batch = next(iter(datamodule.train_dataloader()))
for key in ("condition", "target", "condition_mask", "target_mask"):
    print(key, tuple(batch[key].shape))
```

The default ERA5 condition has 23 channels before model-specific masks and
helpers. The [data contract](../data/dataset-contract.md) describes each field.

## Run one training and validation batch

```bash
WANDB_DISABLED=true uv run geo2wf-train \
  data=geo_sar_common10_era5 model=deterministic_residual \
  data.root=/path/to/geo_sar \
  data.stats_file=/path/to/geo_sar/stats.json \
  data.loader.num_workers=0 \
  trainer.accelerator=cpu trainer.devices=1 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1 trainer.limit_val_batches=1 \
  trainer.enable_checkpointing=false
```

The printed run directory under `logs/` contains `resolved-config.yaml`,
`run-manifest.json`, and `metrics/metrics.csv`. Check that `train/loss` and
`val/loss` are finite. This checks local data/model compatibility; it does not
reproduce the paper's trained models. Continue with [training and checkpoints](../experiments/training.md).
