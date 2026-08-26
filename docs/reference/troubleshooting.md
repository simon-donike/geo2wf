# Troubleshooting

## Manifest or statistics file is missing

**Symptom:** `FileNotFoundError` for `manifest.csv` or `stats.json`.

Training consumes an already exported root. Confirm `data.root/<split>/manifest.csv` and the configured `stats_file` exist. `prepare_data()` intentionally does not download or export anything. Run the matching [exporter](../data/index.md) first or point the config to an existing export.

## U-Net channel mismatch

**Symptom:** convolution expects a different input width.

Channel keys do not mean the same layer of assembly in every model. For the
current common10 + ERA5 data, use these exact checks:

```text
data condition          = 10 GEO + 9 ERA5 + distance + 3 solar = 23
ERA5-residual U-Net     = 23 data + condition mask + ERA5 wind + mask = 26
direct PMW U-Net        = 23 data + condition mask = 24
```

Deterministic and direct U-Net `condition_channels` describe only
`batch["condition"]`. Follow the selected model page and
`DataSpec` error instead of copying a width from another family.

Checkpoints trained before the distance channel was added have a narrower first
convolution and are not shape-compatible with current configs. Retrain with the
current data/model pair.

## No `val/eye_structure_score`

The composite is logged only when eye MAE, inner-core MAE, radial-profile MAE,
RMW error, and eye-to-eyewall contrast are available. Sparse swaths or limited
reconstruction coverage may omit a component. Increase
`model.validation_reconstruction_batches` or monitor a consistently available
metric during debugging.

## ERA5 samples disappear

`require_era5: true` removes rows without a context path. `max_era5_time_gap_hours` also removes missing or stale context timestamps. Inspect manifest `context_path`/`era5_path` and time-gap columns, then re-export if needed.

## Validation is unexpectedly slow

Reduce `model.validation_reconstruction_batches`, loader workers, or validation
batch limits while diagnosing the bottleneck.

## W&B still starts

Set `WANDB_DISABLED=true` before launching, or set `logging.wandb.enabled: false`. `WANDB_MODE=offline` does not disable W&B; it records locally.

## Dataloader workers hang or overload the node

Set `num_workers: 0` to isolate multiprocessing issues. On DDP, worker count applies per rank. Disable `persistent_workers` when worker count is zero, and keep numerical thread counts bounded.

## NaNs or all-zero conditions

The loader rejects condition rasters that look like all-zero fill, applies internal masks, and replaces remaining non-finite normalized values. Inspect GeoTIFF band descriptions, internal mask, manifest paths, and `stats.json`; do not treat raw zeros as proof of valid data.

## Test results are not held out

Modular data configs default to `include_test_in_train: false`; some historical
full-YAML presets set it to `true`. Inspect `resolved-config.yaml`. Changing the
flag after training cannot restore a held-out test.

## Dashboard model name has no training config

StormSense includes imported ViT inference and external ConvLSTM forecast
artifacts. They are not maintained packages under `src/geo2wf/models/`, so a
dashboard label alone is not a reproducible model definition. Use the U-Net,
intensity-correction, or intensity-forecast configs for maintained
training workflows; treat imported artifacts as fixed comparison layers.
