# Troubleshooting

## Manifest or statistics file is missing

**Symptom:** `FileNotFoundError` for `manifest.csv` or `stats.json`.

Training consumes an already exported root. Confirm `data.root/<split>/manifest.csv` and the configured `stats_file` exist. `prepare_data()` intentionally does not download or export anything. Run the matching [exporter](../data/index.md) first or point the config to an existing export.

## U-Net channel mismatch

**Symptom:** convolution expects a different input width.

Recalculate:

```text
prepared condition = GEO + optional ERA5 + distance-to-center + 3 solar-time fields + condition mask
unet.channels      = prepared condition + noisy target
```

The residual config is different: `condition_channels` includes distance-to-center but excludes the three internally appended mask/baseline features.

Checkpoints trained before the distance channel was added have a narrower first
convolution and are not shape-compatible with current configs. Retrain with the
current data/model pair. For deterministic-baseline residual diffusion, its
Stage 1 checkpoint must use the exact input assembly declared by the Stage 2
configuration.

## No `val/eye_structure_score`

The composite is logged only when eye MAE, inner-core MAE, radial-profile MAE,
RMW error, and eye-to-eyewall contrast are available. Sparse swaths or limited
reconstruction coverage may omit a component. Increase
`model.validation_reconstruction_batches` or monitor a consistently available
metric during debugging.

## Resume rejects diffusion coefficients

The checkpoint was trained with a different schedule or timestep count. Restore the original `schedule` and `num_timesteps`, or start a new run. Do not bypass the check: coefficients define the model’s training clock.

## ERA5 samples disappear

`require_era5: true` removes rows without a context path. `max_era5_time_gap_hours` also removes missing or stale context timestamps. Inspect manifest `context_path`/`era5_path` and time-gap columns, then re-export if needed.

## Validation is unexpectedly slow

A diffusion loss needs one U-Net pass; reconstruction requires one pass per
reverse step. Reduce `model.validation_reconstruction_batches`, use the
100-step DDIM setting, or limit validation batches for smoke tests.

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
