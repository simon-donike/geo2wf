# Troubleshooting

## Manifest or statistics file is missing

**Symptom:** `FileNotFoundError` for `manifest.csv` or `stats.json`.

Training consumes an already exported root. Confirm `data.root/<split>/manifest.csv` and the configured `stats_file` exist. `prepare_data()` intentionally does not download or export anything. Run the matching [exporter](../data/index.md) first or point the config to an existing export.

## U-Net channel mismatch

**Symptom:** convolution expects a different input width.

Recalculate:

```text
prepared condition = GEO + optional ERA5 + 1 condition mask
unet.channels      = prepared condition + noisy target
```

The residual config is different: `condition_channels` excludes the three internally appended mask/baseline features.

## No `val/eye_structure_score`

The composite is logged only when eye MAE, inner-core MAE, radial-profile MAE, RMW error, and eye-to-eyewall contrast are all available. Sparse swaths or too few reconstruction batches may omit a component. Increase `validation.reconstruction_batches`/validation coverage or monitor a reliably present metric during debugging.

## Resume rejects diffusion coefficients

The checkpoint was trained with a different schedule or timestep count. Restore the original `schedule` and `num_timesteps`, or start a new run. Do not bypass the check: coefficients define the model’s training clock.

## ERA5 samples disappear

`require_era5: true` removes rows without a context path. `max_era5_time_gap_hours` also removes missing or stale context timestamps. Inspect manifest `context_path`/`era5_path` and time-gap columns, then re-export if needed.

## Validation is unexpectedly slow

A diffusion loss needs one U-Net pass; reconstruction needs 100–1000 sequential passes. Reduce `validation.reconstruction_batches`, use deterministic 100-step DDIM, or limit validation batches for smoke work.

## W&B still starts

Set `WANDB_DISABLED=true` before launching, or set `logging.wandb.enabled: false`. `WANDB_MODE=offline` does not disable W&B; it records locally.

## Dataloader workers hang or overload the node

Set `num_workers: 0` to isolate multiprocessing issues. On DDP, worker count applies per rank. Disable `persistent_workers` when worker count is zero, and keep numerical thread counts bounded.

## NaNs or all-zero conditions

The loader rejects condition rasters that look like all-zero fill, applies internal masks, and replaces remaining non-finite normalized values. Inspect GeoTIFF band descriptions, internal mask, manifest paths, and `stats.json`; do not treat raw zeros as proof of valid data.

## Test results are not held out

All checked-in training presets use `include_test_in_train: true`. Set it to `false` before training when final test generalization is required. Changing it after training cannot restore a held-out test.
