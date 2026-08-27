# Configuration reference

This page describes the active composed training schema. Historical full-YAML
presets are listed in the archive rather than accepted as active choices.

## Composition root

| Key/group | Purpose |
|---|---|
| `seed` | global Lightning and worker seed |
| `data=<choice>` | instantiate a data module from `configs/data/` |
| `model=<choice>` | instantiate a model from `configs/model/` |
| `trainer=<choice>` | Lightning runtime and checkpoint settings |
| `logging=<choice>` | tracking adapters |
| `experiment=<choice>` | optional focused overrides |

`_target_` is owned locally by each data/model choice and is passed to Hydra's
instantiation mechanism.

## Modular `data`

| Key | Purpose |
|---|---|
| `_target_` | data-module constructor/factory |
| `root`, `stats_file` | exported dataset and training statistics |
| `train_split`, `val_split`, `test_split` | split directory names |
| `target_size`, `center_crop_size` | output and optional final crop shape |
| `random_flips` | paired physics-aware train augmentation |
| `include_test_in_train` | explicitly merge test into train; modular default is false |
| `require_era5`, `use_era5` | filter for ERA5 availability and include ERA5 as model input, respectively |
| `include_ibtracs` | request optional best-track metadata |
| `normalization`, `target_normalization` | condition and target transforms |
| `robust_clip`, `max_era5_time_gap_hours` | robust range and context freshness |
| `loader.batch_size`, `num_workers` | per-process loader size/workers |
| `loader.pin_memory`, `persistent_workers` | loader memory/lifetime behavior |
| `sampling.intensity_balanced.enabled` | use intensity-aware train sampling |

Dataset implementations may add focused keys, but their resulting capabilities
must be represented by `DataSpec`.

## Modular deterministic-residual `model`

| Key | Purpose |
|---|---|
| `_target_` | `ERA5ResidualRegressor` constructor |
| `condition_channels`, `base_channels`, `channel_mults` | data width and U-Net sizing |
| `huber_delta_ms`, `off_swath_anchor_weight` | physical residual loss |
| `high_wind_*`, `peak_*` | intensity weighting and robust peak objective |
| `radial_profile_*`, `exceedance_area_*` | optional structural objectives |
| `prediction_min_ms`, `prediction_max_ms` | physical output bounds |
| `psnr_data_range_ms` | physical PSNR range |
| `lr`, `weight_decay` | AdamW settings |
| `lr_scheduler_*` | ReduceLROnPlateau settings and monitor |
| `validation_reconstruction_batches`, `log_reconstruction_images` | validation media coverage |

## Other modular models

| Model choice | Contract-defining keys |
|---|---|
| `bottleneck_unet_mlp` | image/intensity Huber deltas and weights, intensity MLP sizing, optional structure head and loss |
| `intensity_correction` | field/metadata toggles, `anchor_statistic`, robust-peak fraction, scalar Huber delta, optional structure head |
| `intensity_forecast` | five-feature MLP sizing, dropout, scalar Huber delta, optimizer/scheduler settings |

Target source and cohort are primarily data-contract choices, not model-name
implications. Inspect both selected YAML files and the resulting
`DataSpec`/cache metadata.

## `trainer`

| Key | Purpose |
|---|---|
| `max_epochs` | epoch limit |
| `accelerator`, `devices`, `strategy` | hardware and distributed execution |
| `precision`, `float32_matmul_precision` | numerical mode |
| `deterministic` | Lightning deterministic-algorithm request |
| `log_every_n_steps` | step logging interval |
| `enable_checkpointing`, `default_root_dir` | artifacts and run parent |
| `limit_train_batches`, `limit_val_batches` | bounded loops |
| `checkpoint.monitor`, `mode` | selection metric and direction |
| `checkpoint.save_top_k`, `save_last`, `filename` | retention/naming policy |

## `logging`

| Key | Purpose |
|---|---|
| `wandb.enabled` | construct W&B unless disabled by environment |
| `wandb.project`, `wandb.name` | run destination and display name |
| `wandb.log_model` | W&B checkpoint logging policy |

CSV metrics and run manifests are always configured independently of W&B.

## Archived configuration

Retired full-YAML, diffusion, PMW-proxy, and ablation presets are stored under
`archived/configs/`. They are preserved for provenance and excluded from Hydra's
active config search path.

## Export configuration

Export currently retains its established argparse/full-YAML interface. Relevant
keys include source/manifest/output paths, channel set, splits, grid size and
resolution, closest-match limits, PMW/IBTrACS/ERA5 inclusion and freshness,
crop center/shift/padding, and a per-split `limit`. Explicit command flags take
precedence. See [Export GEO–SAR](../data/export-geo-sar.md).
