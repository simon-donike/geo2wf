# Configuration reference

This page describes the composed training schema first. Historical full YAML
keys are retained in a separate compatibility section.

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
| `include_pmw`, `include_ibtracs` | request optional companions/metadata |
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

## Modular diffusion `model`

Both standalone and residual diffusion use descriptive flat constructor keys.

| Key | Purpose |
|---|---|
| `_target_` | model constructor or model-specific factory |
| `condition_channels` / `base_condition_channels` | prepared/base condition width |
| `generated_channels` | target/residual output width |
| `num_timesteps`, `schedule` | forward process |
| `model_dim`, `model_dim_mults`, `model_channels`, `model_out_dim` | U-Net sizing |
| `sampling_method`, `sampling_timesteps`, `sampling_eta` | reverse sampler |
| `guidance_scale`, `condition_dropout_probability` | classifier-free guidance |
| `clip_sample` | clip the clean estimate during reverse sampling |
| `ema_decay`, `ema_update_after_step`, `ema_use_for_eval` | EMA behavior |
| `min_snr_gamma` | optional epsilon-prediction Min-SNR cap |
| `lr`, `lr_scheduler_*` | optimizer and scheduler |
| `validation_seed`, `validation_ensemble_size`, `validation_ensemble_batches` | stable validation members |
| `validation_reconstruction_batches`, `log_reconstruction_images` | reconstruction/media coverage |

Residual diffusion additionally accepts:

| Key | Purpose |
|---|---|
| `baseline_source` | `era5` or `deterministic` |
| `baseline_checkpoint_path` | frozen Stage 1 checkpoint; environment-backed choice uses `GEO2WF_BASELINE_CKPT` |
| `residual_transform`, `residual_soft_scale_ms`, `residual_clip_ms` | signed transform |
| `prediction_min_ms`, `prediction_max_ms` | recomposed output bounds |
| `*_loss_weight` and related thresholds/kernels | optional gradient, spectral, low-frequency, smoothness, peak, radial, exceedance, multiscale, annular objectives |
| `sparse_target_fill`, `unobserved_loss_weight` | weak off-swath supervision |
| `probabilistic_score_*` | ensemble checkpoint-score composition |

Inspect the selected file in `configs/model/` for authoritative defaults.

## Other modular models

| Model choice | Contract-defining keys |
|---|---|
| `direct_unet` | `condition_channels`, U-Net sizing, `huber_delta_k`; target must be one channel in K |
| `bottleneck_unet_mlp` | image/intensity Huber deltas and weights, intensity MLP sizing, optional structure head and loss |
| `bottleneck_encoder_mlp` | encoder/MLP sizing and scalar Huber delta; no image decoder |
| `intensity_correction` | field/metadata toggles, `anchor_statistic`, robust-peak fraction, scalar Huber delta, optional structure head |
| `intensity_forecast` | five-feature MLP sizing, dropout, scalar Huber delta, optimizer/scheduler settings |

Target source and cohort are primarily data-contract choices, not model-name
implications. In particular, joint and correction datasets can represent
IBTrACS intensity or a SAR robust peak, while the direct U-Net requires PMW
brightness temperature. Inspect both selected YAML files and the resulting
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

## Legacy full-YAML reference

Legacy configs remain accepted through `--config`. Their top-level sections are:

| Section | Translation |
|---|---|
| `export` | maintained exporter defaults |
| `data` | adapted to `PairedDataModule.from_config` |
| `model.type` | compatibility model factory only |
| `model.unet`, `model.sampling`, `model.residual` | translated into model constructor arguments |
| `optimization` | translated into optimizer, scheduler, EMA, and objective arguments |
| `validation` | translated into model validation/sampling settings |
| `trainer`, `logging` | consumed by the shared training runtime |

These files may also contain PMW keys such as `pmw_as_condition`,
`max_pmw_time_gap_hours`, and `pmw_include_time_offset`. They preserve historical
experiments but are not templates for new grouped configs. A full YAML file
cannot be combined with Hydra overrides.

## Export configuration

Export currently retains its established argparse/full-YAML interface. Relevant
keys include source/manifest/output paths, channel set, splits, grid size and
resolution, closest-match limits, PMW/IBTrACS/ERA5 inclusion and freshness,
crop center/shift/padding, and a per-split `limit`. Explicit command flags take
precedence. See [Export GEO–SAR](../data/export-geo-sar.md).
