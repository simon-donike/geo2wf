# Configuration reference

## Top level

| Key | Type | Purpose |
|---|---|---|
| `seed` | int | global Lightning/worker seed |
| `export` | mapping | one-time data preparation defaults |
| `data` | mapping | runtime dataset and loader |
| `model` | mapping | selected model and architecture |
| `optimization` | mapping | optimizer/EMA/scheduler |
| `trainer` | mapping | Lightning execution |
| `validation` | mapping | reconstruction sampling coverage |
| `logging.wandb` | mapping | W&B behavior |

## `export`

| Key | Purpose |
|---|---|
| `data_root`, `manifest_file`, `output_root` | source archive, observation index, destination |
| `geo_channel_set` | `common4` or `common10` |
| `splits` | partitions to export |
| `grid_size`, `grid_resolution` | square grid dimensions and degree resolution |
| `closest_match_hours` | maximum target-to-GEO time difference |
| `center` | `image_center` or `ibtracs_center` |
| `shift_center`, `pad` | crop inclusion and source-read padding |
| `limit` | successful samples per split, `null` for all |
| `pmw_sensors` | allowed proxy target platforms |
| `include_era5`, `era5_channels` | context selection |
| `era5_max_time_gap_hours` | context freshness limit |

## `data`

| Key | Purpose |
|---|---|
| `root`, `stats_file` | exported dataset and statistics paths |
| `train_split`, `val_split`, `test_split` | manifest directory names |
| `target_size` | runtime target height/width |
| `center_crop_size` | optional final center crop; distance normalization uses these final bounds |
| `random_flips` | paired, vector-aware augmentation in train |
| `include_test_in_train` | concatenate test into train |
| `require_era5` | filter samples without context |
| `normalization` | `min-max` or `robust-zscore` for conditions |
| `target_normalization` | optional target-specific method |
| `robust_clip`, `target_robust_clip` | symmetric robust range before `[0,1]` mapping |
| `max_era5_time_gap_hours` | runtime stale-context filter |
| `loader.batch_size` | per-process sample count |
| `loader.num_workers` | DataLoader worker processes |
| `loader.pin_memory` | page-lock host tensors |
| `loader.persistent_workers` | keep workers between epochs |

## Diffusion `model`

| Key | Purpose |
|---|---|
| `type: diffusion` | selects diffusion; also the default when absent |
| `in_channels`, `out_channels` | prepared condition and generated widths |
| `num_timesteps` | forward training schedule length |
| `schedule` | linear, cosine, quadratic, or sigmoid |
| `unet.dim`, `dim_mults` | base width and resolution multipliers |
| `unet.channels`, `out_dim` | concatenated U-Net input and prediction width |
| `sampling.method` | `ddpm` or `ddim` |
| `sampling.timesteps` | reverse steps; DDPM must equal training steps |
| `sampling.eta` | DDIM stochasticity |
| `sampling.guidance_scale` | classifier-free guidance strength; `1` disables guidance |
| `classifier_free_guidance.condition_dropout_probability` | complete-condition dropout used to train CFG |
| `sampling.clip_sample` | per-step clean estimate clipping |
| `sparse_target.fill` | `era5` or disabled |
| `sparse_target.unobserved_loss_weight` | weak ERA5 completion weight in `[0,1]` |

## Residual diffusion `model`

| Key | Purpose |
|---|---|
| `type: diffusion_residual` | diffuse a signed physical correction around a dense baseline |
| `residual.baseline.source` | `era5` or `deterministic` |
| `residual.baseline.checkpoint_path` | frozen deterministic checkpoint; may instead use `GEO2WF_BASELINE_CKPT` |
| `residual.soft_scale_ms`, `clip_ms` | odd asinh residual transform parameters |
| `residual.prediction_min_ms`, `prediction_max_ms` | recomposed physical output bounds |
| `residual.loss.gradient_weight`, `spectrum_weight` | sharpness auxiliary weights |
| `residual.loss.low_frequency_weight`, `low_frequency_kernel_size` | broad-field consistency controls |
| `residual.loss.smoothness_weight` | weak total-variation penalty on the physical residual |
| `residual.loss.auxiliary_max_timestep_fraction` | latest normalized training timestep receiving clean-residual auxiliary losses |
| `residual.loss.high_wind_*`, `high_gradient_*` | structural pixel thresholds and weights |
| `residual.loss.inner_core_radius_km`, `inner_core_weight` | storm-centered structural emphasis |
| `unet.channels` | noisy residual + prepared condition + explicit baseline and mask |

## Residual `model`

| Key | Purpose |
|---|---|
| `type: deterministic_residual` | selects residual path |
| `condition_channels` | condition width before masks/baseline are appended |
| `residual.base_channels`, `channel_mults` | compact U-Net sizing |
| `residual.high_wind_threshold_ms` | high-wind metric threshold |
| `residual.prediction_min_ms`, `prediction_max_ms` | physical output bounds |
| `residual.psnr_data_range_ms` | physical PSNR range |

## `optimization`

| Key | Purpose |
|---|---|
| `lr` | AdamW learning rate |
| `min_snr_gamma` | optional epsilon-prediction Min-SNR cap |
| `weight_decay` | residual AdamW decay |
| `huber_delta_ms` | residual Huber transition |
| `off_swath_anchor_weight` | weak ERA5 residual penalty outside SAR |
| `ema.enabled`, `decay` | diffusion EMA activation and coefficient |
| `ema.update_after_step`, `use_for_eval` | EMA warmup/evaluation behavior |
| `reduce_lr_on_plateau.factor`, `patience`, `monitor` | scheduler controls |

## `trainer`, `validation`, and logging

| Key | Purpose |
|---|---|
| `trainer.max_epochs` | epoch limit |
| `accelerator`, `devices`, `strategy` | execution hardware/DDP |
| `precision`, `float32_matmul_precision` | numerical mode |
| `deterministic` | Lightning deterministic algorithms request |
| `log_every_n_steps` | step logging interval |
| `enable_checkpointing`, `default_root_dir` | artifacts |
| `limit_train_batches`, `limit_val_batches` | bounded loops |
| `checkpoint.monitor`, `mode`, `save_top_k`, `save_last`, `filename` | selection policy |
| `validation.reconstruction_batches` | reconstruction-image batches per epoch; expensive reverse sampling for diffusion, direct physical prediction for residual |
| `validation.sampling_seed` | stable per-sample latent namespace |
| `validation.ensemble_size`, `ensemble_batches` | stable validation ensemble width and evaluated prefix |
| `validation.probabilistic_score_sharpness_weight` | sharpness contribution to the CRPS-based checkpoint score |
| `validation.probabilistic_score_target_sharpness_ratio` | preferred sampled/observed gradient ratio; values below one select smoother samples |
| `logging.wandb.enabled`, `project`, `name`, `save_dir`, `log_model` | tracking configuration |
