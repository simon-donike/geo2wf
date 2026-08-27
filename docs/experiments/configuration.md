# Configuration guide

Training composes small Hydra groups instead of copying one complete experiment
file. `configs/modular.yaml` is the composition root:

```yaml
defaults:
  - data: geo_sar_common10_era5
  - model: deterministic_residual
  - trainer: default
  - logging: default
  - optional experiment: null
  - _self_

seed: 42
```

## Groups and choices

A choice is the YAML filename without `.yaml`.

| Group | Checked-in choices | Owns |
|---|---|---|
| `data` | paired raster, joint-intensity, correction-cache, and forecast-cache choices | roots, splits, normalization, companions, loader, sampling |
| `model` | reconstruction, joint-intensity, correction, and forecast choices | constructor, architecture, objective, optimizer |
| `trainer` | `default` | devices, precision, loop bounds, checkpoint policy |
| `logging` | `default` | optional W&B adapter |
| `experiment` | retained task-specific data/model combinations | focused cross-group overrides only |

List filenames below `configs/<group>/` to discover new choices. Each data and
model config has a local `_target_`; adding one does not require a central
runtime dispatch table.

## Select groups

```bash
uv run geo2wf-train \
  data=geo_sar_common10_era5 \
  model=deterministic_residual
```

## Override values

Use dotted keys for one run:

```bash
uv run geo2wf-train \
  model=deterministic_residual \
  trainer.devices=2 \
  trainer.strategy=ddp_find_unused_parameters_false \
  data.loader.batch_size=2 \
  logging.wandb.name=stage1-two-gpu
```

Use `null`, booleans, lists, and strings with Hydra syntax. Quote shell-sensitive
values. Unknown keys require Hydra's explicit `+new.key=value` syntax. An
unrecognized runtime key has no effect unless the selected target consumes it.

## Channel compatibility

`DataSpec` records ordered condition and target channels, spatial shape, units,
and available companions. Training calls `model.validate_data_spec(...)`
before the first batch. A mismatch therefore reports the model width and actual
channel names during startup.

The checked-in ERA5 data provides 23 data-condition channels:

```text
10 GEO + 9 ERA5 + distance + 3 solar = 23
```

Model configs distinguish those data channels from masks, baselines, and noisy
target channels appended internally. Do not copy channel values between model
families without following their assembly description.

## Trainer and checkpoint semantics

`trainer.limit_train_batches` and `trainer.limit_val_batches`
: An integer means that many batches. A float in `[0,1]` means a fraction.

`trainer.devices`
: Passed to Lightning. Requesting two devices still requires two usable devices.

`trainer.checkpoint.monitor`
: Overrides the model's `checkpoint_monitor`. When null, the model default is
  used. The named metric must be logged during every eligible validation epoch.

`trainer.default_root_dir`
: Parent of timestamped run directories. It is not the final run directory.

## Logging configuration

```yaml
logging:
  wandb:
    enabled: true
    project: geo2wf
    name: null
    log_model: false
```

`WANDB_DISABLED=true` disables W&B construction regardless of config. CSV
metrics, run manifests, resolved configuration, and checkpoints remain active.
`WANDB_MODE=offline` retains local W&B artifacts for later synchronization.

## Resolved configuration

Every run writes `resolved-config.yaml` before training and records its absolute
path in `run-manifest.json`. Environment-backed checkpoint paths are materialized
there, making the actual run input inspectable.

## Archived full YAML files

Historical full-YAML presets are preserved under `archived/configs/` for
provenance. Active training requires grouped configs with a model `_target_`;
archived presets are not accepted as current launch choices. Use git history or
an older checkout for exact historical reproduction.
