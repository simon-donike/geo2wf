# Training & checkpoints

## Launch a run

```bash
uv run python train.py --config configs/config_geo_sar_10bands_era5.yaml
```

Optional flags:

```bash
uv run python train.py \
  --config configs/config.yaml \
  --limit-val-batches 5 \
  --ckpt-path /path/to/last.ckpt
```

## Startup sequence

1. Machine-local environment values are loaded.
2. BLAS thread counts and Matplotlib cache location are bounded.
3. YAML is parsed and float32 matrix-multiply precision is set.
4. A timestamped run directory is created and shared with DDP child processes.
5. W&B directories are nested inside that run.
6. Python/PyTorch/worker seeds are set through Lightning.
7. Data and model are built.
8. Callbacks and trainer are configured.
9. `trainer.fit(..., ckpt_path=...)` begins training or resume.

## What gets optimized

The diffusion model uses AdamW over trainable parameters in its online diffusion process. The EMA copy never receives gradients. The residual model uses AdamW with configurable weight decay. Both use `ReduceLROnPlateau` in `min` mode.

## Logging

When enabled, W&B receives:

- step/epoch training loss and backward-step counter;
- validation loss and reconstruction metrics;
- physical and storm-centric metrics;
- georeferenced condition/prediction/target panels with masks, centers, footprints, and optional ERA5 wind;
- training-subset reconstruction panels; and
- optimizer learning rate when the logger is active.

Disable external logging with either:

```bash
export WANDB_MODE=offline
```

or:

```bash
export WANDB_DISABLED=true
```

`WANDB_DISABLED` also prevents construction of the logger. Offline mode retains local W&B artifacts for later sync.

## Resume safety

A Lightning checkpoint restores model, optimizer, scheduler, global step, and the custom backward-step counter. Diffusion resume additionally compares saved beta coefficients with the configured forward process. A mismatch raises an error.

Use this rule:

- same schedule, timestep count, channels, and architecture → resume is plausible;
- changed schedule or target definition → start fresh;
- added EMA to an older compatible checkpoint → supported; EMA is initialized from online weights;
- changed input bands or context width → treat as transfer learning, not resume.

## Checkpoint selection

ERA5 production configs save the best two checkpoints by the lowest `val/eye_structure_score` plus `last.ckpt`. This composite weights eye MAE 0.5, inner-core MAE 1, radial-profile MAE 1, RMW error 0.1, and eye-to-eyewall contrast error 1.

A lower score is better. Eye-center displacement is logged when gated quality conditions pass but is not included in this composite.
