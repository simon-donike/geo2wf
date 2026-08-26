# Training & checkpoints

## Launch a composed run

```bash
uv run geo2wf-train \
  data=geo_sar_common10_era5 \
  model=deterministic_residual
```

Model switching is configuration, not Python dispatch:

```bash
uv run geo2wf-train model=direct_unet
uv run geo2wf-train model=bottleneck_unet_mlp
```

## Startup sequence

1. Load machine-local environment values and bound numerical-library threads.
2. Compose the selected groups and resolve environment interpolation.
3. Create one timestamped run directory, reused by DDP child processes.
4. Save `resolved-config.yaml`, source provenance, and `run-manifest.json`.
5. Seed Python, PyTorch, and DataLoader workers through Lightning.
6. Instantiate the data module and model from their local `_target_` values.
7. Build `DataSpec` and reject incompatible channel/companion contracts.
8. Configure CSV logging, optional W&B, callbacks, scheduler, and checkpoints.
9. Call `trainer.fit(model, datamodule=..., ckpt_path=...)`.

## Resume a run

`--ckpt-path` restores model weights and Lightning training state: optimizer,
scheduler, callbacks, epoch, and global step.

```bash
uv run geo2wf-train \
  model=deterministic_residual \
  --ckpt-path /path/to/last.ckpt
```

The selected model and data configuration must still match the checkpoint.

## Initialize weights only

Use `--weights-only-path` for transfer learning. It strict-loads the
state dictionary but starts optimizer, scheduler, epoch, and step state fresh.
It is mutually exclusive with `--ckpt-path`.

```bash
uv run geo2wf-train \
  model=deterministic_residual \
  --weights-only-path /path/to/source.ckpt
```

Changed condition widths or architecture keys still fail strict loading. A
partial-load policy must be an explicit model-specific migration, not an
implicit training flag.

## Checkpoint selection

When `trainer.checkpoint.monitor` is null, the model supplies its standard
monitor and mode. The callback writes under `<run>/checkpoints/` using the
configured filename, top-k count, and `save_last` policy.

A monitor must be emitted for the validation coverage in use. Very small
validation limits can omit storm metrics when no sample satisfies their
coverage gates; use `val/loss` temporarily or increase coverage for a smoke run.

## Logging and run artifacts

Every run creates:

```text
<default_root_dir>/<timestamp>_modular/
├── checkpoints/
├── metrics/metrics.csv
├── resolved-config.yaml
├── run-manifest.json
├── source-diff.patch
├── source-snapshot/
└── wandb/                         # only used when W&B is active/offline
```

The run manifest records status, resolved config, checkpoint provenance, split
policy, git/source state, runtime metadata, final metrics, and failures. CSV
logging and manifests do not depend on W&B.

## W&B modes

```bash
export WANDB_DISABLED=true   # no W&B logger
export WANDB_MODE=offline    # local W&B files, no online traffic
```

Models import neither W&B nor Matplotlib; reconstruction payloads are routed through
the tracking layer, whose callback can also drain standardized events.

## Resume safety

- Resume only with the same architecture, channel order, schedule, and target definition.
- Changed optimizer only: use weights-only initialization if intentional.
- Changed target normalization: start a fresh run.
- Changed bands, companions, or spatial contract: select compatible config and checkpoint.
- Older compatible checkpoints remain strict-loadable; only new checkpoints receive `geo2wf` metadata.
