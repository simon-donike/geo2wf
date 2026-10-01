# Training and checkpoints

Install the package and obtain a compatible [local dataset](../data/dataset-contract.md).
Choose a retained [model preset](../models/index.md#training-presets), then
pass local paths explicitly. For a joint field/intensity run:

```bash
WANDB_DISABLED=true uv run geo2wf-train \
  experiment=bottleneck_unet_mlp \
  data.root=/path/to/paired \
  data.ibtracs_file=/path/to/ibtracs.ALL.list.v04r01.csv
```

For a field-only run without scalar target filtering:

```bash
WANDB_DISABLED=true uv run geo2wf-train \
  data=geo_sar_common10_era5 model=deterministic_residual \
  data.root=/path/to/paired data.stats_file=/path/to/paired/stats.json
```

These start new training runs. Exact checkpoint reproduction also requires
its original configuration, cohort membership, statistics, and producer files.

## Configuration and hardware

`configs/modular.yaml` composes `data`, `model`, `trainer`, `logging`, and an
optional `experiment`. Choices are the YAML filenames under each group;
dotted overrides change values for one invocation.

| Override | Purpose |
|---|---|
| `data.root`, `data.stats_file` | Local export/cache and its normalization statistics |
| `data.require_era5`, `data.use_era5` | Availability filtering and input inclusion, respectively |
| `trainer.max_epochs` | Training duration |
| `trainer.limit_train_batches`, `trainer.limit_val_batches` | Integer batch limit or floating-point fraction |
| `trainer.accelerator`, `trainer.devices`, `trainer.strategy` | Hardware and distributed execution |
| `trainer.checkpoint.monitor` | Checkpoint metric; null uses the model default |
| `trainer.default_root_dir` | Parent of timestamped run directories |

The basic paired data config uses `data.loader.batch_size` and
`data.loader.num_workers`. Joint, correction, and forecast configs expose
`data.batch_size` and `data.num_workers` directly. Inspect the selected YAML
before overriding these keys.

For two allocated GPUs, append:

```text
trainer.accelerator=gpu trainer.devices=2 trainer.strategy=ddp_find_unused_parameters_false
```

Batch size and workers are per process. Start with zero workers for data-loader
debugging. `DataSpec` validates ordered channels, units, and required companions
before the first batch.

## Resume or initialize

Append `--ckpt-path /path/to/last.ckpt` to restore weights, optimizer, scheduler,
callbacks, epoch, and step. Append `--weights-only-path /path/to/model.ckpt`
for strict weight loading with fresh training state. The flags are mutually
exclusive; both require a matching model architecture and channel contract.

Changing normalization or targets can change model meaning even when tensor
shapes match. Use the checkpoint's resolved configuration when continuing an
existing experiment.

## Checkpoints and run records

Each run writes a timestamped directory:

```text
logs/<timestamp>_modular/
├── checkpoints/
├── metrics/metrics.csv
├── resolved-config.yaml
├── run-manifest.json
├── source-diff.patch
└── source-snapshot/
```

The manifest records configuration, source/checkpoint provenance, split policy,
status, and metrics. The default checkpoint callback retains the best two and
last checkpoint. A monitor must be logged for the validation coverage used;
a tiny smoke run may not produce every structural metric.

`WANDB_DISABLED=true` keeps CSV/run records without W&B. `WANDB_MODE=offline`
keeps local W&B artifacts. Early stopping and learning-rate scheduling have
separate monitors and patience; the resolved config is authoritative.

Continue with [evaluation metrics](evaluation.md) and the
[command reference](../reference/commands.md).
