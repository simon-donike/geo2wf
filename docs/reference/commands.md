# Commands & environment

## Environment and quality

```bash
uv sync --frozen --group dev --group docs
uv run python -m pytest
uv run mkdocs build --strict
uv run mkdocs serve
```

`uv sync` installs four commands: `geo2wf-train`, `geo2wf-evaluate`,
`geo2wf-infer`, and `geo2wf-export`.

## Training

```bash
# Default: common10 + ERA5 data, deterministic residual model
uv run geo2wf-train

# Explicit model/data selection
uv run geo2wf-train \
  data=geo_sar_common10_era5 \
  model=deterministic_residual

# ERA5-baseline residual diffusion
uv run geo2wf-train model=residual_diffusion

# Frozen deterministic-baseline residual diffusion
GEO2WF_BASELINE_CKPT=/path/to/stage1.ckpt \
uv run geo2wf-train model=residual_diffusion_deterministic_baseline
```

Smoke and DDP overrides:

```bash
WANDB_DISABLED=true uv run geo2wf-train \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=1 \
  trainer.enable_checkpointing=false

uv run geo2wf-train \
  trainer.devices=2 \
  trainer.strategy=ddp_find_unused_parameters_false \
  data.loader.batch_size=2
```

Resume or initialize only weights:

```bash
uv run geo2wf-train --ckpt-path /path/to/last.ckpt
uv run geo2wf-train --weights-only-path /path/to/source.ckpt
```

## Data export

Export commands select a workflow first and then pass its established flags.
They do not currently accept training's Hydra override syntax.

```bash
# Tiny GEO–SAR structural export
uv run geo2wf-export geo-sar \
  --config configs/config.yaml \
  --limit 2

# ERA5-enriched GEO–SAR export
uv run geo2wf-export geo-sar \
  --config configs/config_geo_sar_10bands_era5.yaml

# GEO–PMW proxy export
uv run geo2wf-export geo-pmw \
  --config configs/config_pretrain_geo_pmw_10bands_era5.yaml \
  --limit 2
```

Explicit flags such as `--data-root`, `--manifest-file`, `--output-root`,
`--splits`, `--geo-channel-set`, `--include-era5`, and `--limit` override values
loaded from the export config.

## Checkpoint evaluation

The evaluator compares one checkpoint on the PMW-matched validation cohort and
writes a machine-readable JSON report.

```bash
uv run geo2wf-evaluate \
  --config configs/config_geo_sar_10bands_era5_residual.yaml \
  --checkpoint /path/to/stage1.ckpt \
  --data-root data/geotiff/geo_sar_10bands_era5_v2_pmw \
  --pmw-max-time-gap-hours 1 \
  --output logs/stage1-common-pmw.json
```

Useful controls are `--stats`, `--accelerator`, `--device`, and
`--limit-batches`. Use `--limit-batches 1.0` for a full comparison report.

Promotion comparison remains a focused analysis command:

```bash
uv run python scripts/compare_pmw_evaluations.py \
  --stage 1 \
  --current logs/current-stage1-common-pmw.json \
  --candidate logs/pmw-stage1-common-pmw.json \
  --output logs/pmw-stage1-promotion.json
```

## Storm inference

```bash
uv run geo2wf-infer deterministic-residual \
  --config configs/config_geo_sar_10bands_era5_residual.yaml \
  --checkpoint /path/to/stage1.ckpt \
  --storms AL082025 EP112025 \
  --output-root inference/inf_simon

uv run geo2wf-infer residual-diffusion \
  --config configs/config_geo_sar_10bands_era5_diffusion_residual_deterministic.yaml \
  --checkpoint /path/to/stage2.ckpt \
  --ensemble-size 10 \
  --sampling-seed 42 \
  --summary-aggregation medoid \
  --output-root inference/inf_simon_diffusion
```

Both workflows also accept `--data-root`, `--manifest`, `--reference-root`,
`--stats`, `--device`, `--storms`, and `--limit`. Diffusion additionally accepts
batch, guidance, ensemble-summary, member-quantile, and member-field controls.
PMW-aware runs write `pmw-inference-audit.csv` beside their summaries.

## Batch jobs

```bash
qsub scripts/hpc/export_geo_pmw_geotiffs_cpu.pbs
qsub scripts/hpc/export_geo_sar_geotiffs_cpu.pbs
```

Review account, queue, resources, and paths before submitting these site-specific
PBS templates.

## Environment variables

| Variable | Effect |
|---|---|
| `TCD_DATA_ROOT` | source archive root used by maintained exporters |
| `GEO_SAR_OUTPUT_ROOT` | conventional GEO–SAR export destination override |
| `GEO_PMW_OUTPUT_ROOT` | conventional GEO–PMW export destination override |
| `GEO2WF_BASELINE_CKPT` | Stage 1 checkpoint for the deterministic-baseline Stage 2 config |
| `GEO2WF_RUN_DIR` | inherited DDP run path; normally managed internally |
| `WANDB_DISABLED` | disable W&B construction when true-like |
| `WANDB_MODE` | `offline` retains local W&B artifacts |
| `WANDB_PROJECT`, `WANDB_NAME` | override configured run tracking names |
| `WANDB_DIR`, `WANDB_CACHE_DIR`, `WANDB_CONFIG_DIR` | set below each training run |
| `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` | default to 1 in runtime scripts |
| `MPLCONFIGDIR` | defaults to `/tmp/dif_img_rec_matplotlib` |

## Compatibility commands

The root scripts remain forwarding adapters, so existing automation continues
to work:

```bash
uv run python train.py --config configs/config.yaml
uv run python scripts/evaluate_checkpoint.py --help
uv run python scripts/export_geo_sar_geotiffs.py --help
```

Use installed `geo2wf-*` commands in new documentation and launchers.
