# Commands & environment

## Environment and quality

```bash
uv sync --frozen --group dev --group docs
uv run python -m pytest
uv run mkdocs build --strict
```

## Active training

```bash
# Raw field U-Net: matched with/without-ERA5 pair
uv run geo2wf-train experiment=intensity_comparison_unet
uv run geo2wf-train experiment=intensity_comparison_unet_no_era5

# Joint U-Net + bottleneck MLP: matched pair
uv run geo2wf-train experiment=bottleneck_unet_mlp
uv run geo2wf-train experiment=bottleneck_unet_mlp_no_era5

# Joint U-Net/latent MLP structure pair
uv run geo2wf-train experiment=bottleneck_unet_mlp_max_wind
uv run geo2wf-train experiment=bottleneck_unet_mlp_max_wind_radii

# Scalar correction and retained forecast
uv run geo2wf-train experiment=unet_intensity_correction
uv run geo2wf-train experiment=intensity_forecast_pretrain
uv run geo2wf-train experiment=intensity_forecast_finetune
```

Use normal Hydra overrides for smoke tests, hardware, and loader settings:

```bash
WANDB_DISABLED=true uv run geo2wf-train \
  experiment=intensity_comparison_unet \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=1 \
  trainer.enable_checkpointing=false

uv run geo2wf-train \
  experiment=bottleneck_unet_mlp \
  trainer.devices=2 \
  trainer.strategy=ddp_find_unused_parameters_false
```

Resume full state with `--ckpt-path`; initialize weights only with
`--weights-only-path`.

## Data and cache export

```bash
uv run geo2wf-export geo-sar \
  --config configs/config.yaml \
  --limit 2

uv run geo2wf-export intensity-cache \
  --config /path/to/unet-run/resolved-config.yaml \
  --checkpoint /path/to/unet.ckpt \
  --output-root data/unet_intensity

uv run geo2wf-export intensity-forecast-cache --help
```

## Evaluation and inference

```bash
uv run geo2wf-evaluate latent-structure --help
uv run geo2wf-evaluate intensity-comparison --help
uv run geo2wf-evaluate three-storm-nowcasts --help
uv run geo2wf-evaluate intensity-correction --help
uv run geo2wf-evaluate intensity-forecast --help

uv run geo2wf-infer deterministic-residual \
  --config /path/to/unet-run/resolved-config.yaml \
  --checkpoint /path/to/unet.ckpt

uv run geo2wf-infer intensity-correction --help
uv run geo2wf-infer intensity-comparison-storms --help
uv run geo2wf-infer intensity-forecast --help
```

Generate the dense validation-storm nowcasts after both comparison workflows
finish. The optional ablation checkpoints are ERA5-conditioned:

```bash
uv run geo2wf-infer intensity-comparison-storms \
  --era5 with \
  --comparison-run logs/intensity-comparisons/<with-era5-run> \
  --ablation-max-wind-checkpoint /path/to/max-wind-only.ckpt \
  --ablation-radii-checkpoint /path/to/max-wind-plus-radii.ckpt

uv run geo2wf-infer intensity-comparison-storms \
  --era5 without \
  --comparison-run logs/intensity-comparisons/<without-era5-run>

uv run geo2wf-evaluate three-storm-nowcasts
```

The last command writes long-form predictions, per-storm and combined metrics,
PNG/PDF paper figures, provenance JSON, and the generated section on the final
results page.

## Environment variables

| Variable | Effect |
|---|---|
| `TCD_DATA_ROOT` | source observation archive used by exporters |
| `GEO_SAR_OUTPUT_ROOT` | conventional GEO–SAR export destination override |
| `GEO2WF_RUN_DIR` | inherited DDP run path; normally managed internally |
| `WANDB_DISABLED` | disable W&B construction when true-like |
| `WANDB_MODE` | `offline` retains local W&B artifacts |
| `WANDB_PROJECT`, `WANDB_NAME` | override run tracking names |

Retired commands, full-YAML presets, and launchers are preserved in the
[archive](../archived/index.md) and are not active CLI choices.
