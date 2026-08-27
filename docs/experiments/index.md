# Choose an experiment

Active work is intentionally limited to the instantaneous three-model matrix,
the encoder/latent-MLP structure study, and the retained scalar forecast.

## Instantaneous three-model matrix

| Path | Scientific role | ERA5 regimes |
|---|---|---|
| `deterministic_residual` | raw 2D wind-field U-Net and image-derived maximum | with and without |
| `intensity_correction` | scalar correction from a frozen U-Net field | with and without |
| `bottleneck_unet_mlp` | joint 2D field and bottleneck-MLP maximum | with and without |

Use the checked-in experiment pairs to keep model width, data inputs, and
cohort filtering aligned. The complete design is in the [active experiment
matrix](intensity-comparison.md).

## Encoder/latent-MLP study

`bottleneck_encoder_mlp` keeps the U-Net encoder and pooled latent MLP but
removes the image decoder. The next study compares maximum wind alone with
radii predicted by the MLP and radii diagnosed from a 2D U-Net field.

```bash
uv run geo2wf-train experiment=unet_encoder_mlp_ibtracs
uv run geo2wf-train experiment=unet_encoder_mlp_ibtracs_no_era5
```

## Forecast retained

The six-hour scalar forecast remains active and separate from the
instantaneous ERA5 matrix:

```bash
uv run geo2wf-train experiment=intensity_forecast_pretrain
uv run geo2wf-train experiment=intensity_forecast_finetune
```

## Archived work

Diffusion, direct-PMW proxy training, historical full-YAML presets, ablation
suites, and previous results are preserved under the repository's `archived/`
tree and the documentation [archive](../archived/index.md). They are not active
configuration choices.

Continue to [configuration](configuration.md), [training](training.md), and
[evaluation](evaluation.md).
