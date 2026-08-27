# Choose an experiment

Active work is intentionally limited to the instantaneous three-model matrix,
the joint U-Net/latent-MLP structure study, and the retained scalar forecast.

## Instantaneous three-model matrix

| Path | Scientific role | ERA5 regimes |
|---|---|---|
| `deterministic_residual` | raw 2D wind-field U-Net and image-derived maximum | with and without |
| `intensity_correction` | scalar correction from a frozen U-Net field | with and without |
| `bottleneck_unet_mlp` | joint 2D field and bottleneck-MLP maximum | with and without |

Use the checked-in experiment pairs to keep model width, data inputs, and
cohort filtering aligned. The complete design is in the [active experiment
matrix](intensity-comparison.md).

## Joint latent-structure study

This matched pair keeps the U-Net decoder so the radii-supervised run can be
evaluated using both direct latent-head predictions and diagnoses from its 2D
wind field. Its completed held-out metrics and figures are collected on the
[current results page](../results.md).

```bash
uv run geo2wf-train experiment=bottleneck_unet_mlp_max_wind
uv run geo2wf-train experiment=bottleneck_unet_mlp_max_wind_radii
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
