# Active model overview

The active inventory is limited to three instantaneous comparison paths, one
joint latent-structure study, and the retained forecast.

## Instantaneous comparison models

| Model | Output | Role in the ERA5 matrix |
|---|---|---|
| [ERA5-residual/absolute U-Net](era5-residual.md) | 2D surface-wind field | image-derived maximum, trained with and without ERA5 |
| [Single-field intensity correction](intensity-correction.md) | corrected current maximum wind | consumes a frozen field from the matching ERA5 regime |
| [Joint U-Net + bottleneck MLP](bottleneck-unet-mlp.md) | 2D wind field and current maximum wind | shared encoder, image decoder, and scalar head |

The with-ERA5 U-Net predicts a physical correction around an ERA5 anchor. The
no-ERA5 version uses the same architecture to predict absolute wind and keeps
ERA5 only as a cohort-availability filter.

## Joint latent-structure study

The structure variant of the [bottleneck model](bottleneck-unet-mlp.md#joint-latent-structure-experiment)
retains the decoder. Its evaluation separates maximum wind, radii predicted
directly by the latent MLP, and radii diagnosed from the same model's 2D U-Net
image output.

## Forecast

The [six-hour scalar intensity forecast](intensity-forecast.md) remains active.
It consumes current intensity and recent best-track history; it is downstream
of, and separate from, the instantaneous ERA5 comparison.

## Archived models

Diffusion and direct PMW-proxy model code is preserved under `archived/`. Their
documentation is available in [Archived work](../archived/index.md), and they
are no longer selectable from active `configs/model/` choices.

All retained paths share storm-disjoint cohort rules, physical-unit reporting,
and explicit source labels for MLP-derived versus image-derived structure.
