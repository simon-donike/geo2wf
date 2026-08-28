# Active model overview

The active inventory is limited to three instantaneous comparison paths, one
joint latent-structure study, and the retained forecast.

## Model and training summary

Parameter counts below are the exact trainable counts for the checked-in model
configuration. They include every encoder, decoder, normalization, and output-head
parameter, but not optimizer state or frozen upstream models. The correction model,
for example, has 932,257 trainable parameters of its own; the frozen U-Net used to
make its input cache is not counted a second time.

| Model / checked-in experiment | Trained on and model inputs | Prediction | Training objective (all contributing terms) | Trainable parameters | Optimizer and learning-rate schedule | Checkpointing and early stopping |
|---|---|---|---|---:|---|---|
| [Field U-Net, with ERA5](era5-residual.md) (`intensity_comparison_unet`) | Storm-disjoint paired GEO–SAR samples on the common IBTrACS/SAR-center-valid, ERA5-available cohort. Inputs are 10 GEO bands, 7 ERA5 fields, derived ERA5 wind speed and vorticity, storm-center distance, 3 solar-time features, and validity/ERA5 helper channels. | ERA5 plus a learned residual: one 2D surface-wind field. | Pixelwise Huber in physical m/s on the joint SAR/ERA5-valid mask (delta 2); high-wind pixels receive a smooth weight up to 8x; plus `0.05` times off-swath ERA5-anchor Huber and `0.05` times robust top-0.5% inner-core peak Huber. Radial-profile and exceedance-area terms have zero weight. | 6,953,473 | AdamW, LR `2e-4`, weight decay `1e-4`. `ReduceLROnPlateau` watches `val/peak_structure_score`; after 100 unimproved validation checks it multiplies LR by `0.75`, with 5-check cooldown and floor `2e-5`. | At most 1,000 epochs. Save best 2 plus last by minimum `val/eye_structure_score`; stop after 50 validation checks without improvement in that same score. |
| [Field U-Net, without ERA5](era5-residual.md#train-without-era5) (`intensity_comparison_unet_no_era5`) | The same ERA5-available cohort for a matched comparison, but only 10 GEO bands, storm-center distance, 3 solar-time features, and a validity mask enter the model. | One absolute 2D surface-wind field. | The same high-wind-weighted pixel Huber and `0.05` peak Huber as above. The off-swath anchor is disabled (`0.0`); radial-profile and exceedance-area terms are also disabled. | 6,950,305 | Same AdamW and plateau schedule as the ERA5 version. | Same 1,000-epoch cap, checkpoint score, and 50-check early-stopping patience. |
| [Single-field intensity correction](intensity-correction.md) (`unet_intensity_correction`) | A storm-disjoint cache of frozen U-Net fields. Each sample contains field wind, validity, center distance, and 19 current-time/location metadata features; the target is tropical IBTrACS `USA_WIND` in m/s. No history or future observation is an input. | A learned signed correction added to the raw valid-pixel field maximum, clamped nonnegative. | Storm-balanced, category-aware weighted Huber on continuous IBTrACS wind (delta 5). The optional five-value structure head is off, so its loss weight is `0.0`. | 932,257 | AdamW, LR `3e-4`, weight decay `1e-4`. `ReduceLROnPlateau` watches `val/loss`; after 10 unimproved checks it halves LR, down to `1e-6`. | At most 200 epochs. Save best 2 plus last by minimum `val/storm_macro_mae_ms`; stop after 50 checks without improvement in that metric. |
| [Joint U-Net + bottleneck MLP, with ERA5](bottleneck-unet-mlp.md) (`bottleneck_unet_mlp`) | The same paired, storm-disjoint GEO–SAR–IBTrACS cohort and 23 condition channels as the ERA5 field model. `USA_WIND` is linearly interpolated to the SAR time only when bracketed by valid fixes no more than 3 h apart. | A 2D SAR wind field and continuous current IBTrACS maximum wind from a shared encoder. | `1.0` times masked image Huber (delta 2 m/s) plus `1.0` times scalar intensity Huber (delta 5 m/s). Categories are derived metrics, not targets. The optional structure term is off. | 7,027,138 | AdamW, LR `2e-4`, weight decay `1e-4`. `ReduceLROnPlateau` watches `val/loss`; after 25 unimproved checks it halves LR (no positive LR floor). | At most 1,000 epochs. Save best 2 plus last by minimum combined `val/loss`; stop after 50 unimproved checks. |
| [Joint U-Net + bottleneck MLP, without ERA5](bottleneck-unet-mlp.md) (`bottleneck_unet_mlp_no_era5`) | The matched ERA5-available cohort, with only the 14 GEO, center-distance, and solar-time condition channels passed to the network. Targets and time-bracketing are otherwise identical to the ERA5 joint model. | An absolute 2D SAR wind field and current IBTrACS maximum wind. | The same unit-weighted image Huber plus intensity Huber objective; structure supervision is off. | 7,024,546 | Same AdamW and plateau schedule as the ERA5 joint model. | Same 1,000-epoch cap, combined-loss checkpointing, and 50-check patience. |
| [Joint latent-structure variant](bottleneck-unet-mlp.md#joint-latent-structure-experiment) (`bottleneck_unet_mlp_max_wind_radii`) | The ERA5-conditioned joint cohort above, augmented with masked IBTrACS eye size, RMW, and equivalent-area R34/R50/R64 targets in km. In the active cohort, unavailable eye labels remain masked. | A 2D wind field, maximum wind, and five nonnegative latent-head structure values. Field-diagnosed radii are evaluation outputs only. | Image Huber + intensity Huber + `0.25` times masked structure Huber (delta 20 km). Each missing structure value is excluded independently. Diagnosed radii from the decoded image do **not** contribute to loss. | 7,027,463 | AdamW, LR `2e-4`, weight decay `1e-4`; halve LR after 25 unimproved `val/loss` checks, with no positive floor. | At most 1,000 epochs. Save best 2 plus last and stop after 50 unimproved checks of combined `val/loss`. |
| [Six-hour intensity forecast](intensity-forecast.md) (pretrain then fine-tune) | Pretraining uses storm-disjoint IBTrACS sequences from 2000–2018 for training and 2019–2022 for validation, with an IBTrACS current anchor. Fine-tuning uses matched storm-disjoint samples whose current anchor is the frozen correction prediction. Inputs are current, −6 h, and −12 h winds plus the two recent 6 h changes. | A signed change from the current anchor to a nonnegative +6 h maximum wind; +12 h is a recursive diagnostic, not a separate training target. | Capped change-balanced weighted Huber on the +6 h continuous wind (delta 5 m/s). No +12 h rollout error or category term contributes to training. | 833 | AdamW with weight decay `1e-4`; LR is `3e-4` for pretraining and `1e-4` for fine-tuning. `ReduceLROnPlateau` watches `val/loss`, halves LR after 10 unimproved checks, and stops reducing at `1e-6`. | Each stage is capped at 1,000 epochs and saves best 2 plus last by minimum `val/storm_macro_mae_ms`. The checked-in forecast presets do **not** enable early stopping. |

Early-stopping patience counts validation checks, which are once per epoch in
these presets. A patience of 50 therefore ends training after 50 consecutive
epochs without a qualifying improvement (`min_delta` defaults to zero). The
learning-rate scheduler is independent: it can reduce the LR several times
before early stopping, and its own patience is measured against the scheduler's
monitor. Checkpoint selection may also monitor a different diagnostic, as it
does for the field U-Net.

All losses use physical units after model output has been converted back from
normalization. “Masked” means missing targets or invalid pixels contribute
neither to the numerator nor denominator; a zero-filled missing value is not
treated as a training target.

## Instantaneous comparison models

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
