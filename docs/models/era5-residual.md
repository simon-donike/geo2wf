# Stage 1: deterministic baseline

`ERA5ResidualRegressor` is the first stage of the main workflow. It asks whether GEO and ERA5 context can learn a useful physical correction to ERA5 10 m wind:

\[
\hat v_{\mathrm{base}} = v_{\mathrm{ERA5}} + f_\theta(x_{\mathrm{GEO}}, x_{\mathrm{ERA5}}, x_{\mathrm{derived}}, masks)
\]

It is the maintained deterministic wind-field reconstruction model.

## Input and architecture

The dataset supplies 23 condition channels:

- 10 GEO bands;
- seven exported ERA5 fields;
- derived ERA5 wind speed and relative vorticity;
- normalized distance to the IBTrACS center; and
- local-solar-time sine/cosine plus normalized solar zenith.

The model appends:

1. the condition-validity mask;
2. explicit target-normalized ERA5 wind speed; and
3. the ERA5 wind-validity mask.

Its compact U-Net therefore receives 26 channels. It uses two GroupNorm/SiLU residual blocks per resolution, strided-convolution downsampling, bilinear upsampling, encoder skips, and a one-channel residual head.

The final head starts with zero weights and bias. Before the first update, the
predicted residual is zero and the physical prediction equals ERA5.

## Physical-unit objective

Supervised loss is Huber in m/s over the joint SAR/ERA5 valid mask. With `delta = 2 m/s (3.9 kt)`, small errors are quadratic and large errors become linear.

Outside the observed SAR swath, a weak anchor penalizes correction magnitude where ERA5 is valid:

```yaml
model:
  huber_delta_ms: 2.0
  off_swath_anchor_weight: 0.05
```

This discourages unconstrained corrections without pretending ERA5 is observed SAR.

The checked-in default uses the selected peak-aware objective: a smooth high-wind
weight (up to 8x from 25–50 m/s (48.6–97.2 kt)) plus a robust top-0.5% inner-core peak term.
The radial-profile and exceedance losses remain disabled in this default because
the ablation improved the combined peak/structure score without the larger tail
bias seen in the sampling-balanced alternative.

## Prediction bounds

The default clamps physical wind to at least 0 m/s (0 kt). `prediction_max_ms` is unset, so Stage 1 has no upper cap. The configured PSNR data range is 79.8 m/s (155.1 kt), matching the current target export range.

## Diagnostics

The model reports:

- MAE, RMSE, signed bias, Huber loss, and PSNR in physical units;
- ERA5 MAE on identical common-valid pixels;
- MAE skill versus ERA5;
- high-wind MAE and skill for target winds ≥17 m/s (33.0 kt); and
- shared eye, inner-core, radial, RMW, contrast, and eye-displacement metrics.

Validation keeps the physical reconstruction and storm-structure metrics, but
the checked-in grouped preset also logs reconstruction images for the configured
validation batches. Set `model.log_reconstruction_images=false` when a numeric-only
run is preferred.

Three validation choices have different purposes in the grouped default:

- the training objective is the configured Huber/anchor/peak-aware loss;
- `ReduceLROnPlateau` monitors `val/peak_structure_score`; and
- when the trainer does not override it, checkpoint selection uses the model's
  `val/eye_structure_score` default.

Always inspect `resolved-config.yaml` before comparing a historical run: older
full-YAML presets and completed ablations can use different objectives and
monitors.

## Train Stage 1

```bash
uv run geo2wf-train \
  data=geo_sar_common10_era5 \
  model=deterministic_residual
```

## No-ERA5 ablation

The same U-Net can be trained without ERA5:

```bash
uv run geo2wf-train experiment=deterministic_unet_no_era5
```

In this mode the 14-channel condition contains ten GEO bands, storm-center
distance, and three solar-time channels. The U-Net predicts absolute physical
wind speed. The comparison preset retains ERA5 availability filtering to keep
the same cohort, but disables ERA5 inputs, residual addition, off-swath
anchoring, and ERA5 comparison metrics.

The older direct near-89 GHz U-Net has an equivalent preset:

```bash
uv run geo2wf-train experiment=geo_pmw_near89_unet_no_era5
```

After selecting a checkpoint, continue to [Evaluation](../experiments/evaluation.md)
or run storm inference with the [command reference](../reference/commands.md).
