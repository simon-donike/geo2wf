# Stage 1: deterministic baseline

`ERA5ResidualRegressor` is the first stage of the main workflow. It asks whether GEO and ERA5 context can learn a useful physical correction to ERA5 10 m wind:

\[
\hat v_{\mathrm{base}} = v_{\mathrm{ERA5}} + f_\theta(x_{\mathrm{GEO}}, x_{\mathrm{ERA5}}, x_{\mathrm{derived}}, masks)
\]

Its output becomes the frozen baseline consumed by [Stage 2 residual diffusion](residual-diffusion.md). See [Two-stage baseline + diffusion](two-stage.md) for the complete handoff.

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

The final head starts with zero weights and bias. Before the first update, the predicted residual is zero and the physical prediction is exactly ERA5. Training can only improve or worsen that explicit starting point; the baseline is meaningful from step zero.

## Physical-unit objective

Supervised loss is Huber in m/s over the joint SAR/ERA5 valid mask. With `delta = 2 m/s`, small errors are quadratic and large errors become linear.

Outside the observed SAR swath, a weak anchor penalizes correction magnitude where ERA5 is valid:

```yaml
optimization:
  huber_delta_ms: 2.0
  off_swath_anchor_weight: 0.05
```

This discourages unconstrained corrections without pretending ERA5 is observed SAR.

## Prediction bounds

The default clamps physical wind to at least 0 m/s. `prediction_max_ms` is unset, so Stage 1 has no upper cap. The configured PSNR data range is 79.8 m/s, matching the current target export range.

## Diagnostics

The model reports:

- MAE, RMSE, signed bias, Huber loss, and PSNR in physical units;
- ERA5 MAE on identical common-valid pixels;
- MAE skill versus ERA5;
- high-wind MAE and skill for target winds ≥17 m/s; and
- shared eye, inner-core, radial, RMW, contrast, and eye-displacement metrics.

Validation loader 0 logs physical validation reconstructions. Loader 1 logs a fixed training preview for qualitative comparison only.

## Train Stage 1

```bash
python train.py \
  --config configs/config_geo_sar_10bands_era5_residual.yaml
```

After selecting a checkpoint, pass it to Stage 2 through `GEO2WF_BASELINE_CKPT` or `model.residual.baseline.checkpoint_path`. Continue to [Stage 2 residual diffusion](residual-diffusion.md) or [Evaluation](../experiments/evaluation.md).
