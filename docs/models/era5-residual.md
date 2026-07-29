# ERA5 residual baseline

`ERA5ResidualRegressor` asks a sharper control question: can GEO and ERA5 context learn a useful correction to the ERA5 10 m wind field?

\[
\hat y_{wind} = y_{ERA5} + f_\theta(x_{GEO}, x_{ERA5}, masks)
\]

## Input and architecture

The dataset supplies 19 condition channels: 10 GEO, seven source ERA5 fields, derived wind speed, and derived relative vorticity. The model appends:

1. the condition-validity mask;
2. explicit target-normalized ERA5 wind speed; and
3. the ERA5 wind-validity mask.

Its U-Net therefore receives 22 channels. The compact backbone uses two GroupNorm/SiLU residual blocks per resolution, strided convolution downsampling, bilinear upsampling, encoder skips, and a one-channel residual head.

The final head starts with all weights and bias at zero. Before the first update, the predicted residual is zero and the physical prediction is exactly ERA5. This makes the baseline meaningful from step zero and focuses learning on corrections.

## Physical-unit objective

The supervised loss is Huber in m/s over the joint target/ERA5 valid mask. With the configured `delta = 2 m/s`, small errors are quadratic and large errors become linear, reducing sensitivity to extreme pixels without discarding them.

Outside the observed SAR swath, a weak anchor loss penalizes residual magnitude where ERA5 is valid:

```yaml
optimization:
  huber_delta_ms: 2.0
  off_swath_anchor_weight: 0.05
```

This discourages unconstrained corrections in regions with no SAR supervision. It does not label ERA5 as observed SAR.

## Prediction bounds

The default clamps physical wind to at least 0 m/s. `prediction_max_ms` is unset, so no upper cap is applied. The configured PSNR data range is 79.8 m/s, corresponding to the current exported target range.

## Reported diagnostics

The residual model accumulates exact epoch-level pixel statistics and reports:

- MAE, RMSE, signed bias, Huber loss, and PSNR in physical units;
- ERA5 MAE on the identical common-valid pixels;
- MAE skill versus ERA5;
- high-wind MAE and high-wind skill for target winds ≥17 m/s; and
- the shared eye, inner-core, radial, RMW, contrast, and eye-displacement metrics.

Validation loader 0 logs physical validation reconstructions for the configured number of batches; loader 1 logs one fixed training preview. Both use the shared georeferenced W&B helper.

Use `configs/config_geo_sar_10bands_era5_residual.yaml` and see [Evaluation](../experiments/evaluation.md) for interpretation.
