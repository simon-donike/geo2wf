# Normalization & masks

## Statistics source

The exporters accumulate statistics from valid **training** pixels only. Keys are grouped as `{source_type}:{channel}` and include min/max, mean/std, quartiles, median, robust scale, and sample count. A bounded reservoir supplies approximate robust quantiles without retaining every pixel.

## Supported mappings

### Min–max

\[
z = \mathrm{clip}\left(\frac{x - x_{min}}{x_{max}-x_{min}}, 0, 1\right)
\]

This is the default and remains the target mapping for non-negative SAR wind speed in the ERA5 experiment.

### Robust z-score, clipped to `[0, 1]`

The dataset centers by the median and scales by the robust scale derived from IQR, then clips symmetrically at `robust_clip` (4.0 by default) and maps that interval to `[0,1]`. Legacy statistics fall back to mean/std when median/IQR fields are unavailable.

This reduces sensitivity to outlier atmospheric values while retaining the model’s `[0,1]` external contract.

## Why physical values are retained

Image metrics operate on normalized values, but wind errors should be reported in m/s. The loader returns the untouched physical target and an affine inverse mapping:

\[
x_{physical} = z \cdot \text{scale} + \text{offset}
\]

The diffusion module uses that mapping for physical MAE/RMSE and skill against ERA5. The residual model learns and evaluates directly in m/s.

## Three masks with distinct jobs

`condition_mask`
: Identifies pixels supported across the final condition. The diffusion model appends it as an input channel.

`target_mask`
: Identifies observed SAR/PMW pixels. Basic diffusion loss and all target metrics ignore invalid pixels.

`era5_wind_speed_mask`
: Identifies valid ERA5 baseline pixels. It limits sparse completion, residual features, off-swath anchoring, and ERA5 skill comparison.

Invalid values are replaced with neutral zeros **after normalization** and multiplied by masks. The explicit mask channel is what lets the model distinguish “physical value represented by normalized zero” from “missing.”

## Sparse-target completion

The grouped residual-diffusion preset uses:

```yaml
model:
  sparse_target_fill: era5
  unobserved_loss_weight: 0.1
```

Observed SAR pixels keep their target and weight 1. Unobserved pixels with
valid ERA5 receive the target-normalized ERA5 speed and weight 0.1. Remaining
pixels receive neutral 0.5 with weight 0. Metrics still use only observed SAR
pixels.

## Physics-aware flips

Random horizontal/vertical flips are paired across condition, target, and masks. ERA5 u/v components change sign under their corresponding reflection, and relative vorticity behaves as a pseudoscalar. This prevents a common augmentation bug where spatial orientation changes but vector meaning does not.
