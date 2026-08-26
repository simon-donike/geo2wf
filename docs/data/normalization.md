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

Standalone diffusion uses that mapping for physical MAE/RMSE. Stage 1 forms its
prediction and Huber objective directly in m/s. Stage 2 diffuses an encoded
signed residual, but decodes it to m/s for its physical auxiliary losses,
recomposition, metrics, and skill against the frozen baseline.

## Three masks with distinct jobs

`condition_mask`
: Identifies pixels supported across the final condition. The diffusion model appends it as an input channel.

`target_mask`
: Identifies observed SAR/PMW pixels. Basic diffusion loss and all target metrics ignore invalid pixels.

`era5_wind_speed_mask`
: Identifies valid ERA5 baseline pixels. It limits sparse completion, residual features, off-swath anchoring, and ERA5 skill comparison.

Invalid values are replaced with neutral zeros **after normalization** and multiplied by masks. The explicit mask channel is what lets the model distinguish “physical value represented by normalized zero” from “missing.”

## Weak supervision outside the SAR swath

The grouped Stage 2 preset uses:

```yaml
model:
  sparse_target_fill: era5
  unobserved_loss_weight: 0.1
```

For **residual diffusion**, observed baseline-valid pixels contain the encoded
SAR-minus-baseline residual and receive the main loss. Eligible unobserved
baseline pixels use zero encoded residual with weight 0.1. For the ERA5-baseline
ablation that means “retain ERA5”; for deterministic-baseline Stage 2 it means
“retain Stage 1.” It does not insert ERA5 as an absolute target after Stage 1.

Standalone **absolute** conditional diffusion has a separate optional sparse
completion path: unobserved ERA5-valid pixels can receive target-normalized
ERA5 speed with a lower weight, and remaining pixels receive neutral normalized
0.5 with zero weight. The current grouped `conditional_diffusion` config leaves
that option disabled.

In both cases, evaluation metrics remain restricted to observed SAR pixels.

## Physics-aware flips

Random horizontal/vertical flips are paired across condition, target, and masks. ERA5 u/v components change sign under their corresponding reflection, and relative vorticity behaves as a pseudoscalar. This prevents a common augmentation bug where spatial orientation changes but vector meaning does not.
