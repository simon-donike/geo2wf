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

The ERA5-residual model forms its prediction and Huber objective directly in
m/s. Other reconstruction targets use the same inverse mapping for physical
MAE and RMSE.

## Three masks with distinct jobs

`condition_mask`
: Identifies pixels supported across the final condition. Models may append it as an input channel.

`target_mask`
: Identifies observed SAR/PMW pixels. Target losses and metrics ignore invalid pixels.

`era5_wind_speed_mask`
: Identifies valid ERA5 baseline pixels. It limits sparse completion, residual features, off-swath anchoring, and ERA5 skill comparison.

Invalid values are replaced with neutral zeros **after normalization** and multiplied by masks. The explicit mask channel is what lets the model distinguish “physical value represented by normalized zero” from “missing.”

## Weak supervision outside the SAR swath

The ERA5-residual objective can weakly penalize corrections outside the SAR
swath where ERA5 is valid. This regularizes the unobserved region without
treating ERA5 as a SAR observation. Evaluation metrics remain restricted to
observed SAR pixels.

## Physics-aware flips

Random horizontal/vertical flips are paired across condition, target, and masks. ERA5 u/v components change sign under their corresponding reflection, and relative vorticity behaves as a pseudoscalar. This prevents a common augmentation bug where spatial orientation changes but vector meaning does not.
