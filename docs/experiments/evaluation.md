# Evaluation metrics

Evaluation uses only finite pixels inside the SAR target mask. Physical metrics
are computed after converting predictions to metres per second (m/s). A model
comparison must use the same sample IDs, masks, split policy, and aggregation.

## Pixel metrics

| Metric | Interpretation |
|---|---|
| MAE | Mean absolute prediction error over valid pixels; lower is better. |
| RMSE | Square root of mean squared error; gives greater weight to large errors; lower is better. |
| Bias | Mean signed error, prediction minus target. Positive values indicate overestimation. |
| PSNR | `20 log10(79.8 m/s / RMSE)` in physical space, using the fixed 0.2–80.0 m/s SAR range; higher is better. |
| SSIM | Scene-level structural similarity after clipping to 0.2–80.0 m/s. Every 7×7 window touching an invalid prediction or target pixel is excluded, then valid scene means are averaged equally; higher is better. |
| High-wind MAE | MAE where target wind meets the configured threshold. The deterministic default is 17 m/s (33.0 kt). |

When ERA5 is available, the deterministic model also evaluates ERA5 on the
same common-valid pixels:

\[
\mathrm{MAE\ skill\ vs\ ERA5}
=1-\frac{\mathrm{MAE}_{model}}{\mathrm{MAE}_{ERA5}}.
\]

Positive skill indicates lower MAE than ERA5. Zero indicates equal MAE.

## Storm-relative structure

The exporter supplies the IBTrACS center and raster bounds. Evaluation converts
pixel centers to local east/north distances, wraps longitude differences at the
dateline, and computes all metrics within the observed SAR footprint.

| Metric | Definition |
|---|---|
| `eye_mae_ms` | Pixel MAE within 25 km of the IBTrACS center. |
| `eye_mean_wind_error_ms` | Absolute error in mean wind within 25 km. |
| `inner_core_mae_ms` | Pixel MAE within 100 km. |
| `radial_profile_mae_ms` | MAE between 10 km annular-mean profiles out to 200 km. |
| `rmw_error_km` | Absolute difference between radii of peak annular-mean wind. |
| `eye_to_eyewall_contrast_error_ms` | Error in peak radial wind minus eye-mean wind. |
| `eye_center_displacement_km` | Distance between inferred target and predicted low-wind eye minima. |

The center is used to derive the storm-distance condition channel and the
evaluation geometry. It is not a predicted quantity.

### Optional comparison with IBTrACS radii

When a batch carries valid IBTrACS structure companions, deterministic and
joint field models additionally compare field-derived RMW/R34/R50/R64 with the
best-track scalars. Field RMW is the peak of the annular-mean profile. R34,
R50, and R64 are equivalent-circle radii computed from the pixel area exceeding
34, 50, and 64 kt within the largest complete circular domain supported by the
prediction mask. Targets outside that supported radius are omitted.

These metrics answer a different question from the SAR-relative rows above:
they compare a gridded instantaneous field statistic with retrospective
best-track structure. Report availability counts and do not combine the two
reference families as though they were interchangeable.

### Eye-center displacement

For target and prediction, the implementation computes a masked 3 × 3 mean and
finds the minimum within 100 km of the IBTrACS center. A smoothed pixel requires
at least eight valid neighbours. The reported distance is between the two
minima, not between either minimum and IBTrACS.

The metric is available only if the target provides adequate evidence:

- at least 80% valid coverage inside 100 km and in the 20–60 km reference ring;
- target ring-mean wind of at least 17 m/s (33.0 kt);
- target minimum within 50 km of the IBTrACS center;
- target ring-to-eye contrast of at least 5 m/s (9.7 kt); and
- valid geometric and smoothing support.

Failed gates produce an unavailable value, not zero. Comparisons must therefore
report the accepted sample count. The metric is diagnostic only: it is excluded
from the training loss and `eye_structure_score`.

## Aggregation and reporting

Pixel metrics accumulate error sums and valid-pixel counts before distributed
reduction. Storm metrics first produce one value per available sample, then
accumulate sample sums and counts. This distinction prevents differently sized
swaths from weighting storm metrics solely by pixel count.

The deterministic model computes reconstruction metrics over all validation
batches unless validation is explicitly limited. Record the sample and storm
counts, coverage limit, and split policy.

A minimum comparison table should contain physical MAE, RMSE, bias, ERA5 skill
where applicable, high-wind MAE, storm-relative metrics with availability
counts. Fixed-sample images
should accompany the table.

## Rapid-intensification subsets

Scalar comparison and three-storm nowcast reports repeat MAE, RMSE, and signed
bias on rapid-intensification phases. At observation time \(t\), the subset is
defined from IBTrACS as

\[
V(t) - V(t - 24\ \mathrm{h}) \ge 30\ \mathrm{kt}.
\]

Both winds are linearly interpolated only inside the configured three-hour
IBTrACS bracket. An observation without a valid 24-hour reference has an
unavailable change and is not included in the RI subset. Reports retain
`ri_24h_change_ms` and `is_rapid_intensification` on every prediction row and
always publish RI sample counts beside the subset metrics.

!!! caution "Check the split policy"
    Modular data configs default to `include_test_in_train: false`. Some
    historical full-YAML presets set it to `true`. Inspect
    `resolved-config.yaml` before describing test results as held out.
