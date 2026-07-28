# Evaluation & metrics

A plausible-looking wind field can still be physically wrong. geo2wf reports three layers of evidence.

## Normalized image metrics

| Metric | Better | Scope |
|---|---|---|
| L1 | lower | valid target pixels in `[0,1]` |
| PSNR | higher | normalized reconstruction fidelity |
| SSIM | higher | structural similarity, masked when needed |
| saturation fraction | lower/contextual | fraction of raw diffusion output at ±1 |

These are useful for optimization diagnostics, not sufficient scientific conclusions.

## Physical and baseline metrics

Predictions are mapped back to m/s using per-channel affine parameters. Metrics include reconstruction MAE/RMSE, ERA5 MAE on common-valid pixels, and:

\[
\mathrm{MAE\ skill\ vs\ ERA5} = 1 - \frac{\mathrm{MAE}_{model}}{\mathrm{MAE}_{ERA5}}
\]

Positive skill means lower MAE than ERA5; zero ties ERA5; negative skill is worse. The residual model also logs signed bias, Huber loss, PSNR in m/s, and high-wind skill above 17 m/s.

## Storm-centric structure

All storm metrics use only observed target pixels and geographic distance from the IBTrACS center.

| Metric | Definition |
|---|---|
| `eye_mae_ms` | MAE inside 25 km |
| `eye_mean_wind_error_ms` | absolute error in eye-mean wind |
| `inner_core_mae_ms` | MAE inside 100 km |
| `radial_profile_mae_ms` | mean error between 10 km radial-bin profiles out to 200 km |
| `rmw_error_km` | radius-of-maximum-wind difference from radial profiles |
| `eye_to_eyewall_contrast_error_ms` | error in peak radial wind minus eye mean |
| `eye_center_displacement_km` | distance between smoothed predicted and target wind minima |
| `high_wind_mae_ms` | MAE where target wind is at least 17 m/s |

### Eye displacement is gated

The diagnostic is omitted unless the search and ring regions have at least 80% target coverage, the 20–60 km ring averages at least 17 m/s, the target minimum lies within 50 km, and target eye-to-ring contrast is at least 5 m/s. This avoids assigning a crisp eye location to weak or poorly observed storms.

## Exact epoch aggregation

Pixel metrics accumulate sums and counts, then reduce across distributed ranks before forming means. Storm metrics accumulate per-sample metric sums and available counts. Unavailable metrics are omitted instead of being treated as zero.

## Recommended comparison table

For a serious model comparison, report at least:

1. observed-pixel MAE/RMSE in m/s;
2. ERA5 MAE and skill on the same common-valid pixels;
3. high-wind MAE;
4. eye, inner-core, radial-profile, and RMW error with sample counts;
5. qualitative reconstructions using fixed latents; and
6. split policy, number of storms, normalization, and sampling settings.

!!! caution "The checked-in split policy"
    Presets combine the test split into training. Do not label resulting test metrics as held-out generalization unless `include_test_in_train` was disabled before training.
