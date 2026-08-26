# Intensity reconstruction benchmark

This report compares the raw U-Net field maximum, a separately trained U-Net plus correction network, and the jointly trained U-Net+MLP. Every model is evaluated once with ERA5 conditioning and once without it.

!!! warning "Validation results, not a final test claim"
    The matched benchmark and the Humberto, Kiko, and Otis trajectories are all from the `val` split. None of these storm IDs occurs in training, but validation metrics were used for early stopping and checkpoint selection. These are model-selection diagnostics, not an unbiased held-out test estimate.

## Matched IBTrACS versus SAR supervision experiment

This seed-42 experiment asks whether the scalar intensity heads should learn
from temporally interpolated IBTrACS `USA_WIND` or directly from a SAR-derived
robust peak. It uses the same 159 center-valid validation samples from 33 storms
for every target and ERA5 setting. Twenty-four samples from 14 storms satisfy
the rapid-intensification (RI) definition.

### Main findings

| ERA5 | Training target | Best target-aligned model | Overall MAE, m/s (kt) | RI MAE, m/s (kt) |
|---|---|---|---:|---:|
| With | IBTrACS | U-Net + correction | 6.099 (11.856 kt) | 8.109 (15.763 kt) |
| With | SAR robust peak | U-Net + correction | 4.682 (9.101 kt) | 5.181 (10.071 kt) |
| Without | IBTrACS | Joint U-Net + MLP | 6.583 (12.796 kt) | 5.440 (10.575 kt) |
| Without | SAR robust peak | Joint U-Net + MLP | 5.250 (10.205 kt) | 4.260 (8.281 kt) |

Target-aligned MAE is lowest when each model is scored against the reference it
was trained to reproduce, but the two references are not interchangeable.
Against IBTrACS, SAR supervision did not improve either learned scalar head:
with ERA5, correction MAE changed from 6.099 m/s (11.856 kt) to 7.021 m/s
(13.648 kt) and joint-model MAE from 7.641 m/s (14.853 kt) to 8.724 m/s
(16.958 kt); without ERA5, the corresponding changes were 7.578 m/s
(14.730 kt) to 8.778 m/s (17.063 kt) and 6.583 m/s (12.796 kt) to 8.167 m/s
(15.875 kt). The disagreement is larger during RI because the
SAR robust peak is usually lower than IBTrACS then. These are validation/model-
selection findings from one seed, not locked held-out test estimates.

### SAR and IBTrACS divergence

The matched data contain 568 train, 159 validation, and 139 test samples after
requiring a valid SAR center pixel. Validation contains 24 RI samples from 14
storms. Center-valid rates among otherwise usable IBTrACS/SAR matches are 67.5%
train, 68.5% validation, 65.6% test, 68.7% for RI, and 67.2% for non-RI.

| Subset | SAR diagnostic | Samples | Storms | Bias, m/s (kt) | MAE, m/s (kt); 95% CI | RMSE, m/s (kt) | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| All | Maximum | 866 | 175 | 4.319 (8.395 kt) | 7.150 (13.899 kt); 95% CI 6.441–8.014 m/s (12.520–15.578 kt) | 11.097 (21.571 kt) | 0.762 | 0.806 |
| All | Robust peak | 866 | 175 | -2.009 (-3.905 kt) | 5.522 (10.734 kt); 95% CI 4.994–6.012 m/s (9.708–11.686 kt) | 7.826 (15.213 kt) | 0.889 | 0.906 |
| Validation | Maximum | 159 | 33 | 4.877 (9.480 kt) | 7.512 (14.602 kt); 95% CI 5.673–10.633 m/s (11.027–20.669 kt) | 12.634 (24.559 kt) | 0.722 | 0.804 |
| Validation | Robust peak | 159 | 33 | -2.212 (-4.300 kt) | 5.248 (10.201 kt); 95% CI 4.259–6.124 m/s (8.279–11.904 kt) | 6.893 (13.399 kt) | 0.933 | 0.941 |
| RI, all splits | Maximum | 92 | 53 | -2.108 (-4.098 kt) | 5.697 (11.074 kt); 95% CI 4.792–6.755 m/s (9.315–13.131 kt) | 7.446 (14.474 kt) | 0.764 | 0.786 |
| RI, all splits | Robust peak | 92 | 53 | -9.997 (-19.433 kt) | 10.734 (20.865 kt); 95% CI 9.582–11.917 m/s (18.626–23.165 kt) | 12.255 (23.822 kt) | 0.760 | 0.745 |
| Non-RI, all splits | Maximum | 774 | 171 | 5.083 (9.881 kt) | 7.322 (14.233 kt); 95% CI 6.526–8.274 m/s (12.686–16.083 kt) | 11.453 (22.263 kt) | 0.708 | 0.769 |
| Non-RI, all splits | Robust peak | 774 | 171 | -1.060 (-2.060 kt) | 4.903 (9.531 kt); 95% CI 4.467–5.391 m/s (8.683–10.479 kt) | 7.119 (13.838 kt) | 0.870 | 0.887 |

Bias is `SAR diagnostic − IBTrACS`. The robust peak is more correlated with
IBTrACS and has lower overall disagreement than the single-pixel SAR maximum.
During RI, however, its mean bias is -9.997 m/s (-19.433 kt). This explains why learning the
SAR target can be successful against SAR while degrading IBTrACS-aligned RI
estimates.

### Full matched-storm intensity trajectories

The three panels show the RI storms with the most RI-classified matched SAR
observations. Each panel spans the full retained SAR timeline for that storm
and includes every center-valid SAR acquisition between its first and last
plotted timestamps. The prediction is from the with-ERA5, IBTrACS-trained
U-Net + correction model, selected because it had the lowest overall IBTrACS
MAE before inspecting these trajectories.

![Full matched-storm IBTrACS, SAR maximum, and predicted intensity trajectories with RI windows](../assets/images/intensity-comparison/matched-ri-full-storm-trajectories.png)

Blue is interpolated IBTrACS, orange is the SAR-derived maximum, and green is
the model prediction. A yellow interval covers the preceding 24 hours for each
SAR acquisition classified as RI. Overlapping windows are merged. Lines connect
available SAR acquisition times for readability; they do not imply that SAR or
model predictions were observed between acquisitions.

[Download the plotted observations](../assets/data/intensity-comparison/matched-ri-full-storm-trajectories.csv){ .md-button }
[Download all matrix metrics](../assets/data/intensity-comparison/matched-target-validation.csv){ .md-button }
[Download all per-sample predictions](../assets/data/intensity-comparison/matched-target-predictions.csv){ .md-button }
[Download SAR–IBTrACS divergence statistics](../assets/data/intensity-comparison/matched-sar-ibtracs-divergence.csv){ .md-button }

### Complete dual-reference validation table

Rows labeled `sar_field_only` are reference-aligned diagnostics from the shared
field U-Net rather than separately trained scalar heads.

<!-- matched-validation-results:start -->

Generated on `2026-08-24T12:50:55.706353+00:00` from the completed seed-42 validation matrix. All rows use the same cohort fingerprint `b9fb64003b6c6b483aeea9f9052895f6b3c2f08971392322ca62281739d579f8` (159 samples from 33 storms).

All models use the identical SAR-center-valid cohort. RI denotes an IBTrACS gain of at least 30 kt in the preceding 24 hours.

| ERA5 | Trained target | Model | Evaluated against | Subset | Samples | Storms | MAE, m/s (kt); 95% CI | RMSE, m/s (kt) | Bias, m/s (kt) | Storm-macro MAE, m/s (kt) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| with_era5 | ibtracs | Joint U-Net + MLP | ibtracs | overall | 159 | 33 | 7.641 (14.853 kt); 95% CI 5.810–9.618 m/s (11.294–18.696 kt) | 10.240 (19.905 kt) | -0.930 (-1.808 kt) | 6.379 (12.400 kt) |
| with_era5 | ibtracs | U-Net + correction | ibtracs | overall | 159 | 33 | 6.099 (11.856 kt); 95% CI 4.999–7.042 m/s (9.717–13.689 kt) | 8.167 (15.875 kt) | -2.309 (-4.488 kt) | 5.526 (10.742 kt) |
| with_era5 | sar_field_only | U-Net raw field maximum | ibtracs | overall | 159 | 33 | 6.733 (13.088 kt); 95% CI 5.513–7.784 m/s (10.716–15.131 kt) | 8.947 (17.392 kt) | -3.033 (-5.896 kt) | 5.936 (11.539 kt) |
| with_era5 | ibtracs | Joint U-Net + MLP | ibtracs | rapid_intensification | 24 | 14 | 9.090 (17.670 kt); 95% CI 6.535–12.928 m/s (12.703–25.130 kt) | 11.533 (22.418 kt) | -7.800 (-15.162 kt) | 10.434 (20.282 kt) |
| with_era5 | ibtracs | U-Net + correction | ibtracs | rapid_intensification | 24 | 14 | 8.109 (15.763 kt); 95% CI 5.188–11.662 m/s (10.085–22.669 kt) | 10.749 (20.894 kt) | -6.689 (-13.002 kt) | 8.585 (16.688 kt) |
| with_era5 | sar_field_only | U-Net raw field maximum | ibtracs | rapid_intensification | 24 | 14 | 10.310 (20.041 kt); 95% CI 7.799–13.354 m/s (15.160–25.958 kt) | 12.609 (24.510 kt) | -10.059 (-19.553 kt) | 10.928 (21.242 kt) |
| with_era5 | ibtracs | Joint U-Net + MLP | sar_robust_peak | overall | 159 | 33 | 6.665 (12.956 kt); 95% CI 5.168–8.017 m/s (10.046–15.584 kt) | 8.757 (17.022 kt) | 1.283 (2.494 kt) | 6.314 (12.273 kt) |
| with_era5 | ibtracs | U-Net + correction | sar_robust_peak | overall | 159 | 33 | 5.634 (10.952 kt); 95% CI 4.899–6.554 m/s (9.523–12.740 kt) | 7.059 (13.722 kt) | -0.097 (-0.189 kt) | 5.944 (11.554 kt) |
| with_era5 | sar_field_only | U-Net raw field robust peak | sar_robust_peak | overall | 159 | 33 | 5.192 (10.092 kt); 95% CI 4.480–5.899 m/s (8.708–11.467 kt) | 6.947 (13.504 kt) | -3.368 (-6.547 kt) | 5.630 (10.944 kt) |
| with_era5 | ibtracs | Joint U-Net + MLP | sar_robust_peak | rapid_intensification | 24 | 14 | 6.562 (12.756 kt); 95% CI 4.270–10.293 m/s (8.300–20.008 kt) | 9.342 (18.159 kt) | 0.163 (0.317 kt) | 8.306 (16.146 kt) |
| with_era5 | ibtracs | U-Net + correction | sar_robust_peak | rapid_intensification | 24 | 14 | 6.839 (13.294 kt); 95% CI 4.959–9.009 m/s (9.640–17.512 kt) | 8.381 (16.291 kt) | 1.275 (2.478 kt) | 7.212 (14.019 kt) |
| with_era5 | sar_field_only | U-Net raw field robust peak | sar_robust_peak | rapid_intensification | 24 | 14 | 6.165 (11.984 kt); 95% CI 3.932–9.497 m/s (7.643–18.461 kt) | 9.033 (17.559 kt) | -5.761 (-11.198 kt) | 7.589 (14.752 kt) |
| with_era5 | sar_robust_peak | Joint U-Net + MLP | ibtracs | overall | 159 | 33 | 8.724 (16.958 kt); 95% CI 6.686–10.527 m/s (12.997–20.463 kt) | 11.246 (21.860 kt) | -2.891 (-5.620 kt) | 7.642 (14.855 kt) |
| with_era5 | sar_robust_peak | U-Net + correction | ibtracs | overall | 159 | 33 | 7.021 (13.648 kt); 95% CI 5.513–8.254 m/s (10.716–16.045 kt) | 9.361 (18.196 kt) | -3.532 (-6.866 kt) | 6.007 (11.677 kt) |
| with_era5 | sar_robust_peak | Joint U-Net + MLP | ibtracs | rapid_intensification | 24 | 14 | 13.082 (25.429 kt); 95% CI 9.970–16.419 m/s (19.380–31.916 kt) | 15.295 (29.731 kt) | -12.978 (-25.227 kt) | 12.894 (25.064 kt) |
| with_era5 | sar_robust_peak | U-Net + correction | ibtracs | rapid_intensification | 24 | 14 | 11.707 (22.757 kt); 95% CI 8.797–15.072 m/s (17.100–29.298 kt) | 14.060 (27.330 kt) | -11.677 (-22.698 kt) | 12.020 (23.365 kt) |
| with_era5 | sar_robust_peak | Joint U-Net + MLP | sar_robust_peak | overall | 159 | 33 | 5.416 (10.528 kt); 95% CI 4.329–6.544 m/s (8.415–12.721 kt) | 7.088 (13.778 kt) | -0.678 (-1.318 kt) | 5.397 (10.491 kt) |
| with_era5 | sar_robust_peak | U-Net + correction | sar_robust_peak | overall | 159 | 33 | 4.682 (9.101 kt); 95% CI 4.084–5.294 m/s (7.939–10.291 kt) | 6.217 (12.085 kt) | -1.319 (-2.564 kt) | 4.944 (9.610 kt) |
| with_era5 | sar_robust_peak | Joint U-Net + MLP | sar_robust_peak | rapid_intensification | 24 | 14 | 5.460 (10.613 kt); 95% CI 3.636–8.023 m/s (7.068–15.595 kt) | 7.373 (14.332 kt) | -5.014 (-9.746 kt) | 6.704 (13.032 kt) |
| with_era5 | sar_robust_peak | U-Net + correction | sar_robust_peak | rapid_intensification | 24 | 14 | 5.181 (10.071 kt); 95% CI 3.266–8.103 m/s (6.349–15.751 kt) | 7.886 (15.329 kt) | -3.713 (-7.218 kt) | 6.155 (11.964 kt) |
| without_era5 | ibtracs | Joint U-Net + MLP | ibtracs | overall | 159 | 33 | 6.583 (12.796 kt); 95% CI 5.646–7.443 m/s (10.975–14.468 kt) | 8.602 (16.721 kt) | -0.482 (-0.937 kt) | 6.475 (12.586 kt) |
| without_era5 | ibtracs | U-Net + correction | ibtracs | overall | 159 | 33 | 7.578 (14.730 kt); 95% CI 6.364–8.788 m/s (12.371–17.083 kt) | 9.733 (18.919 kt) | -1.503 (-2.922 kt) | 7.300 (14.190 kt) |
| without_era5 | sar_field_only | U-Net raw field maximum | ibtracs | overall | 159 | 33 | 9.400 (18.272 kt); 95% CI 7.410–10.962 m/s (14.404–21.308 kt) | 12.176 (23.668 kt) | -6.746 (-13.113 kt) | 8.026 (15.601 kt) |
| without_era5 | ibtracs | Joint U-Net + MLP | ibtracs | rapid_intensification | 24 | 14 | 5.440 (10.575 kt); 95% CI 4.213–6.862 m/s (8.189–13.339 kt) | 6.732 (13.086 kt) | -1.502 (-2.920 kt) | 5.425 (10.545 kt) |
| without_era5 | ibtracs | U-Net + correction | ibtracs | rapid_intensification | 24 | 14 | 8.844 (17.191 kt); 95% CI 6.465–11.409 m/s (12.567–22.177 kt) | 10.685 (20.770 kt) | -5.389 (-10.475 kt) | 8.378 (16.286 kt) |
| without_era5 | sar_field_only | U-Net raw field maximum | ibtracs | rapid_intensification | 24 | 14 | 14.897 (28.957 kt); 95% CI 11.589–18.099 m/s (22.527–35.182 kt) | 17.081 (33.203 kt) | -14.633 (-28.444 kt) | 14.261 (27.721 kt) |
| without_era5 | ibtracs | Joint U-Net + MLP | sar_robust_peak | overall | 159 | 33 | 7.371 (14.328 kt); 95% CI 6.128–8.437 m/s (11.912–16.400 kt) | 9.207 (17.897 kt) | 1.731 (3.365 kt) | 6.889 (13.391 kt) |
| without_era5 | ibtracs | U-Net + correction | sar_robust_peak | overall | 159 | 33 | 7.344 (14.276 kt); 95% CI 6.554–8.090 m/s (12.740–15.726 kt) | 9.182 (17.848 kt) | 0.710 (1.380 kt) | 7.399 (14.383 kt) |
| without_era5 | sar_field_only | U-Net raw field robust peak | sar_robust_peak | overall | 159 | 33 | 7.798 (15.158 kt); 95% CI 6.326–9.150 m/s (12.297–17.786 kt) | 9.877 (19.199 kt) | -6.544 (-12.721 kt) | 7.577 (14.729 kt) |
| without_era5 | ibtracs | Joint U-Net + MLP | sar_robust_peak | rapid_intensification | 24 | 14 | 8.769 (17.046 kt); 95% CI 6.594–11.414 m/s (12.818–22.187 kt) | 10.503 (20.416 kt) | 6.462 (12.561 kt) | 9.736 (18.925 kt) |
| without_era5 | ibtracs | U-Net + correction | sar_robust_peak | rapid_intensification | 24 | 14 | 7.097 (13.795 kt); 95% CI 4.984–9.442 m/s (9.688–18.354 kt) | 9.030 (17.553 kt) | 2.574 (5.003 kt) | 7.276 (14.143 kt) |
| without_era5 | sar_field_only | U-Net raw field robust peak | sar_robust_peak | rapid_intensification | 24 | 14 | 10.319 (20.059 kt); 95% CI 8.068–12.854 m/s (15.683–24.986 kt) | 11.882 (23.097 kt) | -10.202 (-19.831 kt) | 10.850 (21.091 kt) |
| without_era5 | sar_robust_peak | Joint U-Net + MLP | ibtracs | overall | 159 | 33 | 8.167 (15.875 kt); 95% CI 6.956–9.169 m/s (13.521–17.823 kt) | 9.865 (19.176 kt) | -3.258 (-6.333 kt) | 7.689 (14.946 kt) |
| without_era5 | sar_robust_peak | U-Net + correction | ibtracs | overall | 159 | 33 | 8.778 (17.063 kt); 95% CI 6.965–10.180 m/s (13.539–19.788 kt) | 11.010 (21.402 kt) | -4.155 (-8.077 kt) | 8.322 (16.177 kt) |
| without_era5 | sar_robust_peak | Joint U-Net + MLP | ibtracs | rapid_intensification | 24 | 14 | 10.913 (21.213 kt); 95% CI 8.801–12.861 m/s (17.108–25.000 kt) | 12.387 (24.078 kt) | -10.903 (-21.194 kt) | 10.579 (20.564 kt) |
| without_era5 | sar_robust_peak | U-Net + correction | ibtracs | rapid_intensification | 24 | 14 | 13.742 (26.712 kt); 95% CI 10.903–16.202 m/s (21.194–31.494 kt) | 15.347 (29.832 kt) | -13.099 (-25.462 kt) | 12.868 (25.013 kt) |
| without_era5 | sar_robust_peak | Joint U-Net + MLP | sar_robust_peak | overall | 159 | 33 | 5.250 (10.205 kt); 95% CI 4.494–6.049 m/s (8.736–11.758 kt) | 6.921 (13.453 kt) | -1.046 (-2.033 kt) | 5.520 (10.730 kt) |
| without_era5 | sar_robust_peak | U-Net + correction | sar_robust_peak | overall | 159 | 33 | 5.952 (11.570 kt); 95% CI 5.097–6.779 m/s (9.908–13.177 kt) | 7.588 (14.750 kt) | -1.942 (-3.775 kt) | 6.289 (12.225 kt) |
| without_era5 | sar_robust_peak | Joint U-Net + MLP | sar_robust_peak | rapid_intensification | 24 | 14 | 4.260 (8.281 kt); 95% CI 2.434–6.915 m/s (4.731–13.442 kt) | 6.633 (12.894 kt) | -2.939 (-5.713 kt) | 5.159 (10.028 kt) |
| without_era5 | sar_robust_peak | U-Net + correction | sar_robust_peak | rapid_intensification | 24 | 14 | 6.636 (12.899 kt); 95% CI 4.839–9.090 m/s (9.406–17.670 kt) | 8.499 (16.521 kt) | -5.135 (-9.982 kt) | 6.986 (13.580 kt) |
<!-- matched-validation-results:end -->

### How the tables and metrics were calculated

- **IBTrACS reference:** `USA_WIND` is converted with
  `1 kt = 0.514444 m/s` and linearly interpolated to the SAR timestamp only when
  its enclosing fixes are no more than three hours apart.
- **SAR maximum:** largest finite wind-speed pixel in the resized,
  center-cropped valid SAR mask.
- **SAR robust peak:** arithmetic mean of the highest 0.5% of finite valid
  pixels in that same crop. At least one pixel is always selected.
- **RI subset:** interpolated IBTrACS increased by at least 30 kt over the
  preceding 24 hours. Missing interpolatable 24-hour history is not labeled RI.
- **MAE / RMSE / bias:** mean absolute error, root mean squared error, and mean
  signed `prediction − reference` error over observations.
- **Storm-macro MAE:** compute MAE within each storm, then average storms with
  equal weight.
- **Category metrics:** exact accuracy, macro F1, and accuracy within one TD,
  TS, or Saffir–Simpson category, using continuous unrounded wind thresholds.
- **Field metrics:** pixel-pooled MAE, RMSE, and bias over the common finite
  valid mask. They apply only to field-producing models.
- **Uncertainty:** 95% percentile intervals use 2,000 seed-42 cluster-bootstrap
  repetitions over storms. Each sampled storm contributes all of its
  observations, and every model uses the same resampled storms.

The overall subset has 159 observations from 33 storms. The RI subset has 24
observations from 14 storms. RI metrics were diagnostic only: checkpoint
selection and early stopping always used the complete validation cohort.

### How the matrix was run

The run used seed 42 and two RTX 3090 GPUs. For each ERA5 regime, one field-only
U-Net was trained and reused across both scalar targets. Four joint U-Net+MLP
models and four correction models were trained. IBTrACS correction used the
predicted field maximum as its anchor; SAR correction used the predicted field
top-0.5%-mean. The full command was:

```bash
uv run python scripts/run_matched_intensity_validation_matrix.py \
  --joint-gpu 0 \
  --pipeline-gpu 1 \
  --seed 42 \
  --wandb-project geo2wf
```

The first evaluation was resumed from its selected checkpoints after fixing
JSON handling for undefined RI history; no partial metrics selected a
checkpoint. The final matrix manifest records every command, checkpoint, and
hash. The consolidated W&B evaluation is
[`use15m4l`](https://wandb.ai/simon-donike/geo2wf/runs/use15m4l).

The trajectory asset is reproducible with:

```bash
.venv/bin/python scripts/build_matched_intensity_storm_plot.py
```

For the experiment design, cohort contract, cache schema, and runner interface,
see [Matched IBTrACS versus SAR intensity comparison](intensity-comparison.md).

## Earlier IBTrACS-only benchmark

On the matched **232-observation, 34-storm** validation cohort, the joint model has the lowest intensity MAE: **6.344 m/s (12.332 kt) with ERA5** and **6.738 m/s (13.098 kt) without ERA5**. With ERA5, the separate correction reaches **6.786 m/s (13.191 kt)**, versus **7.910 m/s (15.376 kt)** for the raw field maximum. The storm-bootstrap intervals for both learned scalar heads' improvement over the raw maximum exclude zero in both regimes.

![Validation intensity MAE comparison](../assets/images/intensity-comparison/validation-intensity-mae.png)

[Download validation results as CSV](../assets/data/intensity-comparison/validation-results.csv){ .md-button }

### Matched validation tables

#### With ERA5

| Model | n | MAE, m/s (kt); 95% CI | Δ MAE vs raw, m/s (kt); 95% CI | RMSE, m/s (kt) | Bias, m/s (kt) | Storm-macro MAE, m/s (kt) | Exact category | Macro F1 | Within one | Field MAE / RMSE / bias, m/s (kt) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net raw field maximum | 232 | 7.910 (15.376 kt); 95% CI 6.495–9.205 m/s (12.625–17.893 kt) | 0.000 (0.000 kt) | 10.312 (20.045 kt) | -5.160 (-10.030 kt) | 6.785 (13.189 kt) | 0.487 | 0.265 | 0.784 | 2.032 (3.950 kt) / 3.097 (6.020 kt) / -0.287 (-0.558 kt) |
| U-Net + correction | 232 | 6.786 (13.191 kt); 95% CI 5.823–7.863 m/s (11.319–15.284 kt) | -1.124 (-2.185 kt); 95% CI -1.902–-0.235 m/s (-3.697–-0.457 kt) | 8.847 (17.197 kt) | -2.368 (-4.603 kt) | 6.422 (12.483 kt) | 0.487 | 0.330 | 0.901 | — / — / — |
| Joint U-Net + MLP | 232 | 6.344 (12.332 kt); 95% CI 5.392–7.254 m/s (10.481–14.101 kt) | -1.566 (-3.044 kt); 95% CI -2.686–-0.228 m/s (-5.221–-0.443 kt) | 8.606 (16.729 kt) | -0.800 (-1.555 kt) | 5.693 (11.066 kt) | 0.522 | 0.362 | 0.875 | 2.205 (4.286 kt) / 3.208 (6.236 kt) / 0.434 (0.844 kt) |

#### Without ERA5

| Model | n | MAE, m/s (kt); 95% CI | Δ MAE vs raw, m/s (kt); 95% CI | RMSE, m/s (kt) | Bias, m/s (kt) | Storm-macro MAE, m/s (kt) | Exact category | Macro F1 | Within one | Field MAE / RMSE / bias, m/s (kt) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net raw field maximum | 232 | 8.804 (17.114 kt); 95% CI 7.212–10.223 m/s (14.019–19.872 kt) | 0.000 (0.000 kt) | 11.521 (22.395 kt) | -6.285 (-12.217 kt) | 7.324 (14.237 kt) | 0.448 | 0.256 | 0.806 | 3.604 (7.006 kt) / 4.960 (9.641 kt) / -0.307 (-0.597 kt) |
| U-Net + correction | 232 | 7.011 (13.628 kt); 95% CI 6.063–7.861 m/s (11.786–15.281 kt) | -1.793 (-3.485 kt); 95% CI -2.700–-0.806 m/s (-5.248–-1.567 kt) | 9.060 (17.611 kt) | -1.549 (-3.011 kt) | 6.503 (12.641 kt) | 0.509 | 0.354 | 0.879 | — / — / — |
| Joint U-Net + MLP | 232 | 6.738 (13.098 kt); 95% CI 5.914–7.511 m/s (11.496–14.600 kt) | -2.067 (-4.018 kt); 95% CI -3.162–-0.735 m/s (-6.146–-1.429 kt) | 8.655 (16.824 kt) | 0.179 (0.348 kt) | 7.021 (13.648 kt) | 0.496 | 0.330 | 0.914 | 3.725 (7.241 kt) / 4.934 (9.591 kt) / 0.280 (0.544 kt) |

The two tables use identical sample IDs. ERA5 therefore changes only the conditioning available to the models, not the evaluation cohort.

### What each metric measures

Let the scalar error be `prediction − IBTrACS target` for one observation.

| Metric | Calculation and interpretation |
|---|---|
| **Intensity MAE** | Mean absolute scalar error. It describes the typical magnitude of an intensity miss in m/s; lower is better. The range is a 95% paired cluster-bootstrap interval over storms. |
| **Δ MAE vs raw** | Candidate MAE minus raw-U-Net MAE on the same storm resample. Negative favors the candidate. An interval below zero means the improvement is consistent across the storm bootstrap. |
| **Intensity RMSE** | Square root of mean squared scalar error. Large misses receive extra weight; lower is better. |
| **Intensity bias** | Mean signed scalar error. Negative is systematic underprediction, positive is overprediction, and zero is ideal. Positive and negative errors can cancel. |
| **Storm-macro MAE** | MAE is computed within each storm and then averaged with equal weight per storm. It prevents storms with many images from dominating. |
| **Exact category accuracy** | Fraction assigned exactly the correct TD, TS, or Saffir–Simpson category; higher is better. |
| **Category macro F1** | Per-category harmonic mean of precision and recall, averaged equally across represented categories; higher is better. |
| **Within one** | Fraction no more than one category away from the target; higher is better. |
| **Field MAE / RMSE / bias** | Pixel-pooled U-Net-minus-SAR errors over the common finite valid mask. These diagnose the reconstructed wind field, not scalar intensity. They do not apply to the separate correction head, which emits only a scalar. |

IBTrACS `USA_WIND` is converted from knots with `1 kt = 0.514444 m/s` and linearly interpolated to the image timestamp only when the enclosing fixes are at most three hours apart. Categories use the unrounded thresholds: TD `<34 kt`, TS `34–<64`, C1 `64–<83`, C2 `83–<96`, C3 `96–<113`, C4 `113–<137`, and C5 `≥137 kt`.

The 95% intervals use 2,000 paired cluster-bootstrap repetitions over storm IDs (seed 42). Every resample evaluates all models on the same storms and retains every observation from each sampled storm.

### Training, W&B, and early stopping

All six runs logged metrics to Weights & Biases. Training allowed up to 1,000 epochs but stopped after **50 validation epochs without improvement**. The vertical dashed lines below mark the checkpoint selected by each stage-specific validation monitor.

![Validation monitor histories](../assets/images/intensity-comparison/training-validation-curves.png)

| Conditioning | Raw U-Net | Separate correction | Joint U-Net + MLP |
|---|---|---|---|
| With ERA5 | epoch 61 · `4rqrc3oh` | epoch 27 · `ymivzoau` | epoch 139 · `rj7951rk` |
| Without ERA5 | epoch 77 · `ldd7fp28` | epoch 50 · `frrrn6dl` | epoch 70 · `oyiqs6go` |

#### Validation reconstruction media at the selected checkpoints

The correction images are the W&B three-storm diagnostic nearest each selected checkpoint. They show the automatically selected validation storms (not the dedicated three-storm dense analysis below). The joint images show validation GEO input, predicted and SAR target fields, valid footprints, ERA5 where applicable, and scalar-intensity error.

The raw U-Net run had media logging disabled, so its panels below were regenerated from the exact selected epoch-77 checkpoint (`ldd7fp28`) on the same storm-stratified validation loader. Each row shows GEO input, the reconstructed wind field, the SAR target, and both valid footprints. No ERA5 field is supplied to this model.

=== "Raw U-Net · without ERA5"

    ![Raw U-Net reconstruction samples without ERA5, set 1](../assets/images/intensity-comparison/unet-without-era5-epoch077-batch-03.jpg)

    ![Raw U-Net reconstruction samples without ERA5, set 2](../assets/images/intensity-comparison/unet-without-era5-epoch077-batch-23.jpg)

    ![Raw U-Net reconstruction samples without ERA5, set 3](../assets/images/intensity-comparison/unet-without-era5-epoch077-batch-31.jpg)

=== "Correction · with ERA5"

    ![W&B correction validation media with ERA5](../assets/images/intensity-comparison/wandb-correction-with-era5-best.png)

=== "Correction · without ERA5"

    ![W&B correction validation media without ERA5](../assets/images/intensity-comparison/wandb-correction-without-era5-best.png)

=== "Joint · with ERA5"

    ![W&B joint validation reconstruction with ERA5](../assets/images/intensity-comparison/wandb-joint-with-era5-best.jpg)

=== "Joint · without ERA5"

    ![W&B joint validation reconstruction without ERA5](../assets/images/intensity-comparison/wandb-joint-without-era5-best.jpg)

## Humberto, Kiko, and Otis: dense full-storm inference

The inference manifest contributes every listed 10-minute GEO image: **1,006 Humberto observations (`AL082025`)**, **1,578 Kiko observations (`EP112025`)**, and **684 Otis observations (`EP182023`)**, for **3,268** timestamps. All 3,268 have a valid three-hour-or-narrower IBTrACS bracket. Both conditioning regimes use exactly the same observation IDs, centers, timestamps, and IBTrACS reference targets.

Inference was attempted for all **3,268** scans. **3,266** have a non-empty valid footprint after the model's center crop and are scored; **2 scans** are retained in the download with `inference_valid = false` and excluded identically from both regimes and every model metric.

Across the dense common cohort, **U-Net + correction** has the lowest aggregate MAE with ERA5 (**7.476 m/s (14.532 kt)**), while **Joint U-Net + MLP** is lowest without ERA5 (**7.243 m/s (14.079 kt)**). Per-storm behavior differs; the trajectory and storm-level table report that variation.

The plotted curves are hourly means to keep the dense 10-minute series readable. The table scores every valid individual observation, while the download also retains any explicitly flagged unusable scan.

![Predicted and IBTrACS-reference full-storm intensity trajectories](../assets/images/intensity-comparison/three-storm-intensity-trajectories.png)

![Per-storm dense inference MAE](../assets/images/intensity-comparison/three-storm-mae.png)

Every wind-speed value below is shown in m/s with knots in parentheses; the sample count is unitless.

| Conditioning | Model | valid n / attempted | MAE, m/s (kt) | RMSE, m/s (kt) | Bias, m/s (kt) | Humberto MAE, m/s (kt) | Kiko MAE, m/s (kt) | Otis MAE, m/s (kt) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| With ERA5 | U-Net raw field maximum | 3266 / 3268 | 8.942 (17.382 kt) | 11.882 (23.097 kt) | -6.458 (-12.553 kt) | 7.838 (15.236 kt) | 10.531 (20.471 kt) | 6.908 (13.428 kt) |
| With ERA5 | U-Net + correction | 3266 / 3268 | 7.476 (14.532 kt) | 10.302 (20.026 kt) | -4.430 (-8.611 kt) | 5.614 (10.913 kt) | 9.011 (17.516 kt) | 6.677 (12.979 kt) |
| With ERA5 | Joint U-Net + MLP | 3266 / 3268 | 8.029 (15.607 kt) | 11.146 (21.666 kt) | -3.437 (-6.681 kt) | 6.766 (13.152 kt) | 9.453 (18.375 kt) | 6.604 (12.837 kt) |
| Without ERA5 | U-Net raw field maximum | 3266 / 3268 | 9.113 (17.714 kt) | 11.404 (22.168 kt) | -4.968 (-9.657 kt) | 11.271 (21.909 kt) | 8.586 (16.690 kt) | 7.153 (13.904 kt) |
| Without ERA5 | U-Net + correction | 3266 / 3268 | 7.345 (14.278 kt) | 9.452 (18.373 kt) | -1.483 (-2.883 kt) | 8.338 (16.208 kt) | 7.160 (13.918 kt) | 6.314 (12.273 kt) |
| Without ERA5 | Joint U-Net + MLP | 3266 / 3268 | 7.243 (14.079 kt) | 9.159 (17.804 kt) | 0.071 (0.138 kt) | 7.631 (14.833 kt) | 6.962 (13.533 kt) | 7.318 (14.225 kt) |

[Download all six dense prediction series](../assets/data/intensity-comparison/three-storm-inference.csv){ .md-button } [Download dense metrics](../assets/data/intensity-comparison/three-storm-metrics.csv){ .md-button } [Download JSON summary](../assets/data/intensity-comparison/three-storm-summary.json){ .md-button }

### Split audit

| Storm | Source split | Paired train samples | Paired validation samples | Paired test samples | Dense valid / attempted |
|---|---|---:|---:|---:|---:|
| Humberto (`AL082025`) | `val` | 0 | 18 | 0 | 1,006 / 1,006 |
| Kiko (`EP112025`) | `val` | 0 | 23 | 0 | 1,576 / 1,578 |
| Otis (`EP182023`) | `val` | 0 | 2 | 0 | 684 / 684 |

The split audit confirms that **none of the three storms is in training**. They are validation storms, including Otis; none is in the test split. Because validation guided early stopping, the dense plots are diagnostic case studies rather than independent test cases.

## Reproduce the dense inference

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_intensity_comparison_storm_inference.py --era5 with --device cuda
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_intensity_comparison_storm_inference.py --era5 without --device cuda
.venv/bin/python scripts/build_intensity_comparison_web_report.py
```

Each inference JSON records SHA-256 hashes for the raw U-Net, correction, and joint checkpoints. The correction run additionally verifies that its frozen-field cache was generated by the exact selected raw U-Net checkpoint.

## Limitations

- Validation-guided selection makes all reported results model-selection diagnostics.
- Dense 10-minute observations are strongly temporally correlated; 3,268 rows are not 3,268 independent storms or trials.
- IBTrACS is a best-track estimate interpolated in time, not a direct measurement at each satellite scan.
- The raw scalar is the largest valid pixel in a reconstructed field, while the learned heads estimate IBTrACS maximum wind directly; these are related but not identical physical quantities.
- Final performance estimation requires a locked, storm-disjoint test set after architecture and checkpoint selection.
