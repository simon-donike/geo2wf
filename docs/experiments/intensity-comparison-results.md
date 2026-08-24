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

| ERA5 | Training target | Best target-aligned model | Overall MAE (m/s) | RI MAE (m/s) |
|---|---|---|---:|---:|
| With | IBTrACS | U-Net + correction | 6.099 | 8.109 |
| With | SAR robust peak | U-Net + correction | 4.682 | 5.181 |
| Without | IBTrACS | Joint U-Net + MLP | 6.583 | 5.440 |
| Without | SAR robust peak | Joint U-Net + MLP | 5.250 | 4.260 |

Target-aligned MAE is lowest when each model is scored against the reference it
was trained to reproduce, but the two references are not interchangeable.
Against IBTrACS, SAR supervision did not improve either learned scalar head:
with ERA5, correction MAE changed from 6.099 to 7.021 m/s and joint-model MAE
from 7.641 to 8.724 m/s; without ERA5, the corresponding changes were 7.578 to
8.778 and 6.583 to 8.167 m/s. The disagreement is larger during RI because the
SAR robust peak is usually lower than IBTrACS then. These are validation/model-
selection findings from one seed, not locked held-out test estimates.

### SAR and IBTrACS divergence

The matched data contain 568 train, 159 validation, and 139 test samples after
requiring a valid SAR center pixel. Validation contains 24 RI samples from 14
storms. Center-valid rates among otherwise usable IBTrACS/SAR matches are 67.5%
train, 68.5% validation, 65.6% test, 68.7% for RI, and 67.2% for non-RI.

| Subset | SAR diagnostic | Samples | Storms | Bias (m/s) | MAE (m/s; 95% CI) | RMSE (m/s) | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| All | Maximum | 866 | 175 | 4.319 | 7.150 (6.441–8.014) | 11.097 | 0.762 | 0.806 |
| All | Robust peak | 866 | 175 | -2.009 | 5.522 (4.994–6.012) | 7.826 | 0.889 | 0.906 |
| Validation | Maximum | 159 | 33 | 4.877 | 7.512 (5.673–10.633) | 12.634 | 0.722 | 0.804 |
| Validation | Robust peak | 159 | 33 | -2.212 | 5.248 (4.259–6.124) | 6.893 | 0.933 | 0.941 |
| RI, all splits | Maximum | 92 | 53 | -2.108 | 5.697 (4.792–6.755) | 7.446 | 0.764 | 0.786 |
| RI, all splits | Robust peak | 92 | 53 | -9.997 | 10.734 (9.582–11.917) | 12.255 | 0.760 | 0.745 |
| Non-RI, all splits | Maximum | 774 | 171 | 5.083 | 7.322 (6.526–8.274) | 11.453 | 0.708 | 0.769 |
| Non-RI, all splits | Robust peak | 774 | 171 | -1.060 | 4.903 (4.467–5.391) | 7.119 | 0.870 | 0.887 |

Bias is `SAR diagnostic − IBTrACS`. The robust peak is more correlated with
IBTrACS and has lower overall disagreement than the single-pixel SAR maximum.
During RI, however, its mean bias is -9.997 m/s. This explains why learning the
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

Blue is interpolated IBTrACS, orange is the observed SAR maximum, and green is
the model prediction. A yellow interval covers the preceding 24 hours for each
SAR observation classified as RI. Overlapping windows are merged. Lines connect
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

| ERA5 | Trained target | Model | Evaluated against | Subset | Samples | Storms | MAE (m/s; 95% CI) | RMSE | Bias | Storm-macro MAE |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| with_era5 | ibtracs | Joint U-Net + MLP | ibtracs | overall | 159 | 33 | 7.641 (5.810–9.618) | 10.240 | -0.930 | 6.379 |
| with_era5 | ibtracs | U-Net + correction | ibtracs | overall | 159 | 33 | 6.099 (4.999–7.042) | 8.167 | -2.309 | 5.526 |
| with_era5 | sar_field_only | U-Net raw field maximum | ibtracs | overall | 159 | 33 | 6.733 (5.513–7.784) | 8.947 | -3.033 | 5.936 |
| with_era5 | ibtracs | Joint U-Net + MLP | ibtracs | rapid_intensification | 24 | 14 | 9.090 (6.535–12.928) | 11.533 | -7.800 | 10.434 |
| with_era5 | ibtracs | U-Net + correction | ibtracs | rapid_intensification | 24 | 14 | 8.109 (5.188–11.662) | 10.749 | -6.689 | 8.585 |
| with_era5 | sar_field_only | U-Net raw field maximum | ibtracs | rapid_intensification | 24 | 14 | 10.310 (7.799–13.354) | 12.609 | -10.059 | 10.928 |
| with_era5 | ibtracs | Joint U-Net + MLP | sar_robust_peak | overall | 159 | 33 | 6.665 (5.168–8.017) | 8.757 | 1.283 | 6.314 |
| with_era5 | ibtracs | U-Net + correction | sar_robust_peak | overall | 159 | 33 | 5.634 (4.899–6.554) | 7.059 | -0.097 | 5.944 |
| with_era5 | sar_field_only | U-Net raw field robust peak | sar_robust_peak | overall | 159 | 33 | 5.192 (4.480–5.899) | 6.947 | -3.368 | 5.630 |
| with_era5 | ibtracs | Joint U-Net + MLP | sar_robust_peak | rapid_intensification | 24 | 14 | 6.562 (4.270–10.293) | 9.342 | 0.163 | 8.306 |
| with_era5 | ibtracs | U-Net + correction | sar_robust_peak | rapid_intensification | 24 | 14 | 6.839 (4.959–9.009) | 8.381 | 1.275 | 7.212 |
| with_era5 | sar_field_only | U-Net raw field robust peak | sar_robust_peak | rapid_intensification | 24 | 14 | 6.165 (3.932–9.497) | 9.033 | -5.761 | 7.589 |
| with_era5 | sar_robust_peak | Joint U-Net + MLP | ibtracs | overall | 159 | 33 | 8.724 (6.686–10.527) | 11.246 | -2.891 | 7.642 |
| with_era5 | sar_robust_peak | U-Net + correction | ibtracs | overall | 159 | 33 | 7.021 (5.513–8.254) | 9.361 | -3.532 | 6.007 |
| with_era5 | sar_robust_peak | Joint U-Net + MLP | ibtracs | rapid_intensification | 24 | 14 | 13.082 (9.970–16.419) | 15.295 | -12.978 | 12.894 |
| with_era5 | sar_robust_peak | U-Net + correction | ibtracs | rapid_intensification | 24 | 14 | 11.707 (8.797–15.072) | 14.060 | -11.677 | 12.020 |
| with_era5 | sar_robust_peak | Joint U-Net + MLP | sar_robust_peak | overall | 159 | 33 | 5.416 (4.329–6.544) | 7.088 | -0.678 | 5.397 |
| with_era5 | sar_robust_peak | U-Net + correction | sar_robust_peak | overall | 159 | 33 | 4.682 (4.084–5.294) | 6.217 | -1.319 | 4.944 |
| with_era5 | sar_robust_peak | Joint U-Net + MLP | sar_robust_peak | rapid_intensification | 24 | 14 | 5.460 (3.636–8.023) | 7.373 | -5.014 | 6.704 |
| with_era5 | sar_robust_peak | U-Net + correction | sar_robust_peak | rapid_intensification | 24 | 14 | 5.181 (3.266–8.103) | 7.886 | -3.713 | 6.155 |
| without_era5 | ibtracs | Joint U-Net + MLP | ibtracs | overall | 159 | 33 | 6.583 (5.646–7.443) | 8.602 | -0.482 | 6.475 |
| without_era5 | ibtracs | U-Net + correction | ibtracs | overall | 159 | 33 | 7.578 (6.364–8.788) | 9.733 | -1.503 | 7.300 |
| without_era5 | sar_field_only | U-Net raw field maximum | ibtracs | overall | 159 | 33 | 9.400 (7.410–10.962) | 12.176 | -6.746 | 8.026 |
| without_era5 | ibtracs | Joint U-Net + MLP | ibtracs | rapid_intensification | 24 | 14 | 5.440 (4.213–6.862) | 6.732 | -1.502 | 5.425 |
| without_era5 | ibtracs | U-Net + correction | ibtracs | rapid_intensification | 24 | 14 | 8.844 (6.465–11.409) | 10.685 | -5.389 | 8.378 |
| without_era5 | sar_field_only | U-Net raw field maximum | ibtracs | rapid_intensification | 24 | 14 | 14.897 (11.589–18.099) | 17.081 | -14.633 | 14.261 |
| without_era5 | ibtracs | Joint U-Net + MLP | sar_robust_peak | overall | 159 | 33 | 7.371 (6.128–8.437) | 9.207 | 1.731 | 6.889 |
| without_era5 | ibtracs | U-Net + correction | sar_robust_peak | overall | 159 | 33 | 7.344 (6.554–8.090) | 9.182 | 0.710 | 7.399 |
| without_era5 | sar_field_only | U-Net raw field robust peak | sar_robust_peak | overall | 159 | 33 | 7.798 (6.326–9.150) | 9.877 | -6.544 | 7.577 |
| without_era5 | ibtracs | Joint U-Net + MLP | sar_robust_peak | rapid_intensification | 24 | 14 | 8.769 (6.594–11.414) | 10.503 | 6.462 | 9.736 |
| without_era5 | ibtracs | U-Net + correction | sar_robust_peak | rapid_intensification | 24 | 14 | 7.097 (4.984–9.442) | 9.030 | 2.574 | 7.276 |
| without_era5 | sar_field_only | U-Net raw field robust peak | sar_robust_peak | rapid_intensification | 24 | 14 | 10.319 (8.068–12.854) | 11.882 | -10.202 | 10.850 |
| without_era5 | sar_robust_peak | Joint U-Net + MLP | ibtracs | overall | 159 | 33 | 8.167 (6.956–9.169) | 9.865 | -3.258 | 7.689 |
| without_era5 | sar_robust_peak | U-Net + correction | ibtracs | overall | 159 | 33 | 8.778 (6.965–10.180) | 11.010 | -4.155 | 8.322 |
| without_era5 | sar_robust_peak | Joint U-Net + MLP | ibtracs | rapid_intensification | 24 | 14 | 10.913 (8.801–12.861) | 12.387 | -10.903 | 10.579 |
| without_era5 | sar_robust_peak | U-Net + correction | ibtracs | rapid_intensification | 24 | 14 | 13.742 (10.903–16.202) | 15.347 | -13.099 | 12.868 |
| without_era5 | sar_robust_peak | Joint U-Net + MLP | sar_robust_peak | overall | 159 | 33 | 5.250 (4.494–6.049) | 6.921 | -1.046 | 5.520 |
| without_era5 | sar_robust_peak | U-Net + correction | sar_robust_peak | overall | 159 | 33 | 5.952 (5.097–6.779) | 7.588 | -1.942 | 6.289 |
| without_era5 | sar_robust_peak | Joint U-Net + MLP | sar_robust_peak | rapid_intensification | 24 | 14 | 4.260 (2.434–6.915) | 6.633 | -2.939 | 5.159 |
| without_era5 | sar_robust_peak | U-Net + correction | sar_robust_peak | rapid_intensification | 24 | 14 | 6.636 (4.839–9.090) | 8.499 | -5.135 | 6.986 |
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

On the matched **232-observation, 34-storm** validation cohort, the joint model has the lowest intensity MAE: **6.344 m/s with ERA5** and **6.738 m/s without ERA5**. With ERA5, the separate correction reaches **6.786 m/s**, versus **7.910 m/s** for the raw field maximum. The storm-bootstrap intervals for both learned scalar heads' improvement over the raw maximum exclude zero in both regimes.

![Validation intensity MAE comparison](../assets/images/intensity-comparison/validation-intensity-mae.png)

[Download validation results as CSV](../assets/data/intensity-comparison/validation-results.csv){ .md-button }

### Matched validation tables

#### With ERA5

| Model | n | MAE (95% CI), m/s | Δ MAE vs raw, m/s | RMSE, m/s | Bias, m/s | Storm-macro MAE, m/s | Exact category | Macro F1 | Within one | Field MAE / RMSE / bias, m/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net raw field maximum | 232 | 7.910 (6.495–9.205) | 0.000 | 10.312 | -5.160 | 6.785 | 0.487 | 0.265 | 0.784 | 2.032 / 3.097 / -0.287 |
| U-Net + correction | 232 | 6.786 (5.823–7.863) | -1.124 (-1.902–-0.235) | 8.847 | -2.368 | 6.422 | 0.487 | 0.330 | 0.901 | — / — / — |
| Joint U-Net + MLP | 232 | 6.344 (5.392–7.254) | -1.566 (-2.686–-0.228) | 8.606 | -0.800 | 5.693 | 0.522 | 0.362 | 0.875 | 2.205 / 3.208 / 0.434 |

#### Without ERA5

| Model | n | MAE (95% CI), m/s | Δ MAE vs raw, m/s | RMSE, m/s | Bias, m/s | Storm-macro MAE, m/s | Exact category | Macro F1 | Within one | Field MAE / RMSE / bias, m/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net raw field maximum | 232 | 8.804 (7.212–10.223) | 0.000 | 11.521 | -6.285 | 7.324 | 0.448 | 0.256 | 0.806 | 3.604 / 4.960 / -0.307 |
| U-Net + correction | 232 | 7.011 (6.063–7.861) | -1.793 (-2.700–-0.806) | 9.060 | -1.549 | 6.503 | 0.509 | 0.354 | 0.879 | — / — / — |
| Joint U-Net + MLP | 232 | 6.738 (5.914–7.511) | -2.067 (-3.162–-0.735) | 8.655 | 0.179 | 7.021 | 0.496 | 0.330 | 0.914 | 3.725 / 4.934 / 0.280 |

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

The inference manifest contributes every listed 10-minute GEO image: **1,006 Humberto observations (`AL082025`)**, **1,578 Kiko observations (`EP112025`)**, and **684 Otis observations (`EP182023`)**, for **3,268** timestamps. All 3,268 have a valid three-hour-or-narrower IBTrACS bracket. Both conditioning regimes use exactly the same observation IDs, centers, timestamps, and ground truth.

Inference was attempted for all **3,268** scans. **3,266** have a non-empty valid footprint after the model's center crop and are scored; **2 scans** are retained in the download with `inference_valid = false` and excluded identically from both regimes and every model metric.

Across the dense common cohort, **U-Net + correction** has the lowest aggregate MAE with ERA5 (**7.476 m/s**), while **Joint U-Net + MLP** is lowest without ERA5 (**7.243 m/s**). Per-storm behavior differs; the trajectory and storm-level table report that variation.

The plotted curves are hourly means to keep the dense 10-minute series readable. The table scores every valid individual observation, while the download also retains any explicitly flagged unusable scan.

![Predicted and ground-truth full-storm intensity trajectories](../assets/images/intensity-comparison/three-storm-intensity-trajectories.png)

![Per-storm dense inference MAE](../assets/images/intensity-comparison/three-storm-mae.png)

All values below are m/s except the sample count.

| Conditioning | Model | valid n / attempted | MAE | RMSE | Bias | Humberto MAE | Kiko MAE | Otis MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| With ERA5 | U-Net raw field maximum | 3266 / 3268 | 8.942 | 11.882 | -6.458 | 7.838 | 10.531 | 6.908 |
| With ERA5 | U-Net + correction | 3266 / 3268 | 7.476 | 10.302 | -4.430 | 5.614 | 9.011 | 6.677 |
| With ERA5 | Joint U-Net + MLP | 3266 / 3268 | 8.029 | 11.146 | -3.437 | 6.766 | 9.453 | 6.604 |
| Without ERA5 | U-Net raw field maximum | 3266 / 3268 | 9.113 | 11.404 | -4.968 | 11.271 | 8.586 | 7.153 |
| Without ERA5 | U-Net + correction | 3266 / 3268 | 7.345 | 9.452 | -1.483 | 8.338 | 7.160 | 6.314 |
| Without ERA5 | Joint U-Net + MLP | 3266 / 3268 | 7.243 | 9.159 | 0.071 | 7.631 | 6.962 | 7.318 |

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
