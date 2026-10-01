# Intensity model comparison (test)

| Model | Samples | Storms | Intensity MAE (m/s; 95% CI) | Δ MAE vs raw U-Net (m/s; 95% CI) | Intensity RMSE (m/s) | Intensity bias (m/s) | Storm-macro MAE (m/s) | Category accuracy | Category macro F1 | Within one category | Field MAE (m/s) | Field RMSE (m/s) | Field bias (m/s) | Field PSNR (dB) | Field SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net raw field diagnostic | 212 | 38 | 7.518 (6.053–9.144) | 0.000 (0.000–0.000) | 10.233 | -4.797 | 7.037 | 0.542 | 0.251 | 0.868 | 3.888 | 5.337 | -0.484 | 23.495 | 0.809 |
| U-Net + correction | 212 | 38 | 6.715 (5.515–8.084) | -0.803 (-1.755–0.182) | 9.045 | -1.178 | 6.928 | 0.594 | 0.362 | 0.920 | — | — | — | — | — |
| Joint U-Net + MLP | 212 | 38 | 5.885 (5.047–6.854) | -1.633 (-2.701–-0.526) | 7.806 | -0.851 | 6.123 | 0.637 | 0.339 | 0.939 | 4.170 | 5.597 | -0.393 | 23.081 | 0.798 |

## Rapid-intensification phases

The same scalar metrics are recomputed only where IBTrACS maximum wind increased by at least 30 kt in the preceding 24 hours. Field metrics are omitted because the RI subset is defined per sample.

| Model | Samples | Storms | Intensity MAE (m/s; 95% CI) | Δ MAE vs raw U-Net (m/s; 95% CI) | Intensity RMSE (m/s) | Intensity bias (m/s) | Storm-macro MAE (m/s) | Category accuracy | Category macro F1 | Within one category | Field MAE (m/s) | Field RMSE (m/s) | Field bias (m/s) | Field PSNR (dB) | Field SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net raw field diagnostic | 19 | 11 | 17.859 (10.914–24.483) | 0.000 (0.000–0.000) | 21.051 | -17.859 | 16.143 | 0.211 | 0.213 | 0.421 | 3.298 | 4.556 | 0.266 | 24.869 | 0.809 |
| U-Net + correction | 19 | 11 | 12.507 (5.438–19.416) | -5.353 (-7.777–-2.220) | 16.913 | -11.995 | 10.733 | 0.421 | 0.397 | 0.684 | — | — | — | — | — |
| Joint U-Net + MLP | 19 | 11 | 8.934 (4.277–13.946) | -8.926 (-11.776–-4.843) | 12.068 | -6.241 | 7.206 | 0.316 | 0.303 | 0.789 | 3.699 | 4.915 | 0.427 | 24.210 | 0.795 |

## What the models predict

- **U-Net raw field maximum** is the largest predicted wind speed over all finite, valid pixels in the separately trained U-Net field.
- **U-Net + correction** starts from that same raw maximum and adds the output of a separately trained correction network. The correction network sees the frozen U-Net field, its validity mask, distance to the storm center, and contemporaneously available metadata. Its final intensity is clamped to be non-negative.
- **Joint U-Net + MLP** directly predicts scalar maximum wind from an MLP attached to the U-Net bottleneck. Its field reconstruction and scalar head are optimized jointly and share the encoder.

The correction model emits only a scalar, so field MAE, RMSE, and bias are not applicable and are shown as —.

## Evaluation cohort and target

The `test` cohort contains **212 samples from 38 storms**. Every row in every table uses the exact same sample IDs and storm-disjoint split. The scalar target is IBTrACS `USA_WIND`, expressed in m/s and linearly interpolated to the image timestamp only when the surrounding IBTrACS fixes are no more than three hours apart.

## Metric definitions

Let `prediction - target` be the signed scalar intensity error for one sample.

| Metric | How it is calculated | How to read it |
|---|---|---|
| **Samples / Storms** | Counts of image observations and unique storm IDs in the evaluation cohort. | These counts must match across model rows for a paired comparison. |
| **Intensity MAE** | Mean of `abs(prediction - target)` over samples. | Typical absolute scalar-intensity miss; lower is better. The parenthesized range is the 95% storm-bootstrap interval. |
| **Δ MAE vs raw U-Net** | A model's MAE minus the raw U-Net MAE, computed on the same bootstrap resample. | Negative values favor the model; a 95% interval excluding zero indicates a consistent paired improvement at the sampled-storm level. |
| **Intensity RMSE** | Square root of the mean squared scalar error over samples. | Penalizes large intensity misses more heavily than MAE; lower is better. |
| **Intensity bias** | Mean of `prediction - target` over samples. | Zero is ideal; negative means systematic underprediction and positive means overprediction. Opposite errors can cancel. |
| **Storm-macro MAE** | Compute MAE separately for every storm, then take the unweighted mean over storms. | Gives each storm equal weight regardless of how many images it contributes; lower is better. |
| **Category accuracy** | Fraction whose predicted and target intensity categories match exactly. | Higher is better; range 0–1. |
| **Category macro F1** | For each target category present, calculate the harmonic mean of precision and recall, then average categories equally. | Rewards balanced category performance rather than dominance by frequent categories; higher is better. |
| **Within one category** | Fraction for which the numeric predicted category differs from the target category by at most one. | Measures tolerance to a one-bin miss; higher is better. |
| **Field MAE** | Mean absolute U-Net-versus-SAR wind error over all valid pixels. | Typical pixel-level wind-field miss; lower is better. |
| **Field RMSE** | Square root of mean squared U-Net-versus-SAR error over all valid pixels. | More sensitive than field MAE to large pixel errors; lower is better. |
| **Field bias** | Mean signed U-Net-minus-SAR error over all valid pixels. | Zero is ideal; negative means field underprediction and positive means overprediction. |
| **Field PSNR** | `20 log10(79.8 m/s / field RMSE)` using the fixed 0.2–80.0 m/s SAR export range. | Peak signal-to-noise ratio in physical space; higher is better. |
| **Field SSIM** | Mean scene-level structural similarity after clipping to 0.2–80.0 m/s; only complete 7×7 windows containing valid pixels in both prediction and target are retained. | Local structural fidelity on a 0–1 scale; higher is better. |

### Intensity categories

Continuous one-minute wind is converted without rounding using these thresholds: tropical depression `<34 kt`; tropical storm `34–<64 kt`; Category 1 `64–<83 kt`; Category 2 `83–<96 kt`; Category 3 `96–<113 kt`; Category 4 `113–<137 kt`; and Category 5 `≥137 kt` (`1 kt = 0.514444 m/s`).

### Valid pixels and uncertainty

Field metrics pool pixels over the intersection of the SAR target mask, condition mask, and finite predictions/targets; the cached raw U-Net also uses its exported validity mask. They are therefore pixel-weighted, not sample- or storm-macro metrics.

The reported 95% intervals use **2,000 paired cluster-bootstrap repetitions over storms** with seed `42`. Each repetition samples storm IDs with replacement and evaluates all models on that identical resample, preserving within-storm dependence and the pairing between models. Bounds are the 2.5th and 97.5th percentiles.

These are held-out test results and should not be used for further model selection.
