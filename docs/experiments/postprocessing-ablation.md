# Model C post-processing ablation

This report evaluates cheap post-processing controls for Model C: residual
gain, a symmetric residual amplitude cap, and an optional 3×3 median filter on
the residual. No diffusion weights were changed. The sweep used the existing
guidance-1.2, K=10 Model C fields for 360 observations from `AL082025` and
`EP112025`, and the matching deterministic Stage-1 baseline field.

The experiment retained `include_test_in_train: true`. It is a controlled
diagnostic on two storms, not an unbiased generalization estimate. The Model C
checkpoint predates the corrected peak-aware Stage-1 handoff, so these results
are useful for selecting post-processing behavior but are not a deployment
replacement for retraining Model C on the current Model B checkpoint.

## What was tested

For every ensemble member, the transformed field was constructed as

```text
output = clip(baseline + smooth(clip(gain × (model_c − baseline), ±cap)), 0, 80)
```

The grid contained gains `{0, 0.25, 0.5, 0.75, 1.0}`, residual caps of
`±8`, `±16`, and `±24 m/s`, plus an uncapped control, and residual smoothing
of either none or a 3×3 median filter. This produced 40 variants. Literal MSW
is the maximum valid pixel; robust peak is the mean of the highest 0.5% of
valid pixels. Both are compared with the IBTrACS maximum sustained wind.

## Key results

The deterministic baseline (gain 0) has a literal-MSW MAE of **14.67 m/s** and
bias of **−14.55 m/s**. Its robust-peak MAE is **15.73 m/s** with bias
**−15.65 m/s**.

| Variant | MSW MAE | MSW bias | Robust-peak MAE | Robust-peak bias | High-wind MSW MAE (≥33 m/s) |
|---|---:|---:|---:|---:|---:|
| Baseline (gain 0) | 14.67 | −14.55 | 15.73 | −15.65 | 19.28 |
| gain 0.75, uncapped, raw | 14.85 | +6.29 | 13.86 | −12.70 | **11.07** |
| gain 0.75, cap 24, raw | 13.23 | −5.92 | 14.75 | −12.98 | 14.42 |
| gain 1.0, cap 8, raw | **11.27** | −9.09 | 13.54 | −12.81 | 14.57 |
| gain 1.0, cap 16, raw | 11.66 | −6.39 | 13.66 | −12.10 | 13.40 |
| gain 1.0, uncapped, raw | 21.59 | +19.32 | **13.36** | −10.72 | 15.22 |

The main findings are:

- Increasing the residual gain consistently improves the robust peak, but even
  full gain leaves a large robust-peak underestimation (−10.72 m/s globally
  and −16.83 m/s for targets ≥33 m/s). Post-processing alone cannot solve the
  high-wind tail.
- Uncapped full gain gives the best robust-peak MAE, but it creates implausible
  literal spikes: its median literal-minus-robust peak gap is about **30.1
  m/s** and its high-wind literal bias becomes **+11.99 m/s**. This is exactly
  the exaggerated-shape failure mode, so it should not be deployed.
- A cap around `±8 m/s` with full gain is the safest practical compromise. It
  gives the best literal-MSW MAE (11.27 m/s), limits the median literal-minus-
  robust gap to about **3.8 m/s**, and avoids the large positive tail bias of
  the uncapped variant. A `±16 m/s` cap allows more high-wind amplitude but
  increases the peak gap to roughly 7.2 m/s.
- The `gain 0.75, uncapped` variant nearly removes high-wind literal-MSW bias
  (−1.11 m/s), but its median literal-minus-robust gap is about **19.0 m/s**.
  It improves the score by producing isolated maxima, not by recovering a
  coherent storm structure; reject it without an explicit shape constraint.
- The 3×3 residual median filter does not improve the numeric errors. At
  gain 1/cap 8 it worsens MSW MAE from 11.27 to 12.06 m/s and robust-peak MAE
  from 13.54 to 13.78 m/s. It is therefore not a default fix; use it only as
  a separately justified artifact-suppression option.

## Calibration results

Affine and isotonic calibration were refit for literal MSW and robust peak for
every variant. The corrected inputs preserve explicit storm IDs, so the
leave-one-storm-out (LOSO) folds are valid. For the recommended raw gain-1,
cap-8 variant:

| Predictor | Method | LOSO MAE | LOSO bias | P10–P90 coverage | Mean width |
|---|---|---:|---:|---:|---:|
| Literal MSW | affine | 13.04 | +1.60 | 9.4% | 5.39 m/s |
| Literal MSW | isotonic | 12.70 | −4.42 | 10.3% | 2.74 m/s |
| Robust peak | affine | 13.84 | +1.62 | 8.1% | 4.66 m/s |
| Robust peak | isotonic | 13.04 | −4.57 | 11.7% | 2.80 m/s |

Calibration reduces systematic bias somewhat, but coverage remains far below a
nominal 80% interval. With only two storms, these calibration values should be
treated as diagnostics, not as a production calibration map. More independent
storms and storm-level folds are required.

## Recommendation

For the next Model C training/evaluation cycle, use **raw residual gain 1.0
with a conservative ±8 m/s post-processing cap** as the comparison control.
Keep uncapped full gain as a diagnostic upper-tail reference, not a candidate
default. Do not enable median smoothing by default. The persistent robust-peak
underestimate means the next meaningful intervention must be model-side:
retrain residual diffusion on the current peak-aware Model B, add explicit
high-wind/robust-peak weighting or a tail calibration head, and reject any
checkpoint whose peak improvement is explained by a growing literal-minus-
robust gap.

## Machine-readable artifacts

The completed suite is
`logs/ablation-suites/diffusion-postprocess-20260803T111500Z/`:

- `postprocess/postprocess-results.csv` — flat metrics for all 40 variants;
- `postprocess/postprocess-metadata.json` — sweep definition and per-variant
  evaluations;
- `postprocess/variants/<variant>/inference-summary.csv` — row-level metrics;
- `postprocess/calibration/<variant>/<metric>/<method>/` — calibration models,
  predictions, and corrected LOSO evaluations;
- `postprocess/calibration-index.csv` — calibration artifact index;
- `baseline/` — matching Stage-1 baseline fields and manifest.

The reproducible launcher is
`scripts/experiments/run_diffusion_postprocess_ablations.sh`.
