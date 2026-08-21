# Stage 2 post-processing ablation

This report evaluates cheap post-processing controls for Model C: residual
gain, a symmetric residual amplitude cap, and an optional 3×3 median filter on
the residual. No diffusion weights were changed. The sweep used the existing
guidance-1.2, K=10 Model C fields for 360 observations from `AL082025` and
`EP112025`, and the matching deterministic Stage-1 baseline field.

The experiment retained `include_test_in_train: true`. It is a controlled
diagnostic on two storms, not an unbiased generalization estimate. The Model C
checkpoint predates the corrected peak-aware Stage-1 handoff, so these results
describe post-processing sensitivity but do not replace retraining Stage 2 on
the current Stage 1 checkpoint.

## Sweep

For every ensemble member, the transformed field was constructed as

```text
output = clip(baseline + smooth(clip(gain × (model_c − baseline), ±cap)), 0, 80)
```

The grid contained gains `{0, 0.25, 0.5, 0.75, 1.0}`, residual caps of
`±8`, `±16`, and `±24 m/s`, plus an uncapped control, and residual smoothing
of either none or a 3×3 median filter. This produced 40 variants. Literal MSW
is the maximum valid pixel; robust peak is the mean of the highest 0.5% of
valid pixels. Both are compared with the IBTrACS maximum sustained wind.

## Results

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

The results show:

- Full gain retains robust-peak underestimation: −10.72 m/s overall and
  −16.83 m/s for targets of at least 33 m/s.
- Uncapped full gain yields the lowest robust-peak MAE but a 30.1 m/s median
  literal-minus-robust gap and +11.99 m/s high-wind literal bias.
- Full gain with an ±8 m/s cap yields the lowest literal-maximum MAE
  (11.27 m/s) and limits the median peak gap to 3.8 m/s. An ±16 m/s cap
  increases the gap to 7.2 m/s.
- At gain 0.75 without a cap, high-wind literal bias is −1.11 m/s but the
  median peak gap is 19.0 m/s, indicating isolated maxima.
- The 3 × 3 median filter increases both maximum and robust-peak MAE at full
  gain with an ±8 m/s cap.

## Calibration results

Affine and isotonic calibration were refit for literal MSW and robust peak for
every variant. The corrected inputs preserve explicit storm IDs, so the
leave-one-storm-out (LOSO) folds are valid. For the selected raw gain-1,
cap-8 variant:

| Predictor | Method | LOSO MAE | LOSO bias | P10–P90 coverage | Mean width |
|---|---|---:|---:|---:|---:|
| Literal MSW | affine | 13.04 | +1.60 | 9.4% | 5.39 m/s |
| Literal MSW | isotonic | 12.70 | −4.42 | 10.3% | 2.74 m/s |
| Robust peak | affine | 13.84 | +1.62 | 8.1% | 4.66 m/s |
| Robust peak | isotonic | 13.04 | −4.57 | 11.7% | 2.80 m/s |

Calibration reduces some systematic bias, but coverage remains far below the
nominal 80%. The two-storm folds are diagnostic only.

## Selected control

The comparison control is raw residual gain 1.0 with an ±8 m/s cap. Uncapped
full gain remains an upper-tail diagnostic; median smoothing is disabled. The
persistent robust-peak underestimate requires a model-side change and
retraining on the current peak-aware Stage 1 checkpoint.

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
