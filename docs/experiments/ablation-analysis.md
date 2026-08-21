# Diffusion refinement ablation results

This report is generated from the final machine-readable artifacts in
`logs/ablation-suites/refinement-rerun-20260731T130254Z/` and
`logs/ablation-suites/model-c-rerun-20260731T130254Z/`. It covers 17 training
ablations, nine Stage 2 checkpoint evaluations, the full K=10 Model C guidance
sweep, and calibration diagnostics.

`include_test_in_train: true` was retained as requested. These are controlled
comparisons, not unbiased held-out generalization estimates.

## Summary

- Stage 1 `peak_only` has the lowest peak/structure score (6.62) and
  robust-peak MAE (5.75 m/s); `peak_aware` has lower high-wind MAE and less
  negative peak bias.
- Stage 2 `structured_asinh` has the lowest probabilistic score (4.52), but all
  Stage 2 variants have negative pixel-MAE skill against the frozen baseline.
- Literal maximum and robust peak select different Stage 2 variants. The
  literal maximum is high-biased, whereas robust peaks are low-biased.
- Leave-one-storm-out calibration errors remain large and interval coverage is
  substantially below nominal.

## Model B / Stage 1 results

Lower is better for errors and the composite score. Peak bias closer to zero is
better; negative values indicate underestimation.

| Variant | Peak/structure ↓ | Peak bias m/s | Peak MAE ↓ | High-wind MAE ↓ | Radial MAE ↓ | RMW km ↓ | MAE skill |
|---|---|---|---|---|---|---|---|
| peak_only | 6.62 | -3.17 | 5.75 | 3.46 | 2.58 | 25.60 | 28.9% |
| peak_aware | 6.72 | -3.08 | 5.85 | 3.36 | 2.62 | 25.80 | 29.3% |
| peak_structure_balanced | 7.40 | -4.47 | 6.51 | 3.47 | 2.62 | 22.35 | 29.0% |
| highwind_only | 8.22 | -5.98 | 7.36 | 3.37 | 2.56 | 25.85 | 28.5% |
| radial_only | 8.89 | -6.80 | 8.02 | 3.42 | 2.60 | 24.50 | 28.0% |
| sampling_only | 9.32 | -7.43 | 8.43 | 3.36 | 2.59 | 24.05 | 28.1% |
| exceedance_only | 9.47 | -7.56 | 8.58 | 3.46 | 2.69 | 25.50 | 27.1% |
| control_finetune | 9.79 | -7.79 | 8.87 | 3.56 | 2.73 | 25.45 | 27.1% |


Relative to control, `stage1_peak_only` reduces robust-peak MAE from 8.87 to
5.75 m/s and changes peak bias from −7.79 to −3.17 m/s.

## Model C / Stage 2 training results

`baseline_mae_ms` is the frozen Model B reference. `recon_mae_ms` is the sampled
residual output. The probabilistic score is minimized by checkpoint selection.

| Variant | Prob. score ↓ | Baseline MAE | Refined MAE ↓ | Skill vs baseline | Peak bias | Peak MAE ↓ | Peak CRPS ↓ | Radial MAE ↓ |
|---|---|---|---|---|---|---|---|---|
| structured_asinh | 4.52 | 2.01 | 2.87 | -42.7% | 8.03 | 8.47 | 7.21 | 3.12 |
| structured_linear | 4.58 | 2.01 | 3.49 | -73.5% | 7.59 | 7.86 | 6.82 | 3.31 |
| annular_only | 4.68 | 2.01 | 3.64 | -81.0% | 7.57 | 8.18 | 6.57 | 3.50 |
| radial_only | 4.71 | 2.01 | 3.79 | -88.3% | 8.30 | 8.43 | 6.91 | 3.51 |
| weighting_only | 4.74 | 2.01 | 3.84 | -90.9% | 7.87 | 8.14 | 6.56 | 3.73 |
| multiscale_only | 4.77 | 2.01 | 3.70 | -83.8% | 7.97 | 8.34 | 6.87 | 3.64 |
| anchored_cfg | 4.80 | 2.01 | 3.72 | -85.1% | 8.37 | 8.55 | 6.99 | 3.68 |
| peak_only | 5.14 | 2.01 | 3.05 | -51.4% | 10.98 | 10.98 | 9.12 | 3.16 |
| exceedance_only | 5.58 | 2.01 | 4.07 | -102.3% | 11.94 | 11.94 | 9.40 | 3.91 |


All refined MAEs exceed the frozen baseline value of approximately 2.01 m/s.
`structured_asinh` has the lowest composite score and the least-negative skill.

## Stage 2 checkpoint evaluation

These are member-median summaries at guidance 1.2 over 360 observations. MSW is
the literal maximum pixel; robust peak is the mean of the highest 0.5% of valid
pixels.

| Variant | MSW bias | MSW MAE ↓ | MSW r | Robust bias | Robust MAE ↓ | Robust r |
|---|---|---|---|---|---|---|
| anchored_cfg | 25.03 | 26.24 | 0.22 | -5.13 | 9.23 | 0.88 |
| annular_only | 11.82 | 15.50 | 0.44 | -5.55 | 9.15 | 0.88 |
| exceedance_only | 22.59 | 24.07 | 0.24 | -3.73 | 9.03 | 0.87 |
| multiscale_only | 4.21 | 11.72 | 0.58 | -6.22 | 9.33 | 0.89 |
| peak_only | 21.00 | 21.21 | 0.66 | -1.99 | 8.51 | 0.89 |
| radial_only | 6.44 | 10.08 | 0.79 | -4.30 | 8.65 | 0.88 |
| structured_asinh | 14.91 | 15.31 | 0.83 | -2.85 | 8.81 | 0.89 |
| structured_linear | 8.96 | 10.77 | 0.86 | -2.74 | 9.01 | 0.88 |
| weighting_only | 24.41 | 25.38 | 0.24 | -4.38 | 8.94 | 0.88 |


Raw maximum and robust peak rank the variants differently. Checkpoint
selection therefore requires both statistics plus radial and threshold-area
errors.

## Model C guidance sweep

| Guidance | MSW bias | MSW MAE ↓ | MSW r | Robust bias | Robust MAE ↓ | Robust r | P10–P90 coverage | Width |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 20.50 | 22.16 | 0.24 | -11.00 | 13.34 | 0.76 | 26.1% | 18.71 |
| 1.2 | 19.32 | 21.59 | 0.24 | -10.72 | 13.36 | 0.75 | 26.4% | 18.40 |
| 1.5 | 17.52 | 21.01 | 0.20 | -10.23 | 13.37 | 0.72 | 24.4% | 17.82 |


Guidance 1.5 lowers raw maximum bias and MAE, narrows intervals, and does not
improve correlation.

### Target winds above 60 m/s

| Guidance | N | MSW bias >60 | MSW MAE >60 | Robust bias >60 | Robust MAE >60 |
|---|---|---|---|---|---|
| 1.0 | 90 | 2.87 | 8.50 | -24.09 | 24.09 |
| 1.2 | 90 | 2.03 | 9.05 | -23.85 | 23.85 |
| 1.5 | 90 | 0.41 | 10.18 | -23.43 | 23.43 |


Above 60 m/s, raw maxima are near zero bias while robust peaks are 23–24 m/s
too low. A scalar offset to the raw maximum would not correct this structural
difference.

## Leave-one-storm-out calibration

| Guidance | MSW affine MAE | MSW isotonic MAE | MSW affine bias | MSW isotonic bias | Peak affine MAE | Peak isotonic MAE | Peak affine bias | Peak isotonic bias |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 15.47 | 15.33 | -2.23 | -2.17 | 14.60 | 13.58 | 1.87 | -5.00 |
| 1.2 | 15.44 | 15.63 | -2.24 | -2.33 | 13.85 | 13.59 | 0.99 | -4.89 |
| 1.5 | 15.76 | 15.66 | -2.55 | -2.71 | 12.45 | 13.81 | -1.46 | -5.12 |


The leave-one-storm-out folds contain only two storms. Calibration is therefore
unstable, with poor coverage and weak correlations.

## Residual-transform diagnostic

The initial absolute residual q99.9 was 18.48 m/s and supported a 20 m/s linear
clip. With the peak/structure-aware Stage 1 model it increased to 24.04 m/s,
supporting a 25 m/s clip. Linear scaling therefore changes tail capacity.

## Implications

1. Use `stage1_peak_aware` or `stage1_peak_only` as the Stage 1 baseline.
2. Retrain `stage2_structured_asinh` on that selected baseline and require
   non-negative `mae_skill_vs_baseline` before promotion.
3. Make checkpoint selection multi-objective: penalize positive literal-MSW bias
   and negative robust-peak bias simultaneously, while retaining radial/area
   constraints.
4. Evaluate calibration with more storms and storm-level folds.
5. Report the ten-member ensemble with robust peak, high-wind-bin error,
   radial structure, and interval coverage.

## Post-processing follow-up

The completed gain/cap/median sweep is documented in [Model C post-processing ablation](postprocessing-ablation.md), including corrected storm-level calibration folds and machine-readable artifact paths.
