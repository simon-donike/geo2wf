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
  robust-peak MAE (5.75 m/s (11.18 kt)); `peak_aware` has lower high-wind MAE and less
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

| Variant | Peak/structure ↓ | Peak bias, m/s (kt) | Peak MAE, m/s (kt) ↓ | High-wind MAE, m/s (kt) ↓ | Radial MAE, m/s (kt) ↓ | RMW km ↓ | MAE skill |
|---|---|---|---|---|---|---|---|
| peak_only | 6.62 | -3.17 (-6.16 kt) | 5.75 (11.18 kt) | 3.46 (6.73 kt) | 2.58 (5.02 kt) | 25.60 | 28.9% |
| peak_aware | 6.72 | -3.08 (-5.99 kt) | 5.85 (11.37 kt) | 3.36 (6.53 kt) | 2.62 (5.09 kt) | 25.80 | 29.3% |
| peak_structure_balanced | 7.40 | -4.47 (-8.69 kt) | 6.51 (12.65 kt) | 3.47 (6.75 kt) | 2.62 (5.09 kt) | 22.35 | 29.0% |
| highwind_only | 8.22 | -5.98 (-11.62 kt) | 7.36 (14.31 kt) | 3.37 (6.55 kt) | 2.56 (4.98 kt) | 25.85 | 28.5% |
| radial_only | 8.89 | -6.80 (-13.22 kt) | 8.02 (15.59 kt) | 3.42 (6.65 kt) | 2.60 (5.05 kt) | 24.50 | 28.0% |
| sampling_only | 9.32 | -7.43 (-14.44 kt) | 8.43 (16.39 kt) | 3.36 (6.53 kt) | 2.59 (5.03 kt) | 24.05 | 28.1% |
| exceedance_only | 9.47 | -7.56 (-14.70 kt) | 8.58 (16.68 kt) | 3.46 (6.73 kt) | 2.69 (5.23 kt) | 25.50 | 27.1% |
| control_finetune | 9.79 | -7.79 (-15.14 kt) | 8.87 (17.24 kt) | 3.56 (6.92 kt) | 2.73 (5.31 kt) | 25.45 | 27.1% |


Relative to control, `stage1_peak_only` reduces robust-peak MAE from 8.87 to
5.75 m/s (11.18 kt) and changes peak bias from −7.79 m/s (−15.14 kt) to
−3.17 m/s (−6.16 kt).

## Model C / Stage 2 training results

`baseline_mae_ms` is the frozen Model B reference. `recon_mae_ms` is the sampled
residual output. The probabilistic score is minimized by checkpoint selection.

| Variant | Prob. score ↓ | Baseline MAE, m/s (kt) | Refined MAE, m/s (kt) ↓ | Skill vs baseline | Peak bias, m/s (kt) | Peak MAE, m/s (kt) ↓ | Peak CRPS, m/s (kt) ↓ | Radial MAE, m/s (kt) ↓ |
|---|---|---|---|---|---|---|---|---|
| structured_asinh | 4.52 | 2.01 (3.91 kt) | 2.87 (5.58 kt) | -42.7% | 8.03 (15.61 kt) | 8.47 (16.46 kt) | 7.21 (14.02 kt) | 3.12 (6.06 kt) |
| structured_linear | 4.58 | 2.01 (3.91 kt) | 3.49 (6.78 kt) | -73.5% | 7.59 (14.75 kt) | 7.86 (15.28 kt) | 6.82 (13.26 kt) | 3.31 (6.43 kt) |
| annular_only | 4.68 | 2.01 (3.91 kt) | 3.64 (7.08 kt) | -81.0% | 7.57 (14.71 kt) | 8.18 (15.90 kt) | 6.57 (12.77 kt) | 3.50 (6.80 kt) |
| radial_only | 4.71 | 2.01 (3.91 kt) | 3.79 (7.37 kt) | -88.3% | 8.30 (16.13 kt) | 8.43 (16.39 kt) | 6.91 (13.43 kt) | 3.51 (6.82 kt) |
| weighting_only | 4.74 | 2.01 (3.91 kt) | 3.84 (7.46 kt) | -90.9% | 7.87 (15.30 kt) | 8.14 (15.82 kt) | 6.56 (12.75 kt) | 3.73 (7.25 kt) |
| multiscale_only | 4.77 | 2.01 (3.91 kt) | 3.70 (7.19 kt) | -83.8% | 7.97 (15.49 kt) | 8.34 (16.21 kt) | 6.87 (13.35 kt) | 3.64 (7.08 kt) |
| anchored_cfg | 4.80 | 2.01 (3.91 kt) | 3.72 (7.23 kt) | -85.1% | 8.37 (16.27 kt) | 8.55 (16.62 kt) | 6.99 (13.59 kt) | 3.68 (7.15 kt) |
| peak_only | 5.14 | 2.01 (3.91 kt) | 3.05 (5.93 kt) | -51.4% | 10.98 (21.34 kt) | 10.98 (21.34 kt) | 9.12 (17.73 kt) | 3.16 (6.14 kt) |
| exceedance_only | 5.58 | 2.01 (3.91 kt) | 4.07 (7.91 kt) | -102.3% | 11.94 (23.21 kt) | 11.94 (23.21 kt) | 9.40 (18.27 kt) | 3.91 (7.60 kt) |


All refined MAEs exceed the frozen baseline value of approximately 2.01 m/s (3.91 kt).
`structured_asinh` has the lowest composite score and the least-negative skill.

## Stage 2 checkpoint evaluation

These are member-median summaries at guidance 1.2 over 360 observations. MSW is
the literal maximum pixel; robust peak is the mean of the highest 0.5% of valid
pixels.

| Variant | MSW bias, m/s (kt) | MSW MAE, m/s (kt) ↓ | MSW r | Robust bias, m/s (kt) | Robust MAE, m/s (kt) ↓ | Robust r |
|---|---|---|---|---|---|---|
| anchored_cfg | 25.03 (48.65 kt) | 26.24 (51.01 kt) | 0.22 | -5.13 (-9.97 kt) | 9.23 (17.94 kt) | 0.88 |
| annular_only | 11.82 (22.98 kt) | 15.50 (30.13 kt) | 0.44 | -5.55 (-10.79 kt) | 9.15 (17.79 kt) | 0.88 |
| exceedance_only | 22.59 (43.91 kt) | 24.07 (46.79 kt) | 0.24 | -3.73 (-7.25 kt) | 9.03 (17.55 kt) | 0.87 |
| multiscale_only | 4.21 (8.18 kt) | 11.72 (22.78 kt) | 0.58 | -6.22 (-12.09 kt) | 9.33 (18.14 kt) | 0.89 |
| peak_only | 21.00 (40.82 kt) | 21.21 (41.23 kt) | 0.66 | -1.99 (-3.87 kt) | 8.51 (16.54 kt) | 0.89 |
| radial_only | 6.44 (12.52 kt) | 10.08 (19.59 kt) | 0.79 | -4.30 (-8.36 kt) | 8.65 (16.81 kt) | 0.88 |
| structured_asinh | 14.91 (28.98 kt) | 15.31 (29.76 kt) | 0.83 | -2.85 (-5.54 kt) | 8.81 (17.13 kt) | 0.89 |
| structured_linear | 8.96 (17.42 kt) | 10.77 (20.94 kt) | 0.86 | -2.74 (-5.33 kt) | 9.01 (17.51 kt) | 0.88 |
| weighting_only | 24.41 (47.45 kt) | 25.38 (49.33 kt) | 0.24 | -4.38 (-8.51 kt) | 8.94 (17.38 kt) | 0.88 |


Raw maximum and robust peak rank the variants differently. Checkpoint
selection therefore requires both statistics plus radial and threshold-area
errors.

## Model C guidance sweep

| Guidance | MSW bias, m/s (kt) | MSW MAE, m/s (kt) ↓ | MSW r | Robust bias, m/s (kt) | Robust MAE, m/s (kt) ↓ | Robust r | P10–P90 coverage | Width, m/s (kt) |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 20.50 (39.85 kt) | 22.16 (43.08 kt) | 0.24 | -11.00 (-21.38 kt) | 13.34 (25.93 kt) | 0.76 | 26.1% | 18.71 (36.37 kt) |
| 1.2 | 19.32 (37.56 kt) | 21.59 (41.97 kt) | 0.24 | -10.72 (-20.84 kt) | 13.36 (25.97 kt) | 0.75 | 26.4% | 18.40 (35.77 kt) |
| 1.5 | 17.52 (34.06 kt) | 21.01 (40.84 kt) | 0.20 | -10.23 (-19.89 kt) | 13.37 (25.99 kt) | 0.72 | 24.4% | 17.82 (34.64 kt) |


Guidance 1.5 lowers raw maximum bias and MAE, narrows intervals, and does not
improve correlation.

### Target winds above 60 m/s (116.6 kt)

| Guidance | N | MSW bias, m/s (kt), target >60 m/s (116.6 kt) | MSW MAE, m/s (kt), target >60 m/s (116.6 kt) | Robust bias, m/s (kt), target >60 m/s (116.6 kt) | Robust MAE, m/s (kt), target >60 m/s (116.6 kt) |
|---|---|---|---|---|---|
| 1.0 | 90 | 2.87 (5.58 kt) | 8.50 (16.52 kt) | -24.09 (-46.83 kt) | 24.09 (46.83 kt) |
| 1.2 | 90 | 2.03 (3.95 kt) | 9.05 (17.59 kt) | -23.85 (-46.36 kt) | 23.85 (46.36 kt) |
| 1.5 | 90 | 0.41 (0.80 kt) | 10.18 (19.79 kt) | -23.43 (-45.54 kt) | 23.43 (45.54 kt) |


Above 60 m/s (116.6 kt), raw maxima are near zero bias while robust peaks are
23–24 m/s (44.7–46.7 kt) too low. A scalar offset to the raw maximum would not correct this structural
difference.

## Leave-one-storm-out calibration

| Guidance | MSW affine MAE, m/s (kt) | MSW isotonic MAE, m/s (kt) | MSW affine bias, m/s (kt) | MSW isotonic bias, m/s (kt) | Peak affine MAE, m/s (kt) | Peak isotonic MAE, m/s (kt) | Peak affine bias, m/s (kt) | Peak isotonic bias, m/s (kt) |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 15.47 (30.07 kt) | 15.33 (29.80 kt) | -2.23 (-4.33 kt) | -2.17 (-4.22 kt) | 14.60 (28.38 kt) | 13.58 (26.40 kt) | 1.87 (3.63 kt) | -5.00 (-9.72 kt) |
| 1.2 | 15.44 (30.01 kt) | 15.63 (30.38 kt) | -2.24 (-4.35 kt) | -2.33 (-4.53 kt) | 13.85 (26.92 kt) | 13.59 (26.42 kt) | 0.99 (1.92 kt) | -4.89 (-9.51 kt) |
| 1.5 | 15.76 (30.64 kt) | 15.66 (30.44 kt) | -2.55 (-4.96 kt) | -2.71 (-5.27 kt) | 12.45 (24.20 kt) | 13.81 (26.84 kt) | -1.46 (-2.84 kt) | -5.12 (-9.95 kt) |


The leave-one-storm-out folds contain only two storms. Calibration is therefore
unstable, with poor coverage and weak correlations.

## Residual-transform diagnostic

The initial absolute residual q99.9 was 18.48 m/s (35.92 kt) and supported a
20 m/s (38.9 kt) linear clip. With the peak/structure-aware Stage 1 model it
increased to 24.04 m/s (46.73 kt), supporting a 25 m/s (48.6 kt) clip. Linear
scaling therefore changes tail capacity.

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
