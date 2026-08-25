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
`±8 m/s (±15.6 kt)`, `±16 m/s (±31.1 kt)`, and `±24 m/s (±46.7 kt)`, plus an uncapped control, and residual smoothing
of either none or a 3×3 median filter. This produced 40 variants. Literal MSW
is the maximum valid pixel; robust peak is the mean of the highest 0.5% of
valid pixels. Both are compared with the IBTrACS maximum sustained wind.

## Results

The deterministic baseline (gain 0) has a literal-MSW MAE of **14.67 m/s
(28.52 kt)** and bias of **−14.55 m/s (−28.28 kt)**. Its robust-peak MAE is
**15.73 m/s (30.58 kt)** with bias **−15.65 m/s (−30.42 kt)**.

| Variant | MSW MAE, m/s (kt) | MSW bias, m/s (kt) | Robust-peak MAE, m/s (kt) | Robust-peak bias, m/s (kt) | High-wind MSW MAE (target ≥33 m/s (64.1 kt)), m/s (kt) |
|---|---:|---:|---:|---:|---:|
| Baseline (gain 0) | 14.67 (28.52 kt) | −14.55 (−28.28 kt) | 15.73 (30.58 kt) | −15.65 (−30.42 kt) | 19.28 (37.48 kt) |
| gain 0.75, uncapped, raw | 14.85 (28.87 kt) | +6.29 (12.23 kt) | 13.86 (26.94 kt) | −12.70 (−24.69 kt) | **11.07 (21.52 kt)** |
| gain 0.75, cap 24, raw | 13.23 (25.72 kt) | −5.92 (−11.51 kt) | 14.75 (28.67 kt) | −12.98 (−25.23 kt) | 14.42 (28.03 kt) |
| gain 1.0, cap 8, raw | **11.27 (21.91 kt)** | −9.09 (−17.67 kt) | 13.54 (26.32 kt) | −12.81 (−24.90 kt) | 14.57 (28.32 kt) |
| gain 1.0, cap 16, raw | 11.66 (22.67 kt) | −6.39 (−12.42 kt) | 13.66 (26.55 kt) | −12.10 (−23.52 kt) | 13.40 (26.05 kt) |
| gain 1.0, uncapped, raw | 21.59 (41.97 kt) | +19.32 (37.56 kt) | **13.36 (25.97 kt)** | −10.72 (−20.84 kt) | 15.22 (29.59 kt) |

The results show:

- Full gain retains robust-peak underestimation: −10.72 m/s (−20.84 kt) overall
  and −16.83 m/s (−32.71 kt) for targets of at least 33 m/s (64.1 kt).
- Uncapped full gain yields the lowest robust-peak MAE but a 30.1 m/s (58.5 kt)
  median literal-minus-robust gap and +11.99 m/s (+23.31 kt) high-wind literal bias.
- Full gain with an ±8 m/s (±15.6 kt) cap yields the lowest literal-maximum MAE
  (11.27 m/s (21.91 kt)) and limits the median peak gap to 3.8 m/s (7.4 kt). An
  ±16 m/s (±31.1 kt) cap increases the gap to 7.2 m/s (14.0 kt).
- At gain 0.75 without a cap, high-wind literal bias is −1.11 m/s (−2.16 kt)
  but the median peak gap is 19.0 m/s (36.9 kt), indicating isolated maxima.
- The 3 × 3 median filter increases both maximum and robust-peak MAE at full
  gain with an ±8 m/s (±15.6 kt) cap.

## Calibration results

Affine and isotonic calibration were refit for literal MSW and robust peak for
every variant. The corrected inputs preserve explicit storm IDs, so the
leave-one-storm-out (LOSO) folds are valid. For the selected raw gain-1,
cap-8 variant:

| Predictor | Method | LOSO MAE, m/s (kt) | LOSO bias, m/s (kt) | P10–P90 coverage | Mean width, m/s (kt) |
|---|---|---:|---:|---:|---:|
| Literal MSW | affine | 13.04 (25.35 kt) | +1.60 (3.11 kt) | 9.4% | 5.39 (10.48 kt) |
| Literal MSW | isotonic | 12.70 (24.69 kt) | −4.42 (−8.59 kt) | 10.3% | 2.74 (5.33 kt) |
| Robust peak | affine | 13.84 (26.90 kt) | +1.62 (3.15 kt) | 8.1% | 4.66 (9.06 kt) |
| Robust peak | isotonic | 13.04 (25.35 kt) | −4.57 (−8.88 kt) | 11.7% | 2.80 (5.44 kt) |

Calibration reduces some systematic bias, but coverage remains far below the
nominal 80%. The two-storm folds are diagnostic only.

## Selected control

The comparison control is raw residual gain 1.0 with an ±8 m/s (±15.6 kt) cap. Uncapped
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
