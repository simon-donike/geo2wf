# Diffusion refinement ablations

This suite tests high-wind and structural objectives for the deterministic
Stage 1 U-Net (Model B) and residual-diffusion Stage 2 model (Model C). Runs
within a stage use the same export, split policy, optimizer family, and
checkpoint rule.

!!! warning "Controlled comparison"
    The ablation configs retain `include_test_in_train: true` for compatibility
    with the original dashboard workflow. Results are not held-out test
    estimates.

## Selected configuration

The grouped Stage 1 default incorporates the `stage1_peak_aware` objective:
smooth high-wind weighting and a robust inner-core peak term. The Stage 2
default uses the `stage2_structured_asinh` objective and guidance scale 1.2.
Completed Stage 2 artifacts described in the results page used the earlier
balanced Stage 1 checkpoint; the current pair therefore requires retraining.

## Stage 1 variants

Files are under `configs/ablations/`.

| Config stem | Isolated change |
|---|---|
| `config_stage1_control_finetune` | historical fine-tuning objective and sampling |
| `config_stage1_highwind_only` | continuous high-wind pixel weighting |
| `config_stage1_peak_only` | robust inner-core top-fraction peak loss |
| `config_stage1_radial_only` | flip-aware radial-profile Huber loss |
| `config_stage1_exceedance_only` | soft area losses at 17 m/s (33.0 kt), 33 m/s (64.1 kt), and 43 m/s (83.6 kt) |
| `config_stage1_sampling_only` | intensity-balanced training sampler |
| `config_stage1_peak_aware` | high-wind weighting plus peak loss |
| `config_stage1_peak_structure_balanced` | peak-aware objective plus balanced sampling |

`robust_peak_mae_ms` reduces sensitivity to one extreme pixel. It must be read
with radial-profile error, RMW error, and threshold-area metrics. The composite
checkpoint metric is `val/peak_structure_score`.

## Stage 2 variants

| Config stem | Isolated change |
|---|---|
| `config_stage2_anchored_cfg` | baseline-preserving classifier-free guidance/dropout |
| `config_stage2_weighting_only` | noise-level and high-wind weighting |
| `config_stage2_peak_only` | robust peak loss on the reconstructed field |
| `config_stage2_radial_only` | radial-profile loss |
| `config_stage2_exceedance_only` | threshold-area preservation |
| `config_stage2_multiscale_only` | phase-aware multi-scale field loss |
| `config_stage2_annular_only` | target-relative annular residual constraint |
| `config_stage2_structured_asinh` | combined objective with asinh residuals |
| `config_stage2_structured_linear` | combined objective with linear residual scaling |

The annular term is target-relative and permits a broad positive correction.
Radial and multi-scale terms constrain isolated maxima and oversized rings.

## Ensemble and calibration protocol

Model C uses ten paired member seeds at guidance scales 1.0, 1.2, and 1.5.
Each observation records member-level maximum wind, top-0.5% robust peak, MAE,
and threshold areas. Summary fields retain the median, mean, p10, p90, and
medoid.

Affine and isotonic calibration maps are fitted separately to the member-median
maximum and robust peak. Outputs are bounded to 0–80 m/s (0–155.5 kt). Storm-level
leave-one-out results are the relevant diagnostic; in-sample results are
optimistic.

## Reproduction and outputs

```bash
scripts/experiments/run_refinement_training_ablations.sh
scripts/experiments/run_model_c_ensemble_ablations.sh
```

Training runs write `run-manifest.json`, `metric-history.jsonl`, `result.json`,
CSV metrics, and checkpoint provenance. Ensemble runs add member metrics,
inference summaries, exact seeds, checkpoint hashes, and source-tree hashes.
The collector writes `ablation-results.json` and `ablation-results.csv`.

Compare Stage 1 variants using peak, radial, RMW, and exceedance-area errors.
Compare Stage 2 variants against their frozen baseline using
`baseline_mae_ms` and `mae_skill_vs_baseline`, then report ensemble interval
coverage. Do not select a model from a single peak statistic or the ensemble
mean alone.

See the [completed numerical analysis](ablation-analysis.md) and
[post-processing sweep](postprocessing-ablation.md).
