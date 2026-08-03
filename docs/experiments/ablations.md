# Diffusion refinement ablations

This page documents the structured ablation suite for the two-stage wind-field
reconstruction system. The suite is designed to answer two practical questions:

1. Which losses and sampling choices make the refined wind field less
   geometrically exaggerated?
2. Which choices reduce the persistent underestimation of maximum winds?

All ablations use the same data export, split policy, optimizer family, and
checkpoint selection rule within their stage. The checked-in experiment
configs intentionally keep `include_test_in_train: true`, matching the
existing dashboard workflow. These are therefore controlled ablations, not
unbiased held-out test estimates.

## Model mapping

This study covers two production models:

- **Model B** is the deterministic baseline UNet trained in Stage 1. The
  Stage 1 table measures the losses and sampling choices that shape this
  baseline.
- **Model C** is the stacked model: the frozen Model B field plus residual
  diffusion. The Stage 2 table measures the residual objective, and the K=10
  ensemble section evaluates its stochastic predictions and calibration.

There is no standalone Model A experiment in this suite.

## Selected checked-in defaults

The main presets now use the best overall-utility settings from the completed
suite: Model B uses `stage1_peak_aware` (smooth high-wind weighting plus the
robust inner-core peak term), and Model C uses the `stage2_structured_asinh`
objective with `guidance_scale: 1.2`. The Stage 2 runner hands off the completed
peak-aware Model B checkpoint by default. The control and balanced variants stay
unchanged so the ablation remains reproducible. The completed Stage 2 artifacts
were trained before this handoff change (on the balanced baseline), so retrain
Model C with the updated runner before deploying the new pair.

## Stage 1 ablations

Stage 1 predicts the dense deterministic wind field used as the Stage 2
baseline. The control run is compared with one-change variants and cumulative
variants:

| Config | Change under test |
| --- | --- |
| `config_stage1_control_finetune.yaml` | Existing fine-tuning objective and sampling |
| `config_stage1_highwind_only.yaml` | Continuous high-wind pixel weighting |
| `config_stage1_peak_only.yaml` | Robust inner-core/top-fraction peak loss |
| `config_stage1_radial_only.yaml` | Flip-aware radial-profile Huber loss |
| `config_stage1_exceedance_only.yaml` | Soft area losses at 17, 33, and 43 m/s |
| `config_stage1_sampling_only.yaml` | Intensity-balanced training sampler |
| `config_stage1_peak_aware.yaml` | Smooth high-wind weighting plus robust inner-core/top-fraction peak loss |
| `config_stage1_peak_structure_balanced.yaml` | Peak-aware objective plus intensity-balanced sampling |

The peak and structure metrics should be read together. `robust_peak_mae_ms`
measures maximum-wind error without allowing one noisy pixel to dominate;
`radial_profile_mae_ms`, `rmw_error_km`, and the threshold-area metrics measure
whether the storm shape is plausible. A useful scalar for checkpoint ranking is
`val/peak_structure_score`, but the component metrics should remain visible in
the ablation table.

## Stage 2 ablations

Stage 2 generates a residual around the frozen Stage 1 field. Each config
isolates a refinement mechanism, followed by two complete structured variants:

| Config | Change under test |
| --- | --- |
| `config_stage2_anchored_cfg.yaml` | Baseline-preserving classifier-free guidance/dropout |
| `config_stage2_weighting_only.yaml` | Noise-level and high-wind loss weighting |
| `config_stage2_peak_only.yaml` | Robust peak loss on the reconstructed field |
| `config_stage2_radial_only.yaml` | Radial-profile structure loss |
| `config_stage2_exceedance_only.yaml` | Threshold-area preservation |
| `config_stage2_multiscale_only.yaml` | Phase-aware multi-scale field loss |
| `config_stage2_annular_only.yaml` | Target-relative annular residual constraint |
| `config_stage2_structured_asinh.yaml` | Full structured objective with asinh residuals |
| `config_stage2_structured_linear.yaml` | Same objective with data-derived linear residual scaling |

The important distinction is that the annular term is target-relative: it
does not penalize a broad positive correction merely because the correction is
non-zero. This is intended to address maximum-wind underestimation while the
radial and multi-scale terms discourage isolated spikes or oversized rings.

## Model C ensemble and calibration

The Model C sweep uses the repository's current K=10 stochastic-member
definition, with paired member seeds across guidance scales 1.0, 1.2, and 1.5.
For every storm and observation it records member-level maximum wind,
robust top-0.5% peak, MAE, and threshold-area diagnostics. Summary fields use
the member median by default and retain the mean, p10, p90, and medoid for
comparison.

The calibration sweep fits both affine and isotonic maps for the member-median
maximum wind and robust peak. Outputs are bounded to 0--80 m/s and include
in-sample and leave-one-storm-out diagnostics. Use the leave-one-storm-out
numbers when deciding whether a calibration improves generalization; the
in-sample numbers are intentionally labelled optimistic.

## Machine-readable outputs

All training ablations are logged as separate runs in the shared W&B project `geo2wf-refinement-ablations`; Stage 1 and Stage 2 are separated with run groups.

The experiment runners are:

```bash
scripts/experiments/run_refinement_training_ablations.sh
scripts/experiments/run_model_c_ensemble_ablations.sh
```

The training runner skips completed run directories, so an interrupted
training suite can resume from the next unfinished ablation. Each training run
writes a `run-manifest.json`, `metric-history.jsonl`,
`result.json`, CSV metrics, and checkpoint provenance. Each ensemble member
writes `per-member-metrics.csv`, `inference-summary.csv`, and a metadata file
with the exact seeds, checkpoint hash, and source tree hash. The collector
flattens available artifacts into `ablation-results.json` and
`ablation-results.csv` under the suite directory.

The residual-distribution analysis is saved alongside the training suite and
records the full train-plus-test residual quantiles used to choose the linear
transform clip. No image artifacts are required for this study; image logging
is disabled in all ablation configs while numeric validation metrics remain
enabled.

## Recommended comparison order

1. Compare the Stage 1 control with `peak_aware` and
   `peak_structure_balanced` using robust peak bias/MAE, radial-profile error,
   and threshold-area bias.
2. Compare Stage 2 `structured_asinh` and `structured_linear` against the
   frozen-baseline skill metrics (`baseline_mae_ms` and
   `mae_skill_vs_baseline`).
3. Reject a configuration that improves peak error by producing implausible
   radial profiles or exceedance areas.
4. For Model C, report median peak error together with p10--p90 coverage and
   leave-one-storm-out calibration; do not rank models from the ensemble mean
   alone.
