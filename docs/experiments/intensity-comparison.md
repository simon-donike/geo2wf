# Active experiment matrix

The next experiment cycle uses one fixed, storm-disjoint cohort and keeps ERA5
conditioning as the only input-regime difference. No results are reported on
this page until the new matrix has been run.

The [previous seed-42 benchmark](../archived/results/intensity-comparison-results.md)
and all of its published artifacts are archived for provenance.

## Three-model ERA5 comparison

Each of the following models is trained and evaluated once with ERA5 inputs and
once without them:

| Model | Output used for intensity evaluation | With ERA5 | Without ERA5 |
|---|---|---|---|
| Raw field U-Net | maximum diagnosed from the predicted 2D wind field | `intensity_comparison_unet` | `intensity_comparison_unet_no_era5` |
| U-Net + scalar correction | corrected maximum from a frozen U-Net field | ERA5 cache + `unet_intensity_correction` | no-ERA5 cache + `unet_intensity_correction` |
| Joint U-Net + bottleneck MLP | 2D wind field plus MLP maximum wind | `bottleneck_unet_mlp` | `bottleneck_unet_mlp_no_era5` |

The paired regimes must use identical sample IDs, storm splits, scalar targets,
crop settings, and evaluation code. The no-ERA5 runs still require ERA5
availability while selecting the cohort, but do not pass ERA5 values to the
model. This isolates conditioning rather than changing data availability.

## Joint U-Net/latent-MLP structure experiment

The separate structure study uses the joint U-Net + latent MLP so every model
retains both the decoded 2D wind field and the bottleneck outputs. Two runs use
the same ERA5-conditioned cohort, architecture, seed, and optimizer. Strict
CUDA determinism is disabled in both because reflection-padding backward has no
deterministic CUDA implementation:

```bash
uv run geo2wf-train experiment=bottleneck_unet_mlp_max_wind
uv run geo2wf-train experiment=bottleneck_unet_mlp_max_wind_radii
```

The maximum-wind baseline optimizes the field and scalar maximum-wind losses.
The multi-task run additionally predicts RMW and equivalent-area R34, R50, and
R64 with a masked latent-head loss weighted by `0.25`. Eye size is excluded
from reporting because the frozen cohort contains no valid eye-size labels.

Evaluation has three explicitly labeled output views:

1. maximum wind only;
2. maximum wind plus radii predicted directly by the latent MLP head; and
3. maximum wind plus radii diagnosed from the predicted 2D U-Net wind field.

The radii comparison must report MLP-derived and image-derived values as
different sources. It must not silently substitute one when the other is
missing. Checkpoints are selected on validation data; the held-out test split
is evaluated once after both runs finish.

## Retained forecast

The six-hour scalar intensity forecast remains active as a separate downstream
experiment. It is not one of the three instantaneous reconstruction models and
does not enter the ERA5/no-ERA5 matrix.

```bash
uv run geo2wf-train experiment=intensity_forecast_pretrain
uv run geo2wf-train experiment=intensity_forecast_finetune
```

## Comparison contract

- Freeze one cohort before training any matrix cell.
- Keep train, validation, and test storms disjoint.
- Use the same scalar reference and field target in all six ERA5 matrix cells.
- Select checkpoints from validation only; evaluate the held-out test split
  after model and hyperparameter choices are frozen.
- Record resolved configs, checkpoint hashes, cohort fingerprints, and random
  seeds for every result.
- Report physical field metrics for field-producing models and scalar metrics
  for maximum wind; report radii metrics separately by extraction source.

The existing comparison runner and evaluator remain in `scripts/` as the basis
for the new run. They should not update archived result pages.
