# Final results

This page contains only the final held-out tests requested for the streamlined
experiment set:

1. three instantaneous models, each tested with and without ERA5; and
2. their maximum-wind nowcasts on three validation storms; and
3. the joint U-Net/latent-MLP radii ablation.

## Three-model ERA5 ablation

All six cells use the same storm-disjoint cohort and scalar target. Checkpoints
must be selected from validation data before the held-out test split is run.
The old validation benchmark is not used to fill this table.

| Model | ERA5 | Maximum-wind MAE (m/s) | Maximum-wind RMSE (m/s) | Maximum-wind bias (m/s) | Field MAE (m/s) | Test samples | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Raw field U-Net | With | — | — | — | — | — | Pending new run |
| Raw field U-Net | Without | — | — | — | — | — | Pending new run |
| U-Net + scalar correction | With | — | — | — | N/A | — | Pending new run |
| U-Net + scalar correction | Without | — | — | — | N/A | — | Pending new run |
| Joint U-Net + bottleneck MLP | With | — | — | — | — | — | Pending new run |
| Joint U-Net + bottleneck MLP | Without | — | — | — | — | — | Pending new run |

### Rapid-intensification phases

The same scalar metrics are reported on the subset where IBTrACS maximum wind
increased by at least 30 kt during the preceding 24 hours.

| Model | ERA5 | RI MAE (m/s) | RI RMSE (m/s) | RI bias (m/s) | RI samples | Status |
|---|---|---:|---:|---:|---:|---|
| Raw field U-Net | With | — | — | — | — | Pending new run |
| Raw field U-Net | Without | — | — | — | — | Pending new run |
| U-Net + scalar correction | With | — | — | — | — | Pending new run |
| U-Net + scalar correction | Without | — | — | — | — | Pending new run |
| Joint U-Net + bottleneck MLP | With | — | — | — | — | Pending new run |
| Joint U-Net + bottleneck MLP | Without | — | — | — | — | Pending new run |

<!-- three-storm-nowcast:start -->
## Three-storm maximum-wind nowcasts

Pending the six new comparison checkpoints. The paper figure will contain one
panel each for Humberto 2025, Kiko 2025, and Otis 2023, with IBTrACS and every
model/conditioning series on the same combined figure. The max-wind-only and
radii-supervised joint ablation arms will also be included when supplied to the
inference command. The generated table and CSV include the same MAE, RMSE, and
bias metrics for all observations and for RI phases only.
<!-- three-storm-nowcast:end -->

## Joint U-Net/latent-MLP radii ablation

These are held-out test results from two seed-42 runs on the same
ERA5-conditioned cohort: 568 training, 159 validation, and 139 test samples.
Checkpoints were selected using validation loss before the test split was
evaluated once.

### Maximum wind

| Training objective | MLP MAE (m/s) | MLP RMSE (m/s) | MLP bias (m/s) | Samples |
|---|---:|---:|---:|---:|
| Maximum wind only | 6.471 | 8.181 | 1.133 | 139 |
| Maximum wind + radii | 5.533 | 7.313 | 0.132 | 139 |

![Latent MLP maximum-wind MAE](assets/images/latent-structure/maximum-wind-mae.png)

Adding radii supervision coincided with a 0.938 m/s lower maximum-wind MAE in
this single seeded comparison. This is an observed paired-run result, not a
multi-seed uncertainty estimate.

### 2D wind-field reconstruction

| Training objective | Field MAE (m/s) | Field RMSE (m/s) | Field bias (m/s) | Scenes |
|---|---:|---:|---:|---:|
| Maximum wind only | 2.638 | 3.765 | 0.106 | 139 |
| Maximum wind + radii | 2.565 | 3.684 | -0.185 | 139 |

### Radii from the latent head and 2D field

Both columns evaluate the radii-supervised run. The latent MLP predicts radii
directly, while the field values are diagnosed from the reconstructed U-Net
image. Missing targets are masked; field counts can be smaller when the target
lies outside the image's complete circular domain.

| Radius | Latent MLP MAE (km) | MLP n | 2D field MAE (km) | Field n |
|---|---:|---:|---:|---:|
| RMW | 22.60 | 138 | 34.89 | 132 |
| R34 | 44.97 | 116 | 51.21 | 53 |
| R50 | 32.69 | 72 | 35.12 | 58 |
| R64 | 20.38 | 48 | 29.92 | 46 |

![Radius MAE by extraction source](assets/images/latent-structure/radius-mae-by-source.png)

[Full radii-ablation provenance](experiments/latent-structure-results.md){ .md-button .md-button--primary }
[Download radii-ablation CSV](assets/data/latent-structure/summary.csv){ .md-button }
[Download radii-ablation JSON](assets/data/latent-structure/results.json){ .md-button }
