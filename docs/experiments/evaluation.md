# Evaluation

Compare the same sample IDs, split membership, target definitions, masks, and
aggregation. Current grouped configs keep test out of training, but historical
checkpoints retain their original provenance. A downstream cache split cannot
remove exposure inherited from its producer model.

Use the [evaluation commands](../reference/commands.md#evaluate-local-checkpoints)
for local checkpoints and [paper results](../results.md) for the retained
published values.

## Field metrics

Field comparisons use finite pixels inside the observed SAR footprint. ERA5
skill uses the same common-valid pixels for prediction, target, and baseline.

| Metric | Definition |
|---|---|
| MAE / L1 | Pooled valid-pixel absolute error in m/s |
| RMSE | Square root of pooled squared error in m/s |
| Bias | Prediction minus target in m/s |
| PSNR | \(20\log_{10}(79.8\,\mathrm{m/s}/\mathrm{RMSE})\) |
| SSIM | Mean scene similarity over complete valid 7 × 7 windows, with wind clipped to 0.2–80.0 m/s |
| High-wind MAE | Error where target wind exceeds the configured threshold; default 17 m/s |
| Skill vs ERA5 | \(1-\mathrm{MAE}_{model}/\mathrm{MAE}_{ERA5}\) on identical support |

Storm diagnostics use physical distances derived from raster bounds and
latitude. They include eye MAE within 25 km, inner-core MAE within 100 km,
10 km annular-mean profiles out to 200 km, profile-derived RMW, and eye/eyewall
contrast. Eye-center displacement compares eligible smoothed minima and is
unavailable when SAR coverage or storm-structure checks fail; unavailable
values are not zeros.

## Scalar intensity and radii

Intensity heads are evaluated against continuous IBTrACS `USA_WIND` in m/s.
Report global and storm-macro errors separately. A raw field maximum and
best-track sustained wind are different quantities, which motivates the
paper's scalar heads.

RMW is the radius of maximum wind. R34/R50/R64 summarize wind extent at 34,
50, and 64 kt. The scalar targets use equivalent-area radii from available
best-track quadrant records. Field-derived radii use the predicted spatial
field and its valid physical domain. Report **direct-head** and
**image-derived** estimates separately, with valid-target counts.

## Rapid intensification

At observation time \(t\), RI is defined by

\[
V(t)-V(t-24\,\mathrm{h})\ge30\,\mathrm{kt}.
\]

Both reference winds must be available under the workflow's interpolation
rules. Missing history excludes a sample from RI evaluation. RI is a subset
of the regular cohort, not an RI-only training filter. Report its observation
and storm counts alongside errors.

## Aggregation and interpretation

Pixel metrics pool error sums and valid-pixel counts before reduction. Storm
structure metrics aggregate available sample values; storm-macro scalar
metrics weight storms equally. Temporally dense samples within a storm are
correlated. Case-study curves may be smoothed for display, but their reported
errors use native unsmoothed predictions.

A complete comparison records checkpoint and dataset revisions, eligible
samples/storms, metric availability, and training membership. The
[dataset release](../data/index.md) distinguishes source inventories from
actual evaluation cohorts. Do not combine the architecture test table, latent
validation results, and dense case studies into one benchmark ranking.
