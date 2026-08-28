# Full training corpus

This is the complete paired GEO–ERA5–SAR corpus manifest used by the maintained
wind-field training workflow. It contains **1,205 samples from 176 storms** and
preserves the storm-disjoint training, validation, and test assignment for each
sample. Use horizontal scrolling to inspect all 103 columns, or download the
CSV for analysis.

[Download the full corpus CSV](../assets/data/training-corpus-manifest.csv){ .md-button .md-button--primary download }
[Read the dataset contract](dataset-contract.md){ .md-button }

## Split summary

| Split | Samples | Storms | Role |
|---|---:|---:|---|
| Train | 798 | 111 | Parameter fitting and training-only normalization statistics |
| Validation | 212 | 31 | Model selection and validation metrics |
| Test | 195 | 34 | Held-out final evaluation |
| **Total** | **1,205** | **176** | Complete paired corpus |

## Experiment-ready split sizes

The archive summary above describes the published corpus manifest. The active
experiments use the newer common-cohort export and apply their target-eligibility
rules before constructing a data loader. The resulting counts are:

| Experiment family / training stage | Train samples (storms) | Validation samples (storms) | Test samples (storms) | Total samples (storms) |
|---|---:|---:|---:|---:|
| Field U-Net, joint U-Net + MLP, and latent-structure matrix | 568 (109) | 159 (33) | 139 (33) | 866 (175) |
| Single-field intensity correction | 568 (109) | 159 (33) | 139 (33) | 866 (175) |
| Six-hour forecast: historical pretraining | 73,522 (1,783) | 15,289 (405) | — | 88,811 (2,188) |
| Six-hour forecast: matched fine-tuning | 229 (66) | 79 (24) | 51 (18) | 359 (108) |

### Rapid-intensification subset

For the instantaneous experiments, an RI sample has an IBTrACS intensity
increase of at least 30 kt over the preceding 24 hours. Both the current and
24-hour-prior winds must be available under the experiment's interpolation
rules. “RI storms” counts distinct storms containing at least one such sample;
it is not the number of RI episodes.

!!! important "RI is an evaluation subset, not a training filter"
    The models are fitted on the complete 568-sample training split, including
    both RI and non-RI observations. The 55 training rows labelled RI below are
    reported only to describe the cohort; they do not form a separate training
    set. Likewise, `val_ri/*` metrics are computed by selecting the 24 RI rows
    from the same 159-sample validation split. The ordinary `val/*` metrics
    continue to use all 159 validation samples.

| Experiment cohort | Train RI samples (storms) | Validation RI samples (storms) | Test RI samples (storms) | Total RI samples (storms) |
|---|---:|---:|---:|---:|
| Instantaneous paired cohort | 55 (32) | 24 (14) | 13 (7) | 92 (53) |
| Single-field correction cache | 55 (32) | 24 (14) | 13 (7) | 92 (53) |

The six-hour forecast caches do not attach this split-wide 24-hour RI flag to
every training row, so reporting analogous counts from those manifests would
mix definitions. Forecast validation instead has three explicitly configured
RI rollout case studies (`WP282025`, `WP112024`, and `AL092024`); those are
diagnostic storms, not an RI-only training subset.

The instantaneous rows intentionally have identical counts. The ERA5/no-ERA5,
SAR/no-SAR, wind-only, and wind-plus-radii experiments retain the same
storm-disjoint cohort so their comparisons are paired. “No ERA5” means that
ERA5 is withheld from the model, not that samples lacking ERA5 are admitted.
The correction cache contains one eligible frozen-field record for each sample
in that same cohort.

Counts are dataset lengths after filtering, not minibatch counts or the number
of labelled radius values. A sample can remain eligible when an individual
structure target is missing because each radius has its own loss mask. The
historical forecast workflow defines pretraining train/validation splits only;
its configured evaluation alias reuses `pretrain_val`, so it is not reported
as an independent test set here.

These values come from the resolved active experiment data modules backed by
`data/geotiff/geo_sar_10bands_era5_v2_pmw`,
`data/unet_intensity_structure_v3`, and `data/intensity_forecast`. They should
be updated whenever those versioned caches or eligibility rules change.

Each row identifies the paired GEO condition, ERA5 context, and SAR target;
their observation IDs, timestamps, sensors, channels, grid geometry, and time
offsets; storm-relative metadata; SAR and ERA5 field summaries; common-valid
comparisons; and quality/structure flags. Paths are relative to the exported
dataset root. Blank cells represent unavailable or inapplicable measurements.

!!! important "Manifest, not raster archive"
    The CSV is the complete metadata and split index. The underlying multiband
    GeoTIFF inputs and targets are much larger and are not bundled into the
    documentation site.

<div
  class="csv-table-viewer csv-table-viewer--full"
  data-csv-source="../../assets/data/training-corpus-manifest.csv"
  data-csv-empty="—"
>
  <p class="csv-table-viewer__status">Loading the full training corpus…</p>
</div>

## Reproducibility

The website copy is published verbatim from
`data/geotiff/geo_sar_10bands_era5/manifest.csv` after validating its required
columns, split labels, and unique sample IDs:

```bash
uv run python scripts/export_training_corpus_manifest.py
```

This corpus table is distinct from the [three-storm StormSense case-study
manifest](storm-manifest.md), which contains dense dashboard observations
rather than the paired training examples. Its downloadable CSV is data-only;
model outputs remain in the dashboard JSON.
