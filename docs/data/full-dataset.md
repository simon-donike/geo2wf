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
manifest](storm-manifest.md), which contains dense dashboard observations and
model outputs rather than the paired training examples.
