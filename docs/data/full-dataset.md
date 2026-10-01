# Paired export inventory

This downloadable table records **1,205 paired GEO–ERA5–SAR samples from 176
storms** in the original `geo_sar_10bands_era5` export. It is one source
inventory within the broader [Hugging Face dataset](index.md), not the total
released dataset or a universal paper evaluation cohort.

[Download the paired-export CSV](../assets/data/training-corpus-manifest.csv){ .md-button .md-button--primary download }
[Browse the full dataset on Hugging Face](https://huggingface.co/datasets/simon-donike/geo2wf-data){ .md-button }

| Split | Samples | Storms |
|---|---:|---:|
| Train | 798 | 111 |
| Validation | 212 | 31 |
| Test | 195 | 34 |
| **Total** | **1,205** | **176** |

The 103 columns include raster paths, source IDs and times, channels, grid
geometry, storm metadata, coverage, and field summaries. Paths are relative to
the original export root; blank cells represent unavailable values. The table
contains metadata only. Model-specific eligibility and training membership are
recorded separately in the Hub release's `effective-cohorts/`.

<div
  class="csv-table-viewer csv-table-viewer--full"
  data-csv-source="../../assets/data/training-corpus-manifest.csv"
  data-csv-empty="—"
>
  <p class="csv-table-viewer__status">Loading the paired export inventory…</p>
</div>

For dense Humberto, Kiko, and Otis observations, use the
[StormSense case-study manifest](storm-manifest.md).
