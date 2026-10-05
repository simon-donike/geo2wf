# Paper case-study manifest

This is the earlier viewer's fixed three-storm case-study manifest, separate
from the model-training corpus and the current StormSense rolling archive.
It contains dense geostationary observations for Humberto (`AL082025`),
Kiko (`EP112025`), and Otis (`EP182023`), plus IBTrACS intensity, sparse SAR
matches, and paths to image overlays. Model predictions remain in the source
JSON and [paper results](../results.md); they are excluded from the CSV.

[Browse the scientific dataset](index.md){ .md-button .md-button--primary }

[Download the case-study CSV](../explorer/storm-data.csv){ .md-button download }
[View the source JSON](../explorer/storm-data.json){ .md-button }
[Open StormSense](https://stormsense.hyperalislabs.com/){ .md-button }

## Browse observations

Search, sort, and page through the main observation fields below. The download
contains observation and source-data fields only; it does not include model
predictions or performance metrics.

<div
  class="csv-table-viewer"
  data-csv-source="../../explorer/storm-data.csv"
  data-csv-columns="storm_id,storm_name,time,lat,lon,category,ibtracs_msw,sar.max,sar_dt_minutes"
  data-csv-labels="Storm ID|Storm name|Time|Latitude|Longitude|Category|IBTrACS max m/s|SAR max m/s|SAR offset minutes"
  data-csv-empty="—"
>
  <p class="csv-table-viewer__status">Loading observation manifest…</p>
</div>

## CSV shape

Nested data objects use dotted column names. Array-valued fields such as overlay
bounds remain compact JSON values inside their CSV cells. Global display
configuration, model metadata and predictions, NWP series, PMW observations,
and forecast bundles remain in the source JSON.

The data-only CSV is regenerated alongside `storm-data.json` by:

```bash
uv run python scripts/export_storm_explorer_data.py
```
