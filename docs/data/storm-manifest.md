# StormSense case-study manifest

This is the dashboard's three-storm case-study manifest, not the model-training
corpus. It contains dense geostationary observations for `AL082025`, `EP112025`,
and `EP182023`, plus IBTrACS intensity, sparse SAR matches, and paths to image
overlays. Model predictions remain available in the dashboard and source JSON,
but are intentionally excluded from the CSV.

[Browse the full training corpus](full-dataset.md){ .md-button .md-button--primary }

[Download the case-study CSV](../explorer/storm-data.csv){ .md-button download }
[View the source JSON](../explorer/storm-data.json){ .md-button }
[Open StormSense](../explorer/dashboard.html){ .md-button }

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
