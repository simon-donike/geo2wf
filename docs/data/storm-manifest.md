# StormSense case-study manifest

This is the dashboard's three-storm case-study manifest, not the model-training
corpus. It contains dense geostationary observations for `AL082025`, `EP112025`,
and `EP182023`, plus IBTrACS intensity, available model outputs, sparse SAR
matches, and paths to image overlays.

[Browse the full training corpus](full-dataset.md){ .md-button .md-button--primary }

[Download the case-study CSV](../explorer/storm-data.csv){ .md-button download }
[View the source JSON](../explorer/storm-data.json){ .md-button }
[Open StormSense](../explorer/dashboard.html){ .md-button }

## Browse observations

Search, sort, and page through the main observation fields below. The download
contains the complete flattened dashboard schema for its three storms.

<div
  class="csv-table-viewer"
  data-csv-source="../../explorer/storm-data.csv"
  data-csv-columns="storm_id,storm_name,time,category,ibtracs_msw,vit_prediction.max,unet_prediction.max,unet_mlp_prediction.max,sar.max"
  data-csv-labels="Storm ID|Storm name|Time|Category|IBTrACS max m/s|ViT max m/s|UNet max m/s|UNet+MLP max m/s|SAR max m/s"
  data-csv-empty="—"
>
  <p class="csv-table-viewer__status">Loading observation manifest…</p>
</div>

## CSV shape

Nested JSON objects use dotted column names. Array-valued fields such as
overlay bounds and the available-model list remain compact JSON values inside
their CSV cells. Global display configuration, NWP series, PMW observations,
and forecast bundles remain in the source JSON because they do not map to one
row per geostationary observation.

The CSV is regenerated alongside `storm-data.json` by:

```bash
uv run python scripts/export_storm_explorer_data.py
```
