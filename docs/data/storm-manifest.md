# Storm observation manifest

The StormSense explorer manifest is available as a flat CSV with one row per
geostationary observation. It includes storm metadata, IBTrACS intensity,
predictions from every available model, SAR-match metrics, uncertainty
statistics, and paths to the corresponding image overlays.

[Download the complete CSV](../explorer/storm-data.csv){ .md-button .md-button--primary download }
[View the source JSON](../explorer/storm-data.json){ .md-button }
[Open StormSense](../explorer/dashboard.html){ .md-button }

## Browse observations

Search, sort, and page through the main observation fields below. The download
contains the complete flattened schema, including all model metrics and
diffusion uncertainty columns.

<div
  class="csv-table-viewer"
  data-csv-source="../../explorer/storm-data.csv"
  data-csv-columns="storm_id,storm_name,time,category,ibtracs_msw,vit_prediction.max,unet_prediction.max,unet_mlp_prediction.max,diffusion_prediction.max,sar.max"
  data-csv-labels="Storm ID|Storm name|Time|Category|IBTrACS max m/s|ViT max m/s|UNet max m/s|UNet+MLP max m/s|Diffusion max m/s|SAR max m/s"
  data-csv-empty="—"
>
  <p class="csv-table-viewer__status">Loading observation manifest…</p>
</div>

## CSV shape

Nested JSON objects use dotted column names, such as
`diffusion_prediction.uncertainty.metrics.max.p90`. Array-valued fields such as
overlay bounds and the available-model list remain compact JSON values inside
their CSV cells. Global display configuration, NWP series, PMW observations,
and forecast bundles remain in the source JSON because they do not map to one
row per geostationary observation.

The CSV is regenerated alongside `storm-data.json` by:

```bash
uv run python scripts/export_storm_explorer_data.py
```
