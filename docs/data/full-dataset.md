# Full dataset

This page exposes the complete observation-level StormSense dataset as a
searchable, sortable table. It contains one row per geostationary observation;
nested manifest fields are flattened into dotted column names. Use horizontal
scrolling to inspect the full schema, or download the CSV for analysis.

[Download the full CSV](../explorer/storm-data.csv){ .md-button .md-button--primary download }
[Download the source JSON](../explorer/storm-data.json){ .md-button download }
[Open StormSense](../explorer.md){ .md-button }

!!! note "What is represented"
    Blank cells mean that a measurement, match, or model output is unavailable
    for that observation. Array-valued fields such as map bounds and model lists
    remain compact JSON values in the CSV. The source JSON preserves the nested
    storm, observation, PMW, NWP, and forecast structures used by StormSense.

<div
  class="csv-table-viewer csv-table-viewer--full"
  data-csv-source="../../explorer/storm-data.csv"
  data-csv-empty="—"
>
  <p class="csv-table-viewer__status">Loading the full dataset…</p>
</div>

## Reproducibility

The checked-in CSV is generated from the checked-in JSON manifest by
`scripts/export_storm_explorer_data.py`. The exporter uses deterministic column
ordering, so schema changes are visible in version control. The website serves
this file directly; no browser-side transformation changes its values.
