---
hide:
  - toc
---

# Storm dashboard

The dashboard follows complete tropical-cyclone tracks and compares model-derived wind diagnostics with matched SAR observations.

[Open the dashboard](explorer/dashboard.html){ .md-button .md-button--primary }

Solid red lines show the selected model. Gray dots mark matched SAR observations; dashed gray segments only connect those sparse acquisitions visually. The model selector changes graph series, while the map and summary cards retain their documented reference series.

The dashboard is plain HTML, CSS, JavaScript, and compact JSON. It deploys with the documentation and does not require a runtime server or API key.

For model behavior, start with [Two-stage baseline + diffusion](models/two-stage.md). For data provenance and real raster examples, see [Model inputs and training targets](data/index.md).
