---
hide:
  - toc
---

# StormSense

StormSense follows complete tropical-cyclone tracks and compares model-derived wind diagnostics with matched SAR observations.

[Open StormSense](explorer/dashboard.html){ .md-button .md-button--primary }

Solid teal lines show the selected ViT, U-Net, U-Net+MLP, or diffusion inference series. Gray dots mark matched SAR observations; dashed gray segments only connect these sparse acquisitions. Optional NWP curves use the forecast series stored in the storm manifest.

The dashboard's U-Net+MLP series is the separate single-field correction model. It receives a frozen U-Net field and current metadata, then predicts a signed correction to maximum wind. Spatial diagnostics remain those of the upstream U-Net field. This product is distinct from the jointly trained bottleneck U-Net+MLP benchmark.

StormSense is plain HTML, CSS, JavaScript, and compact JSON. It deploys with the documentation and does not require a runtime server or API key.

For model behavior, start with [Two-stage baseline + diffusion](models/two-stage.md). For data provenance and real raster examples, see [Model inputs and training targets](data/index.md).
