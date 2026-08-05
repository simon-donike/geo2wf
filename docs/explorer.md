---
hide:
  - toc
---

# StormSense

StormSense follows complete tropical-cyclone tracks and compares model-derived wind diagnostics with matched SAR observations.

[Open StormSense](explorer/dashboard.html){ .md-button .md-button--primary }

Solid teal lines show the selected ViT, UNet, UNet+MLP, or Diffusion inference series; a model remains marked pending until its inference folder contains results. Gray dots mark matched SAR observations, and dashed gray segments only connect those sparse acquisitions visually. The optional NWP overlay reads the forecast series stored for each manifest storm.

UNet+MLP is the scalar-only intensity product: it takes one frozen U-Net wind field plus current-time metadata, predicts a signed maximum-wind correction, and derives TD through C5 from the corrected wind. It supplies the corrected Maximum wind series in StormSense; when selected, the remaining spatial-diagnostic charts use values calculated from its upstream frozen U-Net wind-field output.

StormSense is plain HTML, CSS, JavaScript, and compact JSON. It deploys with the documentation and does not require a runtime server or API key.

For model behavior, start with [Two-stage baseline + diffusion](models/two-stage.md). For data provenance and real raster examples, see [Model inputs and training targets](data/index.md).
