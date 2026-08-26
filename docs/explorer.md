---
hide:
  - toc
---

# StormSense

StormSense is a retrospective case-study viewer. It follows complete
tropical-cyclone tracks and compares model-derived wind diagnostics with
matched SAR retrievals and best-track records. It is not an operational storm
forecast product.

[Open StormSense](explorer/dashboard.html){ .md-button .md-button--primary }

## Nowcast view

Solid teal lines show the selected ViT, U-Net, or U-Net+MLP inference
series at observation time. Gray dots mark matched SAR retrievals; dashed gray
segments only interpolate visually between those sparse acquisitions. The
cream intensity reference is IBTrACS maximum sustained wind. Optional AIFS,
AIFS2, ERA5, GFS, GraphCast, and Pangu curves are precomputed full-track series
stored in the generated manifest.

The dashboard's U-Net+MLP series is the separate single-field correction model. It receives a frozen U-Net field and current metadata, then predicts a signed correction to maximum wind. Spatial diagnostics remain those of the upstream U-Net field. This product is distinct from the jointly trained bottleneck U-Net+MLP benchmark.

The ViT series is an imported inference artifact under `inference/inf_vit`, not
a maintained training model in this package. The exporter uses its observation
IDs to define the displayed case-study cohort and aligns other artifacts to
those IDs. The dashboard label therefore
does not imply that a reproducible ViT architecture or training config lives in
this repository.

## Forecast view

Forecast mode shows deterministic retrospective results at a fixed +12 h lead.
Map time is issue time; the highlighted track point is the valid time. Two
artifact families can be present:

- **MLP:** the maintained +6 h scalar model applied recursively twice from
  current and −6/−12 h IBTrACS anchors. It predicts maximum wind only.
- **ConvLSTM:** an external processor artifact using a 12-frame GEO/PMW context
  and a 12-hour lead. Its inference configuration is retained with the
  artifacts, but its model implementation and training workflow are outside
  this package.

These retrospective forecasts can use best-track context unavailable in real
time. They should not be read as an operational skill claim.

## Delivery and provenance

StormSense is plain HTML, CSS, JavaScript, and generated JSON/image assets. It
deploys with the documentation and does not require a runtime server or API
key. The [storm observation manifest](data/storm-manifest.md) exposes the same
records as downloadable JSON and a flat CSV table.

For model behavior, start with the [model overview](models/index.md) and
[ERA5-residual U-Net](models/era5-residual.md). For data provenance and
real raster examples, see [Model inputs and training targets](data/index.md).
