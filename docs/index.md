---
hide:
  - toc
---

<div class="geo-intro" markdown>
<span class="geo-eyebrow">Tropical-cyclone wind reconstruction</span>

# Two-stage wind-field reconstruction

geo2wf reconstructs surface wind fields from geostationary satellite imagery and optional ERA5 context. The principal reconstruction system uses a deterministic baseline followed by diffusion of the remaining signed error.

<div class="geo-actions" markdown>
[Read the two-stage workflow](models/two-stage.md){ .md-button .md-button--primary }
[Understand the data](data/index.md){ .md-button }
[Open StormSense](explorer/dashboard.html){ .md-button }
</div>
</div>

## Reconstruction workflow

<div class="stage-flow">
  <div class="stage-card">
    <span class="geo-kicker">Stage 1</span>
    <strong>Deterministic baseline</strong>
    <p>GEO, ERA5, storm-relative geometry, solar context, and validity masks produce one dense wind field.</p>
    <code>baseline = ERA5 + learned correction</code>
  </div>
  <div class="stage-arrow">→</div>
  <div class="stage-card">
    <span class="geo-kicker">Stage 2</span>
    <strong>Residual diffusion</strong>
    <p>The Stage 1 checkpoint is frozen. Diffusion samples a signed SAR-minus-baseline residual and adds it back in physical m/s.</p>
    <code>wind sample = baseline + sampled residual</code>
  </div>
</div>

Stage 1 estimates the broad field. Stage 2 models unresolved eye, eyewall, gradient, and asymmetric structure without regenerating the complete field from noise. [See the equations, channel counts, and training sequence.](models/two-stage.md)

## What the models receive

The default two-stage configuration uses a 23-channel data condition:

- 10 GEO infrared and water-vapor bands;
- 9 ERA5 fields: seven exported variables plus derived 10 m wind speed and relative vorticity;
- 1 normalized distance-to-IBTrACS-center raster; and
- 3 solar-time fields.

Validity masks and an explicit ERA5 wind anchor are appended by the model. Stage 2 also receives the frozen Stage 1 field and its validity mask. SAR wind is the supervised target during training; it is not an inference-time input.

[![Real GEO, ERA5, SAR, and mask example](assets/images/data-example-target.webp)](data/index.md)

<p class="geo-caption">Real exported sample <code>WP232024_sar_geo_20241030095303_bb2c52ca</code>, rendered from the repository GeoTIFFs with Matplotlib. The data page shows every input family.</p>

## Read by task

<div class="quick-links">
  <a class="quick-link" href="models/two-stage/"><strong>Understand the model</strong><span>The main two-stage baseline + diffusion article.</span></a>
  <a class="quick-link" href="data/"><strong>Understand the inputs</strong><span>Real examples, channel lists, masks, and tensor assembly.</span></a>
  <a class="quick-link" href="getting-started/first-experiment/"><strong>Run an experiment</strong><span>Export a small batch and launch a smoke run.</span></a>
  <a class="quick-link" href="experiments/"><strong>Choose a preset</strong><span>Compare the stacked workflow with standalone controls.</span></a>
  <a class="quick-link" href="experiments/evaluation/"><strong>Evaluate structure</strong><span>Physical error, eye, inner core, radial profile, and RMW.</span></a>
  <a class="quick-link" href="reference/"><strong>Find a file or command</strong><span>Project map, configuration keys, and troubleshooting.</span></a>
</div>

## Scope

The wind-field models operate on paired rasters on a common grid and use a 192 × 192 center crop by default. Separate scalar models estimate current intensity and six-hour intensity change. The repository does not implement joint track and wind-field forecasting or arbitrary observation-set models.

Start with [the two-stage workflow](models/two-stage.md), then follow the [data inputs](data/index.md) into [training](experiments/training.md) and [evaluation](experiments/evaluation.md).
