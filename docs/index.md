---
hide:
  - navigation
  - toc
---

<div class="geo-hero" markdown>
<span class="geo-eyebrow">:material-weather-hurricane: Tropical cyclone reconstruction</span>

# See the storm. <span>Recover the wind.</span>

geo2wf is a deliberately focused research baseline that turns colocated geostationary satellite imagery into SAR-like tropical-cyclone wind fields—with conditional pixel diffusion, an ERA5-anchored control, and a data pipeline you can reason about end to end.

<div class="geo-actions" markdown>
[Run the first experiment :material-arrow-right:](getting-started/first-experiment.md){ .md-button .md-button--primary }
[Understand the system](concepts/architecture.md){ .md-button }
</div>
</div>

<div class="geo-stats">
  <div class="geo-stat"><strong>4 / 10</strong><span>common GEO band options</span></div>
  <div class="geo-stat"><strong>256²</strong><span>default colocated grid</span></div>
  <div class="geo-stat"><strong>2 paths</strong><span>diffusion + residual control</span></div>
  <div class="geo-stat"><strong>DDPM / DDIM</strong><span>ancestral or fast sampling</span></div>
</div>

## One question, kept intentionally clear

> Given a geostationary image crop of a tropical cyclone, can a conditional model generate a plausible SAR-like wind field on the same grid?

The repository is the minimum-complexity counterpart to a larger multi-source reconstruction system. It keeps the useful scientific pieces—real observation pairing, geospatial rasters, validity masks, physically meaningful metrics, experiment configs—and removes heterogeneous occurrence sets, temporal windows, and heavy orchestration.

<div class="geo-flow" markdown>
<div markdown><span class="geo-kicker">Observe</span><br>GOES ABI or Himawari AHI infrared/water-vapor bands</div>
<div class="arrow">→</div>
<div markdown><span class="geo-kicker">Condition</span><br>Normalize, align, mask, and optionally append ERA5 context</div>
<div class="arrow">→</div>
<div markdown><span class="geo-kicker">Reconstruct</span><br>Denoise a SAR wind field or learn a correction to ERA5</div>
<div class="arrow">→</div>
<div markdown><span class="geo-kicker">Evaluate</span><br>Image quality, physical error, eye and radial structure</div>
</div>

## Explore by goal

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } **Get a run moving**

    ---

    Install with UV, export two samples, inspect a batch, and launch a one-batch smoke run.

    [:octicons-arrow-right-24: First experiment](getting-started/first-experiment.md)

-   :material-database-arrow-right-outline:{ .lg .middle } **Understand the data**

    ---

    Follow observations from the source manifest into raw GeoTIFFs, split manifests, train statistics, tensors, and masks.

    [:octicons-arrow-right-24: Data pipeline](data/index.md)

-   :material-creation-outline:{ .lg .middle } **Understand diffusion**

    ---

    See what is noised, what the U-Net predicts, how conditioning enters, and why DDIM changes sampling cost.

    [:octicons-arrow-right-24: Conditional diffusion](models/conditional-diffusion.md)

-   :material-tune-variant:{ .lg .middle } **Choose a configuration**

    ---

    Compare 4-band, 10-band, ERA5, PMW pretraining, residual-control, and multi-GPU experiments.

    [:octicons-arrow-right-24: Experiment matrix](experiments/index.md)

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **Judge storm structure**

    ---

    Go beyond PSNR: measure wind error in m/s, the eye, inner core, radius of maximum wind, and skill against ERA5.

    [:octicons-arrow-right-24: Evaluation](experiments/evaluation.md)

-   :material-map-search-outline:{ .lg .middle } **Find a file or command**

    ---

    Use the project map, configuration key reference, environment variables, and troubleshooting guide.

    [:octicons-arrow-right-24: Reference](reference/index.md)

</div>

## The data, at a glance

[![Five random GEO–SAR training pairs](assets/images/geo-sar-random-pairs.png)](experiments/visual-examples.md)

<p class="geo-caption">GEO false-color context, paired SAR wind-speed target, and valid-area footprint. Select the image to inspect it.</p>

!!! info "Know the boundary"
    geo2wf does **paired image-to-image reconstruction**. It does not currently model arbitrary source sets, observation-time offsets, full storm tracks, multi-temporal windows, or learned source metadata. Those are deliberate non-goals until the focused baseline proves useful.

## Recommended reading path

1. [Install the environment](getting-started/installation.md).
2. Learn the [reconstruction problem](concepts/problem.md) and [system architecture](concepts/architecture.md).
3. Follow the [data pipeline](data/index.md), including [normalization and masks](data/normalization.md).
4. Compare the [conditional diffusion model](models/conditional-diffusion.md) with the [ERA5 residual baseline](models/era5-residual.md).
5. Pick an [experiment](experiments/index.md), then [train](experiments/training.md) and [evaluate](experiments/evaluation.md).
