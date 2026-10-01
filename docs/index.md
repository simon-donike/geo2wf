---
hide:
  - toc
---

<div class="geo-intro" markdown>
<span class="geo-eyebrow">Tropical-cyclone wind reconstruction</span>

# Wind fields and intensity from geostationary imagery

geo2wf estimates tropical-cyclone surface wind fields, maximum sustained wind,
and wind radii from geostationary satellite imagery. These are the companion
docs for the accepted paper *Nowcasting of Tropical Cyclone Wind Fields and
Intensity from Geostationary Imagery*.

The paper asks whether learning spatial wind structure helps estimate storm
intensity, especially during rapid intensification. A shared U-Net encoder
supports both SAR-supervised field reconstruction and a latent MLP for scalar
intensity and radii. SAR is needed for training supervision; inference uses
GEO imagery and deterministic context, with ERA5 as an optional input.

<div class="geo-actions" markdown>
[Paper reasoning](concepts/problem.md){ .md-button .md-button--primary }
[Results](results.md){ .md-button }
[Dataset on Hugging Face](https://huggingface.co/datasets/simon-donike/geo2wf-data){ .md-button }
[Models on Hugging Face](https://huggingface.co/simon-donike/geo2wf-models){ .md-button }
[Open StormSense](explorer/dashboard.html){ .md-button }
</div>
</div>

## Use the project

- [Get started](getting-started/installation.md): install the package and locate data and model files.
- [Dataset](data/index.md): understand the published catalog, inputs, targets, and local loading format.
- [Models](models/index.md): choose a field model, joint model, scalar correction, or forecast.
- [Train and evaluate](experiments/training.md): use the retained presets and checkpoint workflows.

[![GEO, ERA5, SAR, and mask example](assets/images/data-example-target.webp)](data/index.md)

<p class="geo-caption">An exported training example: ERA5 context, a sparse SAR wind retrieval, and its validity mask. Supervised field errors are measured only where SAR is observed.</p>

## Scope

Field models reconstruct the observation time. The separate scalar forecast
predicts six-hour intensity change. [StormSense](explorer.md) provides
retrospective case studies of Humberto, Kiko, and Otis. The
[paper discussion](concepts/problem.md) explains the observational limits and
why this is a research workflow rather than a validated operational product.
