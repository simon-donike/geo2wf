---
hide:
  - toc
---

# StormSense

StormSense follows Humberto, Kiko, and Otis through their full tracks, comparing
model wind diagnostics with sparse SAR retrievals and retrospective IBTrACS
records. It is a case-study viewer, not an operational forecast product.

[Open StormSense](explorer/dashboard.html){ .md-button .md-button--primary }
[Browse observations](data/storm-manifest.md){ .md-button }
[Scientific data on Hugging Face](https://huggingface.co/datasets/simon-donike/geo2wf-data){ .md-button }

## Nowcasts

The selected ViT, U-Net, or U-Net+MLP series is evaluated at observation time.
SAR points are sparse acquisitions; connecting segments only interpolate
visually. IBTrACS provides the intensity reference. Optional numerical-model
and ERA5 curves are precomputed comparison series.

**U-Net+MLP** here means the [post-hoc intensity correction](models/intensity-correction.md)
applied to a frozen field. Spatial diagnostics remain those of that field.
The paper's jointly trained latent MLP is a separate model, shown in the
[paper results](results.md). ViT outputs are imported artifacts; its training
implementation is outside this package.

## Forecasts

Forecast mode shows retrospective results at a fixed +12 h lead. Map time is
the issue time; the highlighted point is valid 12 hours later.

- **MLP:** the retained [six-hour scalar model](models/intensity-forecast.md)
  applied twice, starting from current and −6/−12 h IBTrACS anchors.
- **ConvLSTM:** an external artifact using a 12-frame GEO/PMW context and a
  12-hour lead; its model implementation is outside this package.

Best-track context can be unavailable in real time, so these layers do not
establish operational forecast skill.

## Data and delivery

The frontend ships with the docs. Generated JSON and image overlays can be
served as immutable R2 data releases. Those browser assets are separate from
the full [scientific dataset](data/index.md) on Hugging Face. The
[case-study manifest](data/storm-manifest.md) provides a data-only CSV and
source JSON; model outputs remain in the JSON/dashboard.
