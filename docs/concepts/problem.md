# The reconstruction problem

## Inputs and target

For each tropical-cyclone occurrence, geo2wf wants a spatially aligned pair:

- **Condition `x`** — a fixed `[C, H, W]` crop of GEO imagery from GOES ABI or Himawari AHI, optionally concatenated with ERA5 fields and always followed by a normalized distance-to-IBTrACS-center raster.
- **Target `y`** — a one-channel SAR near-surface wind-speed field, or a one-channel high-frequency PMW brightness-temperature proxy during pretraining.
- **Validity masks** — binary rasters distinguishing observations from padding, missing swath, and invalid pixels.

The minimal mapping is:

\[
x_{\mathrm{GEO}} \longrightarrow p_\theta(y_{\mathrm{SAR}} \mid x_{\mathrm{GEO}})
\]

This is not ordinary super-resolution. GEO and SAR observe different physical quantities with different sensors and footprints. The model must infer a wind-field structure that is compatible with cloud-top and water-vapor context, while the SAR target may cover only part of the shared grid.

## Why keep it small?

A larger system can encode PMW, IR, SAR, ERA5, tracks, coordinates, time offsets, availability, and source metadata. Those additions make attribution harder: when a run improves, it is less obvious which component helped. geo2wf starts with a paired tensor contract so data errors, model behavior, and metrics stay inspectable.

## Scientific constraints reflected in code

Sparse SAR coverage
: Losses and metrics use `target_mask`. ERA5 experiments can weakly fill unobserved target pixels without pretending they are SAR observations.

Physical scale
: The dataset retains `target_physical` and reversible normalization parameters. Evaluation maps predictions back to m/s.

Storm geometry
: Manifests carry IBTrACS center coordinates and raster bounds. The dataset turns them into a crop-normalized distance input, while metrics use the physical coordinates to measure the eye, inner core, radial profile, and radius of maximum wind.

Reproducible qualitative comparison
: Validation noise is derived from a hash of the global validation seed and `sample_id`. The same sample starts from the same latent noise across epochs.

Vector-aware augmentation
: Horizontal and vertical flips also transform ERA5 vector and vorticity channels correctly; they are not treated as generic scalar images.

## Explicit non-goals

The current baseline does not represent arbitrary sets of heterogeneous observations, multi-temporal context windows, source occurrence metadata, learned availability policies, or full storm tracks. It also does not download source data in `prepare_data()`. Export is an explicit, one-time preprocessing step.
