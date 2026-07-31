# The reconstruction problem

geo2wf asks whether colocated GEO imagery and environmental context can recover a plausible tropical-cyclone surface wind field where SAR provides sparse supervision.

## Observations, context, and target

Each sample is aligned to one 256 × 256 EPSG:4326 grid:

- **GEO observation:** ten GOES ABI or Himawari AHI infrared / water-vapor channels;
- **ERA5 context:** seven exported atmospheric/surface fields plus derived wind speed and relative vorticity;
- **derived context:** distance to the IBTrACS center and three solar-time fields;
- **SAR target:** one near-surface wind-speed channel in m/s; and
- **validity masks:** explicit support for the condition, ERA5 anchor, and sparse SAR swath.

See [Model inputs and training targets](../data/index.md) for real examples and exact channel assembly.

GEO and SAR do not observe the same physical quantity, so this is not ordinary super-resolution. The model infers surface-wind structure compatible with cloud-top, water-vapor, environmental, and storm-relative context.

## Main two-stage formulation

Stage 1 makes one deterministic commitment:

\[
\hat v_{\mathrm{base}} =
v_{\mathrm{ERA5}} + f_\theta(x_{\mathrm{GEO}}, x_{\mathrm{ERA5}}, x_{\mathrm{derived}}, m)
\]

Stage 2 models the distribution of signed SAR corrections around that frozen field:

\[
p_\phi\left(
v_{\mathrm{SAR}} - \hat v_{\mathrm{base}}
\mid
x_{\mathrm{GEO}}, x_{\mathrm{ERA5}}, x_{\mathrm{derived}},
\hat v_{\mathrm{base}}, m
\right)
\]

This separates the broad, stable reconstruction from probabilistic fine-structure refinement. [Read the full two-stage workflow.](../models/two-stage.md)

## Scientific constraints reflected in code

Sparse SAR coverage
: Losses and metrics use `target_mask`. A weak off-swath anchor can constrain corrections where ERA5 is valid without calling those pixels SAR observations.

Physical scale
: The dataset retains `target_physical` and reversible normalization parameters. The deterministic model predicts in m/s; diffusion residuals are inverted and added in m/s.

Storm geometry
: Manifests carry IBTrACS center coordinates and raster bounds. The dataset derives a distance input, while evaluation uses physical coordinates for eye, inner-core, radial-profile, and radius-of-maximum-wind metrics.

Illumination
: Pixelwise local-solar-time sine/cosine and solar zenith help interpret daylight-sensitive GEO structure.

Reproducible sampling
: Validation noise is derived from the global validation seed and `sample_id`, so the same sample starts from the same latent across epochs.

Vector-aware augmentation
: Flips transform ERA5 wind components and vorticity according to their physical parity instead of treating every channel as a generic scalar image.

## Explicit non-goals

The current system does not represent arbitrary heterogeneous observation sets, multi-temporal windows, learned availability policies, or complete tracks as model inputs. Source download is also outside `prepare_data()`; export is an explicit preprocessing step.
