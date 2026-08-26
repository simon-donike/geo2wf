# Scientific problem and observations

geo2wf estimates tropical-cyclone surface wind fields from colocated
geostationary imagery and environmental context, using sparse synthetic-aperture
radar (SAR) wind retrievals as supervision. This is an **instantaneous
reconstruction** problem: it estimates conditions near the observation time. It
is not a track forecast. The separate scalar forecast model is described under
[Six-hour intensity forecast](../models/intensity-forecast.md).

## Why the inverse problem is difficult

GEO and SAR do not observe the same physical quantity. GOES ABI and Himawari
AHI measure radiation in infrared and water-vapor bands. Those measurements
describe cloud-top temperature, moisture, and atmospheric structure; they do
not measure surface wind directly. Different surface-wind fields can produce
similar cloud patterns, and cloud structure also depends on shear, moisture,
storm motion, and the stage of the cyclone lifecycle. The mapping from GEO to
surface wind is therefore non-unique.

SAR supplies a much closer view of the target quantity, but its wind product is
also a retrieval rather than direct truth. Ocean-surface roughness changes the
radar backscatter, and a geophysical model function converts that signal to wind
speed. Retrieval quality depends on polarization, incidence angle, ancillary
information, rain, sea state, and the high-wind response. SAR also samples only
an occasional swath. The [NOAA tropical SAR technical
description](https://www.star.nesdis.noaa.gov/socd/mecb/sar/tropical_gmf_tech_doc.php)
explains this backscatter-to-wind inversion.

ERA5 is a physically consistent reanalysis: it combines a forecast model and
assimilated observations into a complete atmospheric estimate. It is useful as
large-scale context and as a dense wind anchor, but it is neither an independent
surface observation nor ground truth. See the [ECMWF ERA5 dataset
description](https://www.ecmwf.int/en/forecasts/datasets/era5-hourly-data-single-levels-1940-present).

IBTrACS provides retrospective best-track storm centers and scalar intensity.
It merges agency records without converting every wind estimate to a single
averaging period, so scalar models deliberately use the documented
`USA_WIND`/`USA_SSHS` contract rather than mixing agencies. The [NOAA IBTrACS
product page](https://www.ncei.noaa.gov/products/international-best-track-archive)
describes that provenance.

## Aligned learning problem

Each export is aligned to one 256 × 256 latitude–longitude grid in EPSG:4326;
the principal grouped training configs take a 192 × 192 center crop:

- **GEO observation:** ten GOES ABI or Himawari AHI bands 7–16;
- **ERA5 context:** seven exported atmospheric/surface fields plus derived
  10 m wind speed and relative vorticity;
- **derived context:** distance to the IBTrACS center and three solar-time
  fields;
- **SAR target:** one retrieved near-surface wind-speed channel in m/s; and
- **validity masks:** explicit support for the condition, ERA5 anchor, and SAR
  swath.

The 0.027° pixels are not equal-area. East–west distance changes with latitude,
so storm metrics use raster bounds and a local physical coordinate conversion
instead of treating degrees as kilometres. See [Model inputs and training
targets](../data/index.md) for real examples and exact channel assembly.

## Two-stage formulation

Stage 1 estimates a deterministic baseline:

\[
\hat v_{\mathrm{base}} =
v_{\mathrm{ERA5}} + f_\theta(x_{\mathrm{GEO}}, x_{\mathrm{ERA5}}, x_{\mathrm{derived}}, m)
\]

Stage 2 models the conditional distribution of signed SAR corrections around
that frozen field:

\[
p_\phi\left(
v_{\mathrm{SAR}} - \hat v_{\mathrm{base}}
\mid
x_{\mathrm{GEO}}, x_{\mathrm{ERA5}}, x_{\mathrm{derived}},
\hat v_{\mathrm{base}}, m
\right)
\]

This separates broad-field estimation from probabilistic fine-structure
refinement. It does not remove the ambiguity of the inverse problem; multiple
diffusion members represent alternatives under the learned conditional
distribution. [Read the full two-stage workflow.](../models/two-stage.md)

## Scientific constraints reflected in code

Sparse and imperfect supervision
: Losses and metrics use `target_mask`. A weak off-swath anchor can constrain
  corrections where ERA5 is valid without relabeling those pixels as SAR. SAR
  is treated as the supervised reference inside its footprint, not as an
  uncertainty-free measurement.

Physical scale
: The dataset retains `target_physical` and reversible normalization
  parameters. The deterministic model predicts in m/s; diffusion residuals are
  decoded and added in m/s.

Storm geometry
: Manifests carry IBTrACS center coordinates and raster bounds. The dataset
  derives a distance input, while evaluation converts the grid to local
  physical distances for eye, inner-core, radial-profile, and
  radius-of-maximum-wind metrics.

Solar context
: Pixelwise local-solar-time sine/cosine and solar zenith represent the
  diurnal cycle and help disambiguate the daytime reflected component of Band
  7. Most selected channels are thermal infrared, so these features should not
  be interpreted as a generic daylight correction for every band. NOAA lists
  the [ABI band wavelengths and purposes](https://www.goes.noaa.gov/abispectralattributes.php)
  and notes Band 7's reflected daytime component in its [band quick
  guide](https://goes-r.noaa.gov/mission/ABI-bands-quick-info.html).

Reproducible probabilistic validation
: Validation noise is derived from the global validation seed and `sample_id`,
  so the same sample starts from the same latent across epochs.

Vector-aware augmentation
: Flips transform ERA5 wind components and vorticity according to their
  physical parity instead of treating every channel as a generic scalar image.

## Interpretation and limits

- Report reconstruction skill only on the observed SAR footprint and against
  the exact baseline used by the model.
- Do not interpret a visually plausible unobserved region as independently
  verified wind. The off-swath field is constrained by context and
  regularization, not by SAR loss.
- Validation and test splits are storm-disjoint in current modular configs,
  but nearby samples within one storm are temporally correlated. Storm-level
  aggregation is therefore important.
- SAR-derived peak wind, IBTrACS best-track intensity, and the maximum of a
  reconstructed grid are related but not interchangeable quantities.
- The current reconstruction models consume a single colocated time, not a
  temporal image window. The current system does not represent arbitrary
  heterogeneous observation sets, learned availability policies, or complete
  tracks as field-model inputs.

Source download is outside `prepare_data()`; export is an explicit preprocessing
step.
