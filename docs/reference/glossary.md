# Glossary

ABI
: Advanced Baseline Imager on GOES geostationary satellites.

AHI
: Advanced Himawari Imager, the analogous geostationary imager used by Himawari.

DDIM
: Denoising Diffusion Implicit Model sampling. It can traverse a subset of training timesteps and is deterministic when eta is zero.

DDPM
: Denoising Diffusion Probabilistic Model. Here it refers to the full ancestral reverse sampler using exact posterior coefficients.

EMA
: Exponential moving average of trained weights, used as a smoother inference model.

ERA5
: ECMWF global atmospheric reanalysis: a model/data-assimilation estimate, not
  a direct observation. geo2wf uses seven single-level fields and derives 10 m
  speed and relative vorticity.

GEO
: Geostationary satellite imagery used as the primary condition.

GeoTIFF
: A TIFF raster carrying geospatial transform/CRS metadata; exports also contain band descriptions, tags, and internal validity masks.

IBTrACS
: International Best Track Archive for Climate Stewardship. It merges
  retrospective agency best-track records; its storm center anchors radial
  evaluation and selected scalar workflows use US-agency wind fields.

Intensity
: A scalar summary, usually maximum sustained wind. In this repository,
  IBTrACS `USA_WIND`, a SAR robust peak, and a reconstructed field maximum are
  distinct target/statistic contracts.

Eye-center displacement
: Post-hoc distance in kilometres between the lowest eligible smoothed wind
  value in the reconstructed field and the corresponding minimum in the
  observed SAR field. IBTrACS constrains the search but is not itself either
  endpoint of the reported distance.

Inner core
: The region within 100 km of the configured storm center in current metrics.

PMW
: Passive microwave observations. A single high-frequency brightness-temperature channel is used as a proxy pretraining target.

Reconstruction / nowcast
: An estimate near the input observation time. geo2wf field models are
  instantaneous reconstructions; they do not predict a future track or field.

RMW
: Radius of maximum wind, estimated here from the peak of an observed-pixel radial wind profile.

Robust scale
: IQR divided by 1.349, with standard-deviation fallback, used for robust condition normalization.

Robust peak
: Mean of the highest configured fraction of valid field pixels. It is less
  sensitive than a raw maximum to one extreme pixel, but it is not the same
  quantity as best-track maximum sustained wind.

SAR
: Synthetic-aperture radar. The supervised target is a C-band-derived
  near-surface wind-speed swath retrieved from ocean radar backscatter through
  a geophysical model function.

Sparse completion
: Weak supervision outside the SAR swath during diffusion training. Absolute
  diffusion can use low-weight ERA5 wind; residual diffusion uses low-weight
  zero correction around its selected baseline. These pixels remain excluded
  from SAR metrics.

Target mask
: Boolean support of observed target pixels; it controls training loss and evaluation validity.

Forecast
: A prediction for a future valid time. The maintained scalar forecast model
  trains at +6 h and can be rolled recursively to +12 h; this is separate from
  instantaneous field reconstruction.
