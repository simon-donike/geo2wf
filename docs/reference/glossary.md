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
: ECMWF global atmospheric reanalysis. geo2wf uses seven single-level fields and derives 10 m speed/vorticity.

GEO
: Geostationary satellite imagery used as the primary condition.

GeoTIFF
: A TIFF raster carrying geospatial transform/CRS metadata; exports also contain band descriptions, tags, and internal validity masks.

IBTrACS
: International Best Track Archive for Climate Stewardship. Its storm center anchors radial evaluation.

Inner core
: The region within 100 km of the configured storm center in current metrics.

PMW
: Passive microwave observations. A single high-frequency brightness-temperature channel is used as a proxy pretraining target.

RMW
: Radius of maximum wind, estimated here from the peak of an observed-pixel radial wind profile.

Robust scale
: IQR divided by 1.349, with standard-deviation fallback, used for robust condition normalization.

SAR
: Synthetic-aperture radar. The supervised target is a C-band-derived near-surface wind-speed swath.

Sparse completion
: Weakly filling unobserved SAR target pixels with ERA5 during diffusion training. These pixels receive lower loss weight and remain excluded from SAR metrics.

Target mask
: Boolean support of observed target pixels; it controls training loss and evaluation validity.
