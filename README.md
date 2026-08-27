# geo2wf

**From raw satellite observations to tropical-cyclone surface wind fields.**

`geo2wf` is a research codebase for reconstructing an instantaneous, spatially
resolved tropical-cyclone wind field from geostationary (GEO) satellite
radiances. It can use ERA5 as environmental context and a dense physical
anchor, while colocated synthetic-aperture radar (SAR) wind retrievals provide
sparse spatial supervision during training.

[Documentation](https://tcd.hyperalis.com/) ·
[Scientific problem](docs/concepts/problem.md) ·
[Model inputs](docs/data/index.md) ·
[Models](docs/models/index.md) ·
[Results](docs/results.md) ·
[StormSense explorer](https://tcd.hyperalis.com/explorer/dashboard.html)

![Examples of colocated GEO imagery, SAR wind retrievals, and observation footprints](docs/assets/images/geo-sar-random-pairs.png)

*Colocated observations from several storms. GEO provides broad, frequent
coverage of cloud structure; SAR provides an occasional, irregular swath of
retrieved surface wind. The red cross marks the IBTrACS storm center.*

## Scientific framing

The central problem is an inverse problem: GEO instruments observe radiation
from clouds and the atmosphere, not surface wind. Cloud-top temperature,
moisture structure, eye definition, and convective organization contain
information about the circulation below, but the mapping is not unique. Storm
motion, vertical wind shear, moisture, illumination, and lifecycle stage can
produce similar imagery for different surface-wind structures.

The maintained field model asks a deliberately narrower, testable question:

> Given a colocated GEO observation, storm-relative context, validity masks,
> and optionally ERA5, what surface wind field is most consistent with the SAR
> retrieval available at that observation time?

With ERA5 enabled, the model learns a correction to the reanalysis wind field:

```text
reconstructed wind = ERA5 10 m wind + learned GEO-conditioned correction
```

In notation, this is

$$
\hat v = v_{\mathrm{ERA5}} +
f_\theta(x_{\mathrm{GEO}}, x_{\mathrm{ERA5}},
x_{\mathrm{storm}}, x_{\mathrm{solar}}, m).
$$

The formulation keeps the large-scale physical state explicit while allowing
the network to recover finer storm structure supported by the observations.
The no-ERA5 comparison uses the same U-Net family to predict absolute wind
directly from GEO and derived context.

This is **instantaneous wind-field reconstruction**, not track forecasting or
future wind-field forecasting. A separate downstream model estimates scalar
intensity change over a short forecast horizon.

## Raw observation to wind field

The end-to-end workflow is:

1. **Pair observations.** Match GEO imagery and SAR wind retrievals by storm
   and time, using IBTrACS for storm identity and center position.
2. **Create a common geospatial sample.** Reproject the sources onto a shared
   latitude–longitude grid while preserving physical values, masks, bounds,
   timestamps, and sensor provenance in GeoTIFFs and manifests.
3. **Assemble model context.** Load the infrared and water-vapor GEO bands,
   optional ERA5 fields, storm-center distance, local solar time, solar zenith,
   and explicit validity masks.
4. **Learn from sparse SAR supervision.** Compute the physical training loss
   only where the SAR target is valid. An optional weak ERA5 anchor constrains
   unsupported off-swath corrections without treating ERA5 as observed truth.
5. **Reconstruct in physical units.** Produce a dense surface wind-speed field
   and derive storm-scale diagnostics or scalar intensity from that field.
6. **Evaluate by storm.** Compare predictions with the observed SAR footprint
   and diagnose eye, inner-core, radial, peak-wind, and wind-radius structure
   using georeferenced distances.

![Four infrared and water-vapor GEO channels used as model inputs](docs/assets/images/data-example-geo.webp)

*Four channels from one real Himawari sample. The model receives the full
multiband tensor rather than the false-color composites used for display.*

![ERA5 wind anchor, SAR wind target, and SAR target mask on a common grid](docs/assets/images/data-example-target.webp)

*ERA5 supplies a dense but comparatively smooth physical anchor. SAR is the
spatial training reference only inside its valid swath; missing target pixels
are never interpreted as zero wind.*

## What the repository produces now

The active package supports several related outputs around the same observation
and data contracts:

- a deterministic two-dimensional surface wind-speed reconstruction, either
  as an ERA5 residual or as an absolute GEO-conditioned field;
- a current maximum-wind estimate obtained from a reconstructed field;
- a learned scalar correction applied to a frozen field reconstruction;
- a joint U-Net and bottleneck MLP that produces both the wind field and
  current storm intensity;
- an optional latent structure head for maximum wind and wind radii while
  retaining the spatial decoder; and
- a separate scalar intensity model for short-range intensity change.

Training runs also emit a resolved configuration, machine-readable run
manifest, CSV logs, optional Weights & Biases media, and checkpoints. Evaluation
and storm inference export structured prediction files and georeferenced
reconstruction figures.

![Qualitative validation output showing GEO condition, reconstructed wind, SAR target, and valid footprints](docs/assets/images/wandb-logging-preview.png)

*A qualitative validation output. Prediction and SAR target are shown on the
same geospatial frame, with the observed footprint exposed explicitly. Such
panels are intended to be interpreted alongside held-out physical and
storm-structure evaluation, not as standalone evidence of skill.*

## Interpretation boundaries

The scientific contracts are reflected directly in the code:

- **SAR is a retrieval, not uncertainty-free ground truth.** It is closer to
  the target quantity than GEO radiance, but rain, sea state, incidence angle,
  polarization, and high-wind calibration affect the retrieval.
- **ERA5 is context, not an independent label.** It combines a forecast model
  and assimilated observations and is used as an environmental description and
  optional baseline.
- **Masks define what is observed.** Target-based losses and evaluation exclude
  pixels outside the valid SAR swath.
- **The grid is geographic.** Physical storm diagnostics use raster bounds and
  local distance conversion rather than treating degrees as kilometres.
- **Splits are storm-disjoint in the maintained modular configurations.** This
  reduces leakage from temporally correlated samples of the same cyclone.
- **Off-swath output is conditional reconstruction.** A plausible unobserved
  region is constrained by context and regularization, but is not independently
  verified by SAR.

See [Scientific problem and observations](docs/concepts/problem.md) for the
paper-style statement of the inverse problem, provenance of each observing
system, and the limits on scientific interpretation. Detailed held-out results
and provenance live in [Results](docs/results.md); the README intentionally does
not duplicate benchmark tables.

## Install

Python 3.10 or 3.11 and [uv](https://docs.astral.sh/uv/) are supported.

```bash
git clone https://github.com/simon-donike/geo2wf.git
cd geo2wf
uv sync --frozen --group dev --group docs
uv run python -m pytest
uv run mkdocs build --strict
```

The source observations and exported training corpus are not bundled with the
repository. Exporters normally read the larger tropical-cyclone archive from
`TCD_DATA_ROOT` or an explicit `--data-root`.

## Train the field model

Configuration is composed from Hydra-style data, model, trainer, logging, and
experiment groups. The default composition is `configs/modular.yaml`.

```bash
# ERA5-residual wind-field reconstruction
uv run geo2wf-train experiment=intensity_comparison_unet

# Matched no-ERA5 comparison
uv run geo2wf-train experiment=intensity_comparison_unet_no_era5
```

For a minimal local smoke run without external tracking:

```bash
WANDB_DISABLED=true uv run geo2wf-train \
  model=deterministic_residual \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=1 \
  trainer.enable_checkpointing=false \
  data.loader.num_workers=0
```

The retained joint, structure, and forecast presets are listed in
[Choose an experiment](docs/experiments/index.md).

## Export, evaluate, and infer

```bash
# Build aligned GEO–SAR training samples
uv run geo2wf-export geo-sar --config configs/config.yaml

# Inspect an evaluation workflow
uv run geo2wf-evaluate intensity-comparison --help

# Reconstruct a storm with a selected checkpoint
uv run geo2wf-infer deterministic-residual \
  --config /path/to/run/resolved-config.yaml \
  --checkpoint /path/to/model.ckpt
```

Training uses Hydra composition. Export, evaluation, and inference use
workflow-specific command-line interfaces; use `--help` before a long run.

## Repository map

```text
configs/                 composable data, model, trainer, and experiment YAML
src/geo2wf/
├── cli/                 training, export, evaluation, and inference entry points
├── config/              composition, schemas, and compatibility loading
├── data/                datasets, contracts, masks, features, and sampling
├── preprocessing/       source-to-feature and export logic
├── models/              field, intensity, structure, and forecast models
├── objectives/          reusable physical loss primitives
├── metrics/             pixel and storm-relative evaluation
├── evaluation/          shared prediction evaluation
├── inference/           checkpoint loading and prediction services
├── visualization/       georeferenced reconstruction figures
└── tracking/            manifests, CSV logs, callbacks, and W&B media
scripts/                 dataset, evaluation, inference, and figure workflows
tests/                   unit and integration coverage
docs/                    scientific, technical, and experiment documentation
archived/                retired diffusion, PMW-proxy, and historical workflows
```

Model packages share `WindFieldBatch`, `DataSpec`, `PredictionRequest`, and
`PredictionBatch` contracts. Deterministic and joint paths therefore use the
same physical-unit prediction, evaluation, visualization, and checkpoint
interfaces.

## Read next

- [Scientific problem and observation limits](docs/concepts/problem.md)
- [Model inputs and training targets](docs/data/index.md)
- [ERA5-residual field model](docs/models/era5-residual.md)
- [Active model overview](docs/models/index.md)
- [First experiment](docs/getting-started/first-experiment.md)
- [Training and checkpoints](docs/experiments/training.md)
- [Evaluation methodology](docs/experiments/evaluation.md)
- [Commands and environment](docs/reference/commands.md)
- [Modular architecture](docs/concepts/modular-architecture.md)
- [Archived work](docs/archived/index.md)
