# geo2wf — conference release

Reconstruct tropical-cyclone surface wind fields from geostationary satellite
imagery, with optional ERA5 conditioning and sparse SAR supervision.

[Documentation](https://tcd.hyperalis.com/) ·
[Published results](docs/results.md) ·
[Reproduction guide](docs/reproduction.md) ·
[StormSense stormtracker](https://tcd.hyperalis.com/explorer/dashboard.html)

This branch contains the models and ablations behind the two conference paper
tables, the Humberto/Kiko/Otis case studies, and the complete documentation
website and stormtracker. Historical experiments remain in Git history.

## Included models and evidence

- Field U-Net, post-hoc intensity correction, and joint latent MLP, with and
  without ERA5: six original architecture checkpoints.
- Eight latent-MLP supervision ablations: ERA5/no ERA5, SAR/no SAR, and
  wind-only/wind-plus-radii supervision.
- Case-study correction heads, their frozen-field and initialization
  dependencies, and the dashboard MLP forecast.
- Exact paper tables, original full-precision reports, dataset fingerprints,
  resolved configurations, and source provenance in `release/`.

The architecture table uses its original test cohort; the latent table uses
configured validation cohorts. Their joint models are different checkpoints.
The [reproduction guide](docs/reproduction.md) documents dataset differences
and the historical case-study field's training/export configuration discrepancy.

## Install and reproduce

```bash
uv sync --frozen --group dev --group docs
uv run geo2wf-evaluate conference tables
uv run geo2wf-evaluate conference figures
uv run geo2wf-evaluate conference verify --artifact-root /path/to/conference-artifacts
uv run geo2wf-evaluate conference smoke --artifact-root /path/to/conference-artifacts
uv run python -m pytest
uv run mkdocs build --strict
```

The separate checksummed artifact bundle contains 21 local checkpoints and their
saved training provenance. Checkpoint binaries and raw observations are outside
Git. Follow the [reproduction guide](docs/reproduction.md) for exact registry IDs,
portable data paths, checkpoint reevaluation, and training commands.

ConvLSTM forecast exports are retained for the stormtracker; regenerating them
requires the external HPC project and checkpoint identified in the registry.

## Repository

```text
src/geo2wf/   shared model, data, training, evaluation, and tracking code
configs/      retained model, dataset, training, and experiment presets
scripts/      preprocessing, reproduction, inference, figures, and R2 publishing
release/      publication tables, checkpoint registry, original provenance
docs/         scientific documentation and StormSense UI
tests/        model, data, release, and stormtracker checks
```

SAR masks define observed targets; off-swath reconstructions are conditional
predictions. These are instantaneous wind reconstructions. The dashboard's
retrospective forecasts are a separate model capability.
