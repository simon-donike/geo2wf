# geo2wf

Reconstruct tropical-cyclone surface wind fields and estimate intensity from
geostationary satellite imagery, with optional ERA5 inputs and SAR supervision.

[Documentation](https://tcd.hyperalis.com/) ·
[Data guide](docs/data/index.md) ·
[Model guide](docs/models/index.md) ·
[StormSense](https://tcd.hyperalis.com/explorer/dashboard.html)

## Get started

Use Python 3.10 or 3.11:

```bash
uv sync --frozen
```

The repository includes data loaders, model implementations, and training
presets. Observation rasters and checkpoint binaries are stored separately;
point the commands below at your local data and model files.
See the [dataset layout](docs/data/dataset-contract.md) and
[corpus manifest](docs/data/full-dataset.md) for the expected inputs.

## Use a trained model

Run wind-field inference with a checkpoint and its matching configuration:

```bash
uv run geo2wf-infer deterministic-residual \
  --config /path/to/resolved-config.yaml \
  --checkpoint /path/to/model.ckpt
```

Use `--help` for input/output options. Scalar intensity correction and forecasting
are available through `geo2wf-infer intensity-correction` and
`geo2wf-infer intensity-forecast`.

## Train on your data

```bash
WANDB_DISABLED=true uv run geo2wf-train \
  data=geo_sar_common10_era5 \
  model=deterministic_residual \
  data.root=/path/to/geo_sar \
  data.stats_file=/path/to/geo_sar/stats.json
```

The [first experiment](docs/getting-started/first-experiment.md) walks through
loading a batch and running a small training job. [Model presets](docs/experiments/intensity-comparison.md)
cover field U-Nets, joint field/intensity models, and scalar heads.

## Repository

```text
src/geo2wf/   data loaders, models, training, inference, and metrics
configs/      data and model presets
scripts/      data preparation, inference, evaluation, and website utilities
docs/         usage guides, model descriptions, results, and StormSense
tests/        data, model, and integration checks
```

For development:

```bash
uv sync --frozen --group dev --group docs
uv run python -m pytest
uv run mkdocs build --strict
```
