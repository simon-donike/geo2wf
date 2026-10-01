# geo2wf

Reconstruct tropical-cyclone surface wind fields and estimate intensity from
geostationary satellite imagery, with optional ERA5 inputs and SAR supervision.

[Documentation](https://tcd.hyperalis.com/) ·
[Dataset on Hugging Face](https://huggingface.co/datasets/simon-donike/geo2wf-data) ·
[Models on Hugging Face](https://huggingface.co/simon-donike/geo2wf-models) ·
[Model guide](docs/models/index.md) ·
[StormSense](https://tcd.hyperalis.com/explorer/dashboard.html)

## Get started

Use Python 3.10 or 3.11:

```bash
uv sync --frozen
```

The repository includes data loaders, model implementations, and training
presets. Observation rasters and checkpoint binaries are stored separately;
see the [model guide](docs/models/index.md#checkpoints) for release availability,
then point the commands below at your local data and model files.
See the [dataset layout](docs/data/dataset-contract.md) and
[dataset guide](docs/data/index.md) for the expected inputs.

## Use a trained model

Run wind-field inference with a checkpoint and its matching configuration:

```bash
uv run geo2wf-infer deterministic-residual \
  --config /path/to/resolved-config.yaml \
  --checkpoint /path/to/model.ckpt
```

See the [command reference](docs/reference/commands.md) for required data inputs
and workflow-specific help. Scalar intensity correction and forecasting
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
loading a batch and running a small training job. [Model presets](docs/models/index.md#training-presets)
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
