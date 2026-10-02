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

## Download data and models

Both releases are public on Hugging Face; no login is required. The
[dataset](https://huggingface.co/datasets/simon-donike/geo2wf-data) contains
about **32.9 GB of scientific assets**, plus catalogs and reference results.
The [model release](https://huggingface.co/simon-donike/geo2wf-models) includes
**21 checkpoints (1.27 GB)**, configurations, provenance, and matching source code.
These binaries are stored separately from Git.

From this repository's root, install the download CLI and start with metadata:

```bash
uv tool install huggingface_hub
python3 scripts/download_artifacts.py                 # Small metadata download
python3 scripts/download_artifacts.py all --dry-run   # Preview full download sizes
python3 scripts/download_artifacts.py all             # Download data and models
```

Use `data` or `models` instead of `all` to download only one release. Files go
to `downloads/data/` and `downloads/models/`; change the parent with
`--output-dir /path/to/storage`. Rerun the same command after an interruption;
the Hub client reuses completed downloads. The script pins matching immutable
data/model commits.

For paper reproduction, use `downloads/models/code/conference-source.tar.gz`:
it includes the catalog loaders and selective download tools. Follow the
[download and reproduction guide](docs/data/index.md#use-the-downloads).
The commands below use the current checkout and expect a
[local raster export](docs/data/dataset-contract.md), not the Hub catalog root.

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
