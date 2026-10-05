# Get started

Use Python 3.10 or 3.11 and [uv](https://docs.astral.sh/uv/).
From the repository root:

```bash
uv sync --frozen
```

This installs the locked dependencies and the `geo2wf-train`, `geo2wf-export`,
`geo2wf-evaluate`, and `geo2wf-infer` commands.

## Get data and model files

- [Dataset on Hugging Face](https://huggingface.co/datasets/simon-donike/geo2wf-data): imagery, caches, labels, indexes, and reference results.
- [Models on Hugging Face](https://huggingface.co/simon-donike/geo2wf-models): 21 checkpoints, original configurations, provenance, and matching source code.

Download without logging in, starting with the small metadata files:

```bash
uv tool install huggingface_hub
python3 scripts/download_artifacts.py
python3 scripts/download_artifacts.py all --dry-run
python3 scripts/download_artifacts.py all
```

The script saves pinned releases under `downloads/data/` and `downloads/models/`.
Use `data` or `models` instead of `all` to fetch one release, and
`--output-dir /path/to/storage` to choose another destination. The full download
includes about 32.9 GB of scientific assets and 1.27 GB of checkpoint weights,
plus metadata and source files.

Read the [dataset guide](../data/index.md#use-the-downloads) for selective
downloads and using the matching source archive for reproduction. Current
training in this checkout consumes a
[local raster export or task cache](../data/dataset-contract.md); the Hub
catalog uses the loaders in the released source archive.

Once data are available locally, run the [first experiment](first-experiment.md)
or select a [model preset](../models/index.md). Inference requires a compatible
checkpoint, its resolved configuration, and the original normalization statistics.

## StormSense and containers

The hosted [StormSense app](https://stormsense.hyperalislabs.com/) requires no
local installation. See the [viewer guide](../explorer.md) for its maps,
imagery, and predictions. To run the app or its hourly inference pipeline,
follow the [application setup guide](https://github.com/simon-donike/geo2wf/tree/main/apps/stormsense).
The [Docker guide](https://github.com/simon-donike/geo2wf/blob/main/apps/stormsense/runner/DOCKER.md)
covers CPU/GPU research workflows, data downloads, and website hosting.

## Local paths and logging

Pass data paths explicitly in commands. Optional machine defaults can be kept
in `.local.env`, copied from `.local.example.env`. `TCD_DATA_ROOT` selects the
source archive for export/inference; `WANDB_DISABLED=true` disables W&B while
keeping local CSV metrics and run records. `WANDB_MODE=offline` records W&B
artifacts locally.

## Development and docs

```bash
uv sync --frozen --group dev --group docs
uv run python -m pytest
uv run mkdocs build --strict
uv run python scripts/check_site_links.py
uv run mkdocs serve
```

The preview opens at `http://127.0.0.1:8000`. GPU training requires a compatible
PyTorch/CUDA installation and allocated devices; use the
[trainer overrides](../experiments/training.md#configuration-and-hardware) for your machine.
