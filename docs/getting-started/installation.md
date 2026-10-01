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
- [Model repository on Hugging Face](https://huggingface.co/simon-donike/geo2wf-models): the linked destination for model releases; see [availability and checkpoint requirements](../models/index.md#checkpoints).

Read the [dataset guide](../data/index.md) before downloading large assets.
Current training consumes a [local raster export or task cache](../data/dataset-contract.md).
The Hub catalog and the original export layout are different; release-specific
catalog tools are not included in this checkout.

Once data are available locally, run the [first experiment](first-experiment.md)
or select a [model preset](../models/index.md). Inference requires a compatible
checkpoint, its resolved configuration, and the original normalization statistics.

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
