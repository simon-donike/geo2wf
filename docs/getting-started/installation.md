# Installation

The supported environment uses Python 3.10 or 3.11 and
[uv](https://docs.astral.sh/uv/) for dependency locking and command execution.

## Create the environment

```bash
uv sync --frozen
```

For tests and documentation:

```bash
uv sync --frozen --group dev --group docs
```

`--frozen` uses the checked-in `uv.lock`. Omit it only when intentionally
changing dependencies. Installation registers `geo2wf-train`,
`geo2wf-evaluate`, `geo2wf-infer`, and `geo2wf-export`.

## Verify the checkout

```bash
uv run python -m pytest
uv run mkdocs build --strict
uv run geo2wf-train --help
```

The test suite covers config composition, data/model contracts, checkpoint
compatibility, data transforms, schedules/samplers, learning behavior,
prediction shapes, metrics, and architecture boundaries. The strict docs build
checks pages, internal links, Markdown extensions, and assets.

## Configure machine-local values

Copy the ignored template only when local overrides are needed:

```bash
cp .local.example.env .local.env
```

`geo2wf.config.local_environment.load_local_env()` is used by the canonical
training runtime. Maintained compatibility scripts load the same file through
their forwarding environment module.

| Variable | Purpose |
|---|---|
| `TCD_DATA_ROOT` | source-observation archive for exporters |
| `GEO_SAR_OUTPUT_ROOT` | conventional GEO–SAR export destination |
| `GEO_PMW_OUTPUT_ROOT` | conventional GEO–PMW export destination |
| `GEO2WF_BASELINE_CKPT` | frozen Stage 1 checkpoint for composed Stage 2 |
| `WANDB_MODE=offline` | keep W&B activity local |
| `WANDB_DISABLED=true` | disable W&B construction completely |
| `WANDB_PROJECT`, `WANDB_NAME` | override tracking names |

!!! danger "Keep secrets local"
    `.local.env` is ignored by Git. Never commit credentials or private
    cluster paths in YAML.

## GPU notes

The refactor intentionally retains Lightning `1.9.3` and the existing PyTorch
range. A generic sync may select a CPU/platform build; managed GPU systems may
need the site's supported CUDA wheel or module strategy.

Use trainer overrides instead of copying a config:

```bash
uv run geo2wf-train \
  trainer.accelerator=gpu \
  trainer.devices=2 \
  trainer.strategy=ddp_find_unused_parameters_false
```

Only request devices that the machine or scheduler allocation provides. See
[HPC & multi-GPU](../experiments/hpc.md).

## Preview the documentation

```bash
uv run mkdocs serve
```

Open `http://127.0.0.1:8000`. The server live-reloads Markdown, YAML, CSS, and
theme overrides.
