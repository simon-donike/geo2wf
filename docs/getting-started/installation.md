# Installation

The supported environment uses Python 3.10 or 3.11 and [UV](https://docs.astral.sh/uv/) for reproducible dependency resolution.

## Create the runtime environment

```bash
uv sync --frozen
```

For tests and documentation tools:

```bash
uv sync --frozen --group dev --group docs
```

`--frozen` respects the checked-in `uv.lock`. Omit it only when intentionally changing dependencies.

## Verify the checkout

```bash
uv run python -m pytest
uv run mkdocs build --strict
```

The first command exercises schedules, samplers, data helpers, ERA5 handling, residual learning, diffusion helpers, and wind metrics. The second resolves every documentation page, link, extension, and theme asset.

## Configure machine-local paths

Copy the template only if you need local overrides:

```bash
cp .local.example.env .local.env
```

`scripts.local_env.load_local_env()` reads `.local.env` before exporters and training initialize. Common overrides are:

| Variable | Purpose |
|---|---|
| `TCD_DATA_ROOT` | Root of the larger source-observation archive |
| `GEO_SAR_OUTPUT_ROOT` | Local GEO–SAR export root for standard variants |
| `GEO_PMW_OUTPUT_ROOT` | Local GEO–PMW export root for standard variants |
| `WANDB_MODE=offline` | Keep W&B logging local |
| `WANDB_DISABLED=true` | Disable the W&B logger completely |
| `WANDB_PROJECT` | Override the config’s W&B project |

!!! danger "Keep secrets local"
    `.local.env` is ignored by Git. Never put credentials or cluster-specific private paths into committed YAML files.

## GPU notes

The dependency range intentionally targets PyTorch `<2.2` and Lightning `1.9.3`. A generic `uv sync` may resolve a CPU or platform-appropriate build; on managed GPU systems, use the site’s supported PyTorch installation strategy if CUDA wheels are not supplied by the default index.

The prepared multi-GPU configs use `ddp_find_unused_parameters_false`; only choose them when two CUDA devices are available. See [HPC & multi-GPU](../experiments/hpc.md).

## Preview the documentation

```bash
uv run mkdocs serve
```

Open `http://127.0.0.1:8000`. The development server live-reloads Markdown, YAML, CSS, and theme overrides.
