# Project map

```text
geo2wf/
├── configs/
│   ├── modular.yaml
│   ├── data/ model/ trainer/ logging/
│   ├── experiment/ export/
│   └── config*.yaml                 compatibility research presets
├── src/geo2wf/
│   ├── cli/
│   ├── config/
│   ├── data/
│   │   ├── contracts.py collation.py datamodule.py
│   │   ├── datasets/paired_geotiff.py
│   │   └── manifests.py raster_io.py normalization.py features.py
│   ├── models/
│   │   ├── base.py
│   │   ├── conditional_diffusion/
│   │   ├── deterministic_residual/
│   │   └── residual_diffusion/
│   ├── diffusion/
│   │   ├── process.py forward_process.py beta_schedules.py
│   │   ├── samplers/
│   │   └── backbones/
│   ├── objectives/ metrics/ visualization/ tracking/
│   ├── evaluation/ inference/ preprocessing/
│   └── training.py
├── scripts/                            maintained workflow implementations
├── tests/
├── train.py and legacy modules         forwarding compatibility adapters
├── mkdocs.yml
└── pyproject.toml
```

## Where to make a change

| Responsibility | Canonical location |
|---|---|
| training composition and Trainer wiring | `src/geo2wf/training.py`, `src/geo2wf/config/` |
| batch/capability types | `src/geo2wf/data/contracts.py` |
| metadata-safe collation | `src/geo2wf/data/collation.py` |
| split loaders and `DataSpec` | `src/geo2wf/data/datamodule.py` |
| paired GeoTIFF behavior | `src/geo2wf/data/datasets/paired_geotiff.py` |
| raster reads, normalization, derived features, augmentation, sampling | named modules under `src/geo2wf/data/` |
| common model lifecycle contract | `src/geo2wf/models/base.py` |
| one model's network/objective/sampling | its descriptive package under `src/geo2wf/models/` |
| forward diffusion and schedules | `src/geo2wf/diffusion/forward_process.py`, `beta_schedules.py` |
| reverse sampling | `src/geo2wf/diffusion/samplers/` |
| reusable loss primitives | `src/geo2wf/objectives/` |
| physical/storm metrics | `src/geo2wf/metrics/`, framework adaptation in `evaluation/` |
| plotting | `src/geo2wf/visualization/` |
| W&B/CSV/media/run records | `src/geo2wf/tracking/` |
| strict loading and unified prediction | `src/geo2wf/inference/` |
| source pairing and export | reusable APIs under `preprocessing/`; thin maintained scripts under `scripts/` |

Do not add new behavior to `data/`, CamelCase `src/*.py`, the root `train.py`,
or the old `src/DenoisingDiffusionProcess/` paths. They exist so old imports and
checkpoints remain usable.

## Test map

The suite covers composition, contract validation, metadata collation, strict
checkpoint compatibility, data transforms and sampling, schedules, samplers,
model learning behavior, prediction shapes/seeds, metric aggregation, and
architecture boundaries.

```bash
uv run python -m pytest
```

<div class="grid cards" markdown>

- **[Adding components](adding-components.md)** — extension workflow and acceptance checklist.
- **[Configuration reference](configuration.md)** — grouped keys and legacy translation.
- **[Commands](commands.md)** — copy-ready installed commands.
- **[Troubleshooting](troubleshooting.md)** — common data, channel, checkpoint, and runtime failures.

</div>
