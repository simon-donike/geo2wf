# Code map

| Location | Responsibility |
|---|---|
| `src/geo2wf/data/` | Raster/cache loading, contracts, normalization, features, sampling, and augmentation |
| `src/geo2wf/models/` | Field, joint/encoder-only, correction, and forecast models |
| `src/geo2wf/config/`, `training.py` | Hydra composition, compatibility checks, and Lightning training |
| `src/geo2wf/metrics/`, `evaluation/` | Physical errors, storm structure, and evaluation adapters |
| `src/geo2wf/visualization/`, `tracking/` | Figures, CSV/W&B logging, and run provenance |
| `src/geo2wf/cli/`, `scripts/` | Retained export, inference, evaluation, and website workflows |
| `src/geo2wf/operational/` | Storm discovery, GOES ingestion, hourly estimates, forecasts, and versioned data publication |
| `apps/stormsense/` | Current React application, website hosting, browser checks, and runner setup |
| `configs/` | Grouped training presets and the export-only `config.yaml` |
| `tests/` | Data/model contracts, numerical behavior, integration, and dashboard checks |
| `docs/` | These guides, paper results, and legacy three-storm viewer assets |

The root `train.py` and compatibility modules under `src/` preserve existing
imports/checkpoints. New model and loader work belongs under `src/geo2wf/`.
Data/model configs name their `_target_`; training composes and validates them
without a central architecture registry.

Large scientific inputs and model releases belong in the linked
[Hugging Face repositories](../data/index.md). The legacy viewer's external
ViT and ConvLSTM comparison outputs do not imply maintained training
implementations. The current [StormSense app](https://stormsense.hyperalislabs.com/)
uses the maintained joint nowcast and scalar forecast models.

Use [commands](commands.md) for entry points,
[training](../experiments/training.md) for configuration, and
[troubleshooting](troubleshooting.md) for common failures.
