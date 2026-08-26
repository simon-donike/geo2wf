# geo2wf

`geo2wf` reconstructs tropical-cyclone surface wind fields from geostationary
satellite imagery and optional ERA5 or PMW context, using matched SAR wind
retrievals as spatial supervision. The main workflow trains a deterministic
ERA5-residual wind-field model. Separate model
families handle PMW brightness-temperature proxy reconstruction, current scalar
intensity, and future scalar intensity.

![Random GEO-SAR training pairs](docs/assets/images/geo-sar-random-pairs.png)

The repository is an installable Python package. Dataset, model, trainer, and
logging choices are composed independently, and model packages share the same
batch, prediction, metrics, visualization, and checkpoint contracts.

## Install and verify

Python 3.10 or 3.11 and [uv](https://docs.astral.sh/uv/) are supported.

```bash
uv sync --frozen --group dev --group docs
uv run python -m pytest
uv run mkdocs build --strict
```

## Canonical workflow

Training uses Hydra-style config groups. The default composition is
`configs/modular.yaml`.

```bash
# Deterministic correction around ERA5
uv run geo2wf-train \
  data=geo_sar_common10_era5 \
  model=deterministic_residual
```

Switching models is a config override:

```bash
uv run geo2wf-train model=direct_unet data=geo_pmw_near89_common10_era5
uv run geo2wf-train model=bottleneck_unet_mlp
```

Override individual values on the command line:

```bash
WANDB_DISABLED=true uv run geo2wf-train \
  model=deterministic_residual \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=1 \
  trainer.enable_checkpointing=false \
  data.loader.num_workers=0
```

The other installed entry points provide the maintained export, evaluation,
and storm-inference workflows:

```bash
uv run geo2wf-export geo-sar --config configs/config_geo_sar_10bands_era5.yaml

uv run geo2wf-evaluate \
  --config configs/config_geo_sar_10bands_era5_residual.yaml \
  --checkpoint /path/to/model.ckpt \
  --output logs/evaluation.json

uv run geo2wf-infer deterministic-residual \
  --config configs/config_geo_sar_10bands_era5_residual.yaml \
  --checkpoint /path/to/model.ckpt
```

Training is the first command migrated to native Hydra composition. Export,
evaluation, and storm inference use canonical package entry points but retain
their established argparse/full-YAML interfaces for compatibility.

## Configuration layout

```text
configs/
├── modular.yaml                   default composition
├── data/                          dataset and loader choices
├── model/                         model constructors and hyperparameters
├── trainer/                       Lightning runtime and checkpoints
├── logging/                       W&B settings
├── experiment/ablations/          focused overrides
└── export/                        reusable export settings
```

Every model config has its own `_target_`; there is no model-name registry to
edit. Available choices are the YAML filenames in the corresponding group.

## Package layout

```text
src/geo2wf/
├── cli/                 thin command entry points
├── config/              composition, schemas, and compatibility loading
├── data/                contracts, datasets, collation, features, and sampling
├── models/              model-specific Lightning packages
├── objectives/          reusable loss primitives
├── metrics/             physical and storm metrics
├── visualization/       plotting functions that return Matplotlib figures
├── tracking/            callbacks, W&B media, CSV logs, and run manifests
├── evaluation/          shared prediction evaluation
├── inference/           checkpoint loading and physical prediction service
└── preprocessing/       reusable source-to-feature logic
```

The shared model contract consists of:

- `WindFieldBatch` and `DataSpec` in `geo2wf.data.contracts`;
- `WindFieldLightningModule`, `LossOutput`, `PredictionRequest`, and
  `PredictionBatch` in `geo2wf.models.base`; and
- `CheckpointLoader` and `PredictionService` in `geo2wf.inference`.

Deterministic predictions use an ensemble dimension of one through the shared
physical-unit interface.

## Data

Exported datasets contain raw GeoTIFF values, internal masks, split manifests,
and training-only statistics:

```text
data/geotiff/geo_sar_10bands_era5/
├── stats.json
├── train/manifest.csv
├── val/manifest.csv
└── test/manifest.csv
```

Source observations are not bundled with this repository. Exporters normally
read the larger tropical-cyclone archive selected by `TCD_DATA_ROOT` or an
explicit `--data-root`.

## Run outputs

Each training launch creates a timestamped directory under
`trainer.default_root_dir` (normally `logs/`). It contains a machine-readable
run manifest, CSV metrics regardless of W&B availability, optional W&B files,
and checkpoints when enabled.

Disable external tracking without changing a model:

```bash
export WANDB_DISABLED=true
```

Use `WANDB_MODE=offline` instead to keep local W&B artifacts for later sync.

## Compatibility

Existing full YAML files, `python train.py --config ...`, the old `data.*` and
CamelCase `src.*` imports, and existing checkpoints remain supported through
deprecated forwarding adapters. New work should import from `geo2wf` and use
the installed commands. Compatibility removal is not part of this refactor.

## Documentation

- [Scientific problem and observation limits](docs/concepts/problem.md)
- [Complete maintained model overview](docs/models/index.md)
- [First experiment](docs/getting-started/first-experiment.md)
- [Configuration and Hydra overrides](docs/experiments/configuration.md)
- [Training and checkpoints](docs/experiments/training.md)
- [Commands and environment](docs/reference/commands.md)
- [Modular architecture](docs/concepts/modular-architecture.md)
- [Adding a model, dataset, or metric](docs/reference/adding-components.md)
- [Two-stage scientific workflow](docs/models/two-stage.md)
