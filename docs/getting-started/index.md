# Start here

The supported workflow is an installable `geo2wf` package with a composed
training configuration and thin export, evaluation, and inference commands.

```mermaid
flowchart LR
  A[Source manifest] --> B[geo2wf-export]
  B --> C[WindFieldBatch + DataSpec]
  C --> D[geo2wf-train]
  D --> E[CheckpointLoader]
  E --> F[geo2wf-evaluate / geo2wf-infer]
```

## Recommended reading order

1. [Install and verify the package](installation.md).
2. Run the [first smoke experiment](first-experiment.md).
3. Learn [how config groups and overrides compose](../experiments/configuration.md).
4. Read the [two-stage scientific workflow](../models/two-stage.md).
5. Use the [command reference](../reference/commands.md) for evaluation and inference.

## Main runtime pieces

`geo2wf-train`
: Composes `data`, `model`, `trainer`, `logging`, and optional `experiment`
  groups; validates the model against the dataset `DataSpec`; creates the run
  directory; and starts Lightning.

`PairedDataModule`
: Builds datasets and loaders from split manifests. Its canonical collator
  stacks tensors while keeping metadata sample-oriented.

`WindFieldLightningModule`
: Defines the common training-objective and physical-prediction extension
  points used by modular models.

`CheckpointLoader` and `PredictionService`
: Strict-load old or new checkpoints and expose deterministic and ensemble
  predictions through one physical-unit `PredictionBatch`.

## Choose a route

- To prove the installation and configuration, follow [First experiment](first-experiment.md).
- To train the main stack, use the [Stage 1 → Stage 2 sequence](../models/two-stage.md#training-sequence).
- To add a component, follow [Adding models, datasets, and metrics](../reference/adding-components.md).
- To understand package ownership, read [Modular package architecture](../concepts/modular-architecture.md).

!!! warning "Dataset access is external"
    Source observations are not bundled with the repository. Exporters normally
    read the larger tropical-cyclone archive selected by `TCD_DATA_ROOT` or
    `--data-root`. An existing export only needs split manifests, rasters, and
    `stats.json`.
