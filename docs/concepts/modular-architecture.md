# Modular package architecture

The installable `geo2wf` package under `src/geo2wf/` is the source of truth.
Root scripts and CamelCase model modules are compatibility adapters.

## Package ownership

| Package | Responsibility |
|---|---|
| `geo2wf.cli` | thin train, evaluate, infer, and export entry points |
| `geo2wf.config` | Hydra composition, local environment loading, schemas, legacy loading |
| `geo2wf.data` | contracts, collation, data module, datasets, raster I/O, features, normalization, augmentation, sampling |
| `geo2wf.models` | Lightning modules and model-specific networks/objectives/transforms |
| `geo2wf.objectives` | reusable masked loss primitives |
| `geo2wf.metrics` | physical and storm tensor calculations |
| `geo2wf.visualization` | plotting functions returning Matplotlib figures |
| `geo2wf.tracking` | callbacks, reconstruction media adaptation, CSV/W&B run records |
| `geo2wf.evaluation` | shared prediction evaluation |
| `geo2wf.preprocessing` | source/feature logic reusable by export and raw inference |

A model package may import shared contracts, objectives, and metrics. It must
not import raster I/O, a concrete dataset, CLI code, W&B, or
Matplotlib.

## Configuration-driven construction

```text
configs/modular.yaml
  ├── data/<choice>.yaml       -> local data _target_
  ├── model/<choice>.yaml      -> local model _target_
  ├── trainer/<choice>.yaml
  ├── logging/<choice>.yaml
  └── experiment/<choice>.yaml (optional overrides)
```

```bash
uv run geo2wf-train model=deterministic_residual
uv run geo2wf-train model=direct_unet trainer.devices=2
```

Available choices are discoverable from filenames. Experiments contain only
focused overrides; they do not copy complete model/data/trainer configurations.
The resolved result is saved in every run directory.

## Model extension contract

A modular model subclasses `WindFieldLightningModule` and implements:

```python
def compute_training_objective(batch: WindFieldBatch) -> LossOutput: ...

def predict_batch(
    batch: WindFieldBatch,
    request: PredictionRequest,
) -> PredictionBatch: ...
```

It also implements `configure_optimizers()` or returns an optimizer through its
normal Lightning mechanism. `validate_data_spec()` can be overridden for
companion, target, unit, or shape requirements; the default checks the declared
condition-channel count.

The shared base validates required batch keys, logs standardized training loss
and objective components, exposes Lightning prediction through the common
request, and writes versioned metadata into new checkpoints.

## Prediction contract

`PredictionBatch.samples_physical` always has shape `[B, E, C, H, W]`.
`central_physical` has `[B, C, H, W]`; `baseline_physical` is optional.
Maintained deterministic models use `E=1`, keeping downstream metrics and
serialization independent of the model architecture.

Installed inference subcommands call the maintained workflow scripts. Modular
models expose `predict_batch()` for physical predictions; older full-YAML
checkpoints continue through their workflow-specific compatibility paths.

## Tracking and visualization boundary

Models log scalars and route reconstruction payloads through the tracking layer.
Plotting functions accept structured data and return figures without knowing
about Lightning or W&B. The tracking adapter owns optional imports, and the
callback supports standardized queued events. The
W&B-specific import is kept in the tracking layer; CSV metrics and
machine-readable manifests remain independent.

## Command migration status

- `geo2wf-train` natively composes Hydra groups.
- `geo2wf-evaluate`, `geo2wf-infer`, and `geo2wf-export` are installed entry
  points over maintained workflows and retain their argparse and full-YAML
  options.
- Root scripts and legacy imports forward to the same source implementation and
  remain supported with deprecation warnings.

See [Configuration](../experiments/configuration.md), [Commands](../reference/commands.md),
and [Adding components](../reference/adding-components.md).
