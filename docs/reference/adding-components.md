# Adding models, datasets, and metrics

The extension rule is intentionally narrow: a new model adds its package,
model config, and tests; a new dataset adds its implementation, data config,
and tests. Existing runtime modules should not need edits.

## Add a model

### 1. Create a descriptive package

```text
src/geo2wf/models/quantile_residual/
├── __init__.py
└── module.py
```

Subclass `WindFieldLightningModule` and implement its two scientific extension
points plus optimizer construction:

```python
import torch

from geo2wf.models.base import (
    LossOutput,
    PredictionBatch,
    PredictionRequest,
    WindFieldLightningModule,
)


class QuantileResidualModel(WindFieldLightningModule):
    checkpoint_monitor = "val/loss"
    checkpoint_mode = "min"

    def __init__(self, condition_channels: int, lr: float = 1e-3):
        super().__init__()
        self.condition_channels = condition_channels
        self.lr = lr
        self.network = torch.nn.Conv2d(condition_channels, 1, 1)

    def compute_training_objective(self, batch) -> LossOutput:
        prediction = self.network(batch["condition"])
        valid = batch["target_mask"].to(prediction)
        error = (prediction - batch["target"]) * valid
        loss = error.square().sum() / valid.sum().clamp_min(1)
        return LossOutput(loss, {"mse": loss})

    def predict_batch(self, batch, request: PredictionRequest) -> PredictionBatch:
        normalized = self.network(batch["condition"])
        offset = batch["target_norm_offset"].to(normalized)
        scale = batch["target_norm_scale"].to(normalized)
        physical = offset + scale * normalized
        members = physical.unsqueeze(1).expand(
            -1, request.ensemble_size, -1, -1, -1
        )
        return PredictionBatch(members, physical)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
```

A production model must handle normalization shapes carefully and should use
shared objective/metric primitives rather than duplicating them.

### 2. Add one model config

```yaml
# configs/model/quantile_residual.yaml
_target_: geo2wf.models.quantile_residual.QuantileResidualModel
condition_channels: 23
lr: 0.001
```

It is immediately selectable:

```bash
uv run geo2wf-train model=quantile_residual
```

### 3. Test the boundaries

Test at least:

- construction from `_target_` without a registry edit;
- accepted and rejected `DataSpec` values;
- finite objective and named components;
- `[B,E,C,H,W]` physical member shape and deterministic seeds where relevant;
- train, validation, and predict smoke behavior;
- strict checkpoint round-trip; and
- absence of concrete dataset, raster, CLI, W&B, and Matplotlib imports.

## Add a dataset

1. Put the implementation below `geo2wf.data.datasets` with a scientific name.
2. Return all required `WindFieldBatch` sample fields.
3. Use `collate_wind_field_samples` so `meta` and `ibtracs` remain per-sample.
4. Expose a `DataSpec` with ordered channel names, target units/shape, and
   available companions.
5. Add one `configs/data/<choice>.yaml` with its local `_target_` or data-module
   factory.
6. Test manifest filtering, tensors/masks, normalization inversion, feature
   order, collation, fixed seeds/augmentation, and model compatibility.

Existing models must consume the shared batch and must not import the new
concrete dataset.

## Add a metric

Put pure tensor logic in `geo2wf.metrics` and framework adaptation in
`geo2wf.evaluation` when required. A reusable metric should:

- accept `PredictionBatch`, `WindFieldBatch`, or their explicitly documented
  tensors;
- operate in physical units when its name says so;
- return sums/counts or otherwise expose correct distributed reduction state;
- treat unavailable masked/storm values as unavailable, not zero; and
- import neither Lightning, W&B, plotting, CLI, nor concrete datasets.

Wire the metric into the shared evaluation collection or tracking callback—not
into every model's prediction method. Preserve established scalar names when
migrating an existing calculation.

## Add a plot

Add a focused function under `geo2wf.visualization`:

```python
def plot_prediction(...):
    ...
    return figure
```

It must not call a logger, access a Trainer, or import W&B. The tracking adapter or reconstruction callback owns media logging and figure
lifetime.

## Naming rules

Use lowercase `snake_case` filenames and names that describe scientific
responsibility. Avoid `utils.py`, `helpers.py`, central registries, and model
letters such as `model_a.py`.

## Acceptance checklist

A component is modular when switching to it requires only a config choice, its
scientific implementation is isolated, incompatible data fails before training,
and the existing trainer/evaluator/tracking/plotting source remains unchanged.
