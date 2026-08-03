from __future__ import annotations

from unittest.mock import patch

import torch

from geo2wf.config import compose_config, instantiate_model
from geo2wf.data.collation import collate_wind_field_samples
from geo2wf.data.contracts import DataSpec, validate_batch
from geo2wf.models.base import (
    LossOutput,
    PredictionBatch,
    PredictionRequest,
    WindFieldLightningModule,
)


class _DummyWindFieldModel(WindFieldLightningModule):
    condition_channels = 2

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def compute_training_objective(self, batch) -> LossOutput:
        loss = ((batch["condition"][:, :1] * self.scale - batch["target"]) ** 2).mean()
        return LossOutput(loss, {"mse": loss})

    def predict_batch(self, batch, request: PredictionRequest) -> PredictionBatch:
        central = batch["condition"][:, :1] * self.scale
        members = central.unsqueeze(1).expand(-1, request.ensemble_size, -1, -1, -1)
        return PredictionBatch(members, central)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _sample(identifier: str) -> dict:
    return {
        "condition": torch.ones(2, 4, 4),
        "condition_mask": torch.ones(1, 4, 4, dtype=torch.bool),
        "target": torch.ones(1, 4, 4),
        "target_physical": torch.ones(1, 4, 4),
        "target_mask": torch.ones(1, 4, 4, dtype=torch.bool),
        "target_norm_offset": torch.zeros(1, 1, 1),
        "target_norm_scale": torch.ones(1, 1, 1),
        "condition_bounds": torch.tensor([-1.0, 1.0, -1.0, 1.0]),
        "target_bounds": torch.tensor([-1.0, 1.0, -1.0, 1.0]),
        "center": torch.tensor([0.0, 0.0]),
        "sample_id": identifier,
        "meta": {
            "storm_id": "AL012026",
            "condition_channels": ["ir", "mask"],
            "target_channels": ["wind_speed"],
        },
    }


def test_hydra_model_group_switches_without_dispatch() -> None:
    config = compose_config(["model=deterministic_residual"])
    assert config["model"]["_target_"].endswith("ERA5ResidualRegressor")
    model = instantiate_model(config)
    assert type(model).__module__ == "geo2wf.models.deterministic_residual.module"


def test_canonical_collator_preserves_sample_oriented_metadata() -> None:
    batch = collate_wind_field_samples([_sample("a"), _sample("b")])
    validate_batch(batch)
    assert batch["sample_id"] == ["a", "b"]
    assert [item["storm_id"] for item in batch["meta"]] == [
        "AL012026",
        "AL012026",
    ]


def test_minimal_model_uses_shared_train_and_prediction_contract() -> None:
    model = _DummyWindFieldModel()
    batch = collate_wind_field_samples([_sample("a"), _sample("b")])
    model.validate_data_spec(DataSpec(("ir", "mask"), ("wind_speed",), (4, 4)))
    with patch.object(model, "log"):
        loss = model.training_step(batch, 0)
    prediction = model.predict_batch(batch, PredictionRequest(ensemble_size=3))
    assert torch.equal(loss, torch.zeros_like(loss))
    assert prediction.samples_physical.shape == (2, 3, 1, 4, 4)


def test_data_spec_rejects_channel_mismatch_before_training() -> None:
    model = _DummyWindFieldModel()
    with torch.no_grad():
        try:
            model.validate_data_spec(DataSpec(("ir",), ("wind_speed",), (4, 4)))
        except ValueError as exception:
            assert "expects 2 condition channels" in str(exception)
        else:
            raise AssertionError("channel mismatch was accepted")


def test_new_checkpoints_include_architecture_metadata() -> None:
    checkpoint = {}
    _DummyWindFieldModel().on_save_checkpoint(checkpoint)
    assert checkpoint["geo2wf"] == {
        "schema_version": 1,
        "model_target": ("test_modular_architecture._DummyWindFieldModel"),
        "batch_schema_version": 1,
    }


def test_model_modules_do_not_depend_on_data_io_or_cli() -> None:
    from pathlib import Path
    import geo2wf.models

    root = Path(geo2wf.models.__file__).parent
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    for forbidden in (
        "import rasterio",
        "from geo2wf.cli",
        "from geo2wf.data.datasets",
    ):
        assert forbidden not in source
