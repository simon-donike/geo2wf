from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from src.reconstruction_logging import log_wandb_reconstruction


def test_shared_logger_emits_physical_reconstruction_to_wandb() -> None:
    figure = MagicMock(dpi=100.0)
    figure.get_size_inches.return_value = (20.0, 8.0)
    experiment = MagicMock()
    module = SimpleNamespace(
        _trainer=SimpleNamespace(is_global_zero=True),
        logger=SimpleNamespace(experiment=experiment),
        global_step=12,
    )
    batch = {
        "condition": torch.zeros((1, 3, 4, 4)),
        "target": torch.zeros((1, 1, 4, 4)),
        "target_physical": torch.full((1, 1, 4, 4), 22.0),
        "target_mask": torch.ones((1, 1, 4, 4), dtype=torch.bool),
    }
    prediction = torch.full((1, 1, 4, 4), 20.0)
    baseline = torch.full((1, 1, 4, 4), 18.0)

    with (
        patch(
            "utils.plotting.plot_validation_reconstruction_batch",
            return_value=figure,
        ) as plot_batch,
        patch("wandb.Image", return_value="wandb-image") as wandb_image,
        patch("matplotlib.pyplot.close") as close_figure,
    ):
        log_wandb_reconstruction(
            module,
            batch,
            prediction,
            wandb_key="images/val_reconstruction",
            target_batch=batch["target_physical"],
            baseline_batch=baseline,
        )

    samples = plot_batch.call_args.args[0]
    assert len(samples) == 1
    assert torch.equal(samples[0]["prediction"], prediction[0])
    assert torch.equal(samples[0]["target"], batch["target_physical"][0])
    assert torch.equal(samples[0]["baseline"], baseline[0])
    wandb_image.assert_called_once_with(figure, file_type="jpg")
    experiment.log.assert_called_once_with(
        {"images/val_reconstruction": "wandb-image"},
        step=12,
    )
    close_figure.assert_called_once_with(figure)
