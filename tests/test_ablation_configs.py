from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_PATHS = sorted(Path("configs/ablations").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda path: path.stem)
def test_ablation_config_has_reproducible_contract(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    ablation = config["ablation"]

    assert ablation["id"] == path.stem.removeprefix("config_")
    assert ablation["design"] in {
        "control_finetune",
        "single_factor_ablation",
        "cumulative_bundle",
        "anchoring_ablation",
        "single_factor_transform_ablation",
    }
    assert ablation["changes"]
    assert config["data"]["include_test_in_train"] is True
    assert config["validation"]["log_reconstruction_images"] is False
    assert config["trainer"]["devices"] == 1
    assert config["trainer"]["strategy"] is None

    model_type = config["model"]["type"]
    monitor = config["trainer"]["checkpoint"]["monitor"]
    if model_type == "deterministic_residual":
        assert monitor == "val/peak_structure_score"
        assert config["optimization"]["reduce_lr_on_plateau"]["monitor"] == monitor
    else:
        assert model_type == "diffusion_residual"
        assert monitor == "val/probabilistic_refinement_score"
        assert config["validation"]["probabilistic_score_peak_weight"] > 0
        assert (
            config["model"]["classifier_free_guidance"][
                "preserve_baseline_condition_on_dropout"
            ]
            is True
        )
