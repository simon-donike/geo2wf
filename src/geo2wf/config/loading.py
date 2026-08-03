"""Resolve Hydra configs and instantiate components without central registries."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def _plain(value: Any) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)  # type: ignore[return-value]
    return dict(value)


def compose_config(
    overrides: Sequence[str] = (),
    *,
    config_dir: str | Path | None = None,
    config_name: str = "modular",
) -> dict[str, Any]:
    """Compose one workspace config while remaining safe in repeated test calls."""

    root = (
        Path(config_dir)
        if config_dir is not None
        else Path(__file__).resolve().parents[3] / "configs"
    ).resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(root)):
        configured = compose(config_name=config_name, overrides=list(overrides))
    return _plain(configured)


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a legacy full YAML file or compose a Hydra defaults file."""

    resolved = Path(path).expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if "defaults" not in payload:
        return payload
    return compose_config(config_dir=resolved.parent, config_name=resolved.stem)


def instantiate_model(
    config: dict[str, Any],
    *,
    legacy_factory: Callable[[dict[str, Any]], Any] | None = None,
):
    """Instantiate a canonical target model or delegate legacy translation."""

    model_config = config.get("model", {})
    if "_target_" in model_config:
        return instantiate(OmegaConf.create(model_config), _convert_="all")
    if legacy_factory is None:
        raise ValueError(
            "legacy model config has no _target_; use the compatibility loader"
        )
    return legacy_factory(config)


def instantiate_datamodule(config: dict[str, Any]):
    """Instantiate canonical data or preserve the legacy nested data section."""

    data_config = config.get("data", {})
    if "_target_" in data_config:
        return instantiate(OmegaConf.create(data_config), _convert_="all")
    from geo2wf.data.datamodule import PairedDataModule

    return PairedDataModule.from_config(config)


def build_paired_datamodule(**data_config: Any):
    """Adapt readable nested data config to the DataModule."""

    from geo2wf.data.datamodule import PairedDataModule

    return PairedDataModule.from_config({"data": data_config})
