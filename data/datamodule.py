from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dif_img_rec_matplotlib")

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .dataset import PairedImageDataset


class PairedDataModule(pl.LightningDataModule):
    """Barebones LightningDataModule for paired (x, y) samples."""

    def __init__(
        self,
        root: str | Path = "data/geotiff/geo_sar",
        stats_file: str | Path | None = None,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
        target_size: tuple[int, int] = (256, 256),
        random_flips: bool = True,
        include_test_in_train: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        self.stats_file = Path(stats_file).expanduser() if stats_file else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.target_size = target_size
        self.random_flips = random_flips
        self.include_test_in_train = include_test_in_train
        self._include_test_logged = False

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[PairedImageDataset] = None
        self.test_dataset: Optional[PairedImageDataset] = None

    @classmethod
    def from_config(cls, config: dict) -> "PairedDataModule":
        data_cfg = config.get("data", {})
        loader_cfg = data_cfg.get("loader", {})
        root = _local_data_root(data_cfg.get("root", "data/geotiff/geo_sar"))
        stats_file = _local_stats_file(root, data_cfg.get("stats_file"))
        return cls(
            root=root,
            stats_file=stats_file,
            batch_size=loader_cfg.get("batch_size", 4),
            num_workers=loader_cfg.get("num_workers", 0),
            pin_memory=loader_cfg.get("pin_memory", False),
            persistent_workers=loader_cfg.get("persistent_workers", False),
            train_split=data_cfg.get("train_split", "train"),
            val_split=data_cfg.get("val_split", "val"),
            test_split=data_cfg.get("test_split", "test"),
            target_size=tuple(data_cfg.get("target_size", [256, 256])),
            random_flips=data_cfg.get("random_flips", True),
            include_test_in_train=data_cfg.get("include_test_in_train", False),
        )

    def prepare_data(self) -> None:
        # Intentionally empty: implement download/indexing logic later.
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._make_dataset(
                self.train_split, augment=self.random_flips
            )
            if self.include_test_in_train:
                test_train_dataset = self._make_dataset(
                    self.test_split, augment=self.random_flips
                )
                train_count = len(self.train_dataset)
                test_count = len(test_train_dataset)
                self.train_dataset = ConcatDataset(
                    [self.train_dataset, test_train_dataset]
                )
                if not self._include_test_logged:
                    rank_zero_info(
                        "include_test_in_train is active: adding %d test samples "
                        "to %d train samples (%d combined); validation remains separate.",
                        test_count,
                        train_count,
                        len(self.train_dataset),
                    )
                    self._include_test_logged = True
            self.val_dataset = self._make_dataset(self.val_split)

        if stage in (None, "test"):
            self.test_dataset = self._make_dataset(self.test_split)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def _make_dataset(
        self, split: str, *, augment: bool = False
    ) -> PairedImageDataset:
        return PairedImageDataset(
            root=self.root,
            split=split,
            stats_file=self.stats_file,
            target_size=self.target_size,
            augment=augment,
        )


def _local_data_root(configured_root: str | Path) -> str:
    root = str(configured_root)
    if root.endswith("geo_sar"):
        return os.environ.get("GEO_SAR_OUTPUT_ROOT", root)
    if root.endswith("geo_pmw"):
        return os.environ.get("GEO_PMW_OUTPUT_ROOT", root)
    return root


def _local_stats_file(root: str, configured_stats_file: str | Path | None) -> str:
    if root in {
        os.environ.get("GEO_SAR_OUTPUT_ROOT"),
        os.environ.get("GEO_PMW_OUTPUT_ROOT"),
    }:
        return str(Path(root) / "stats.json")
    if configured_stats_file is not None:
        return str(configured_stats_file)
    return str(Path(root) / "stats.json")
