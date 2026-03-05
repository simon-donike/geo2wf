from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import PairedImageDataset


class PairedDataModule(pl.LightningDataModule):
    """Barebones LightningDataModule for paired (x, y) samples."""

    def __init__(
        self,
        train_length: int = 64,
        val_length: int = 16,
        test_length: int = 16,
        image_size: int = 64,
        x_channels: int = 3,
        y_channels: int = 3,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        shuffle_train: bool = True,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        self.train_length = train_length
        self.val_length = val_length
        self.test_length = test_length
        self.image_size = image_size
        self.x_channels = x_channels
        self.y_channels = y_channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle_train = shuffle_train
        self.persistent_workers = persistent_workers and num_workers > 0

        self.train_dataset: Optional[PairedImageDataset] = None
        self.val_dataset: Optional[PairedImageDataset] = None
        self.test_dataset: Optional[PairedImageDataset] = None

    @classmethod
    def from_config(cls, config: dict) -> "PairedDataModule":
        data_cfg = config.get("data", {})
        dataset_cfg = data_cfg.get("dataset", {})
        loader_cfg = data_cfg.get("loader", {})
        return cls(
            train_length=dataset_cfg.get("train_length", 64),
            val_length=dataset_cfg.get("val_length", 16),
            test_length=dataset_cfg.get("test_length", 16),
            image_size=dataset_cfg.get("image_size", 64),
            x_channels=dataset_cfg.get("x_channels", 3),
            y_channels=dataset_cfg.get("y_channels", 3),
            batch_size=loader_cfg.get("batch_size", 4),
            num_workers=loader_cfg.get("num_workers", 0),
            pin_memory=loader_cfg.get("pin_memory", False),
            drop_last=loader_cfg.get("drop_last", False),
            shuffle_train=loader_cfg.get("shuffle_train", True),
            persistent_workers=loader_cfg.get("persistent_workers", False),
        )

    def prepare_data(self) -> None:
        # Intentionally empty: implement download/indexing logic later.
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        # Intentionally barebones placeholders.
        if stage in (None, "fit"):
            self.train_dataset = PairedImageDataset(
                length=self.train_length,
                x_channels=self.x_channels,
                y_channels=self.y_channels,
                image_size=self.image_size,
            )
            self.val_dataset = PairedImageDataset(
                length=self.val_length,
                x_channels=self.x_channels,
                y_channels=self.y_channels,
                image_size=self.image_size,
            )

        if stage in (None, "test"):
            self.test_dataset = PairedImageDataset(
                length=self.test_length,
                x_channels=self.x_channels,
                y_channels=self.y_channels,
                image_size=self.image_size,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers,
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
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )
