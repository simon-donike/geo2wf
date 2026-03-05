from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import PairedImageDataset


class PairedDataModule(pl.LightningDataModule):
    """Barebones LightningDataModule for paired (x, y) samples."""

    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset: Optional[PairedImageDataset] = None
        self.val_dataset: Optional[PairedImageDataset] = None
        self.test_dataset: Optional[PairedImageDataset] = None

    def prepare_data(self) -> None:
        # Intentionally empty: implement download/indexing logic later.
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        # Intentionally barebones placeholders.
        if stage in (None, "fit"):
            self.train_dataset = PairedImageDataset()
            self.val_dataset = PairedImageDataset()

        if stage in (None, "test"):
            self.test_dataset = PairedImageDataset()

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
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
        )
