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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .dataset import DEFAULT_ROBUST_CLIP, PairedImageDataset


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
        center_crop_size: tuple[int, int] | None = None,
        random_flips: bool = True,
        include_test_in_train: bool = False,
        require_era5: bool = False,
        normalization: str | None = None,
        target_normalization: str | None = None,
        robust_clip: float = DEFAULT_ROBUST_CLIP,
        target_robust_clip: float | None = None,
        max_era5_time_gap_hours: float | None = None,
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
        self.center_crop_size = center_crop_size
        self.random_flips = random_flips
        self.include_test_in_train = include_test_in_train
        self.require_era5 = require_era5
        self.normalization = normalization
        self.target_normalization = target_normalization
        self.robust_clip = robust_clip
        self.target_robust_clip = target_robust_clip
        self.max_era5_time_gap_hours = max_era5_time_gap_hours
        self._include_test_logged = False

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[PairedImageDataset] = None
        self.test_dataset: Optional[PairedImageDataset] = None

    @classmethod
    def from_config(cls, config: dict) -> "PairedDataModule":
        data_cfg = config.get("data", {})
        loader_cfg = data_cfg.get("loader", {})
        require_era5 = data_cfg.get(
            "require_era5",
            config.get("export", {}).get("include_era5", False),
        )
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
            center_crop_size=(
                tuple(data_cfg["center_crop_size"])
                if data_cfg.get("center_crop_size") is not None
                else None
            ),
            random_flips=data_cfg.get("random_flips", True),
            include_test_in_train=data_cfg.get("include_test_in_train", False),
            require_era5=require_era5,
            normalization=data_cfg.get("normalization"),
            target_normalization=data_cfg.get("target_normalization"),
            robust_clip=float(data_cfg.get("robust_clip", DEFAULT_ROBUST_CLIP)),
            target_robust_clip=(
                float(data_cfg["target_robust_clip"])
                if data_cfg.get("target_robust_clip") is not None
                else None
            ),
            max_era5_time_gap_hours=(
                float(data_cfg["max_era5_time_gap_hours"])
                if data_cfg.get("max_era5_time_gap_hours") is not None
                else None
            ),
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

    def val_dataloader(self) -> list[DataLoader]:
        if self.val_dataset is None or self.train_dataset is None:
            self.setup("fit")
        world_size = self.trainer.world_size if self.trainer is not None else 1
        train_sample_count = min(
            self.batch_size * world_size, len(self.train_dataset)
        )
        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers and self.num_workers > 0,
        }
        return [
            DataLoader(
                Subset(
                    self.val_dataset,
                    _storm_stratified_indices(self.val_dataset),
                ),
                shuffle=False,
                **loader_kwargs,
            ),
            DataLoader(
                Subset(self.train_dataset, range(train_sample_count)),
                shuffle=False,
                **loader_kwargs,
            ),
        ]

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
        dataset = PairedImageDataset(
            root=self.root,
            split=split,
            stats_file=self.stats_file,
            target_size=self.target_size,
            center_crop_size=self.center_crop_size,
            augment=augment,
            require_era5=self.require_era5,
            normalization=self.normalization,
            target_normalization=self.target_normalization,
            robust_clip=self.robust_clip,
            target_robust_clip=self.target_robust_clip,
            max_era5_time_gap_hours=self.max_era5_time_gap_hours,
        )
        if dataset.filtered_missing_era5_count:
            rank_zero_info(
                "ERA5 is required: filtered %d of %d samples without ERA5 "
                "from the %s split.",
                dataset.filtered_missing_era5_count,
                dataset.manifest_sample_count,
                split,
            )
        if dataset.filtered_stale_era5_count:
            rank_zero_info(
                "ERA5 age guard: filtered %d samples with missing or > %.2f h "
                "ERA5 timestamps from the %s split.",
                dataset.filtered_stale_era5_count,
                self.max_era5_time_gap_hours,
                split,
            )
        return dataset


def _storm_stratified_indices(dataset: Dataset) -> list[int]:
    """Round-robin validation samples by storm for a representative prefix."""
    samples = getattr(dataset, "samples", None)
    if samples is None or "storm_id" not in samples:
        return list(range(len(dataset)))
    groups = []
    for _, group in samples.groupby("storm_id", sort=True):
        groups.append(group.index.tolist())
    ordered = []
    offset = 0
    while True:
        added = False
        for group in groups:
            if offset < len(group):
                ordered.append(int(group[offset]))
                added = True
        if not added:
            break
        offset += 1
    return ordered


def _local_data_root(configured_root: str | Path) -> str:
    root = str(configured_root)
    if root.endswith(("geo_sar", "geo_sar_10bands")):
        return os.environ.get("GEO_SAR_OUTPUT_ROOT", root)
    if root.endswith(("geo_pmw", "geo_pmw_10bands")):
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
