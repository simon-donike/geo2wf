from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dif_img_rec_matplotlib")

import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset

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
        include_pmw: bool = False,
        pmw_as_condition: bool = False,
        max_pmw_time_gap_hours: float | None = None,
        pmw_include_time_offset: bool = False,
        include_ibtracs: bool = False,
        normalization: str | None = None,
        target_normalization: str | None = None,
        robust_clip: float = DEFAULT_ROBUST_CLIP,
        target_robust_clip: float | None = None,
        max_era5_time_gap_hours: float | None = None,
        intensity_balanced_sampling: bool = False,
        intensity_balance_bins_ms: tuple[float, ...] = (
            0.0,
            17.0,
            25.0,
            33.0,
            43.0,
            60.0,
            float("inf"),
        ),
        intensity_balance_quantile: float = 0.995,
        intensity_balance_power: float = 1.0,
        intensity_balance_max_weight_ratio: float = 8.0,
        sampling_seed: int = 42,
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
        self.include_pmw = bool(include_pmw)
        self.pmw_as_condition = bool(pmw_as_condition)
        self.max_pmw_time_gap_hours = max_pmw_time_gap_hours
        self.pmw_include_time_offset = bool(pmw_include_time_offset)
        self.include_ibtracs = bool(include_ibtracs)
        self.normalization = normalization
        self.target_normalization = target_normalization
        self.robust_clip = robust_clip
        self.target_robust_clip = target_robust_clip
        self.max_era5_time_gap_hours = max_era5_time_gap_hours
        if len(intensity_balance_bins_ms) < 2:
            raise ValueError("intensity_balance_bins_ms requires at least two edges")
        if any(
            right <= left
            for left, right in zip(
                intensity_balance_bins_ms[:-1],
                intensity_balance_bins_ms[1:],
            )
        ):
            raise ValueError("intensity_balance_bins_ms must be strictly increasing")
        if not 0.0 < float(intensity_balance_quantile) <= 1.0:
            raise ValueError("intensity_balance_quantile must be in (0, 1]")
        if intensity_balance_power < 0:
            raise ValueError("intensity_balance_power must be non-negative")
        if intensity_balance_max_weight_ratio < 1:
            raise ValueError("intensity_balance_max_weight_ratio must be at least one")
        self.intensity_balanced_sampling = bool(intensity_balanced_sampling)
        self.intensity_balance_bins_ms = tuple(
            float(value) for value in intensity_balance_bins_ms
        )
        self.intensity_balance_quantile = float(intensity_balance_quantile)
        self.intensity_balance_power = float(intensity_balance_power)
        self.intensity_balance_max_weight_ratio = float(
            intensity_balance_max_weight_ratio
        )
        self.sampling_seed = int(sampling_seed)
        self._include_test_logged = False
        self._intensity_balance_logged = False
        self._train_sampling_weights: torch.Tensor | None = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[PairedImageDataset] = None
        self.test_dataset: Optional[PairedImageDataset] = None

    @classmethod
    def from_config(cls, config: dict) -> "PairedDataModule":
        data_cfg = config.get("data", {})
        loader_cfg = data_cfg.get("loader", {})
        sampling_cfg = data_cfg.get("sampling", {}).get("intensity_balanced", {})
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
            include_pmw=data_cfg.get("include_pmw", False),
            pmw_as_condition=data_cfg.get("pmw_as_condition", False),
            max_pmw_time_gap_hours=(
                float(data_cfg["max_pmw_time_gap_hours"])
                if data_cfg.get("max_pmw_time_gap_hours") is not None
                else None
            ),
            pmw_include_time_offset=data_cfg.get(
                "pmw_include_time_offset", False
            ),
            include_ibtracs=data_cfg.get("include_ibtracs", False),
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
            intensity_balanced_sampling=sampling_cfg.get("enabled", False),
            intensity_balance_bins_ms=tuple(
                sampling_cfg.get(
                    "bins_ms",
                    [0.0, 17.0, 25.0, 33.0, 43.0, 60.0, float("inf")],
                )
            ),
            intensity_balance_quantile=sampling_cfg.get("quantile", 0.995),
            intensity_balance_power=sampling_cfg.get("power", 1.0),
            intensity_balance_max_weight_ratio=sampling_cfg.get(
                "max_weight_ratio", 8.0
            ),
            sampling_seed=config.get("seed", 42),
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
            if self.intensity_balanced_sampling:
                intensities = _dataset_target_intensities(
                    self.train_dataset,
                    quantile=self.intensity_balance_quantile,
                )
                weights, counts = _balanced_intensity_weights(
                    intensities,
                    bins=self.intensity_balance_bins_ms,
                    power=self.intensity_balance_power,
                    max_weight_ratio=self.intensity_balance_max_weight_ratio,
                )
                self._train_sampling_weights = weights
                if not self._intensity_balance_logged:
                    rank_zero_info(
                        "Intensity-balanced sampling is active: q%.3f target "
                        "wind bins %s have counts %s and weight ratio %.2f.",
                        self.intensity_balance_quantile,
                        self.intensity_balance_bins_ms,
                        counts,
                        float(weights.max() / weights.min().clamp_min(1e-12)),
                    )
                    self._intensity_balance_logged = True
            self.val_dataset = self._make_dataset(self.val_split)

        if stage in (None, "test"):
            self.test_dataset = self._make_dataset(self.test_split)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        sampler = None
        shuffle = True
        if self.intensity_balanced_sampling:
            if self._train_sampling_weights is None:
                raise RuntimeError("intensity sampling weights were not prepared")
            world_size = self.trainer.world_size if self.trainer is not None else 1
            global_rank = self.trainer.global_rank if self.trainer is not None else 0
            sampler = DistributedWeightedSampler(
                self._train_sampling_weights,
                num_replicas=world_size,
                rank=global_rank,
                seed=self.sampling_seed,
            )
            shuffle = False
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def val_dataloader(self) -> list[DataLoader]:
        if self.val_dataset is None or self.train_dataset is None:
            self.setup("fit")
        world_size = self.trainer.world_size if self.trainer is not None else 1
        train_sample_count = min(self.batch_size * world_size, len(self.train_dataset))
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

    def _make_dataset(self, split: str, *, augment: bool = False) -> PairedImageDataset:
        dataset = PairedImageDataset(
            root=self.root,
            split=split,
            stats_file=self.stats_file,
            target_size=self.target_size,
            center_crop_size=self.center_crop_size,
            augment=augment,
            require_era5=self.require_era5,
            include_pmw=self.include_pmw,
            include_ibtracs=self.include_ibtracs,
            pmw_as_condition=self.pmw_as_condition,
            max_pmw_time_gap_hours=self.max_pmw_time_gap_hours,
            pmw_include_time_offset=self.pmw_include_time_offset,
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
        if dataset.filtered_missing_pmw_count:
            rank_zero_info(
                "PMW is enabled: filtered %d of %d samples without a PMW "
                "companion from the %s split.",
                dataset.filtered_missing_pmw_count,
                dataset.manifest_sample_count,
                split,
            )
        if dataset.filtered_stale_pmw_count:
            rank_zero_info(
                "PMW age guard: filtered %d samples with missing or > %.2f h "
                "PMW timestamps from the %s split.",
                dataset.filtered_stale_pmw_count,
                self.max_pmw_time_gap_hours,
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


class DistributedWeightedSampler(Sampler[int]):
    """Draw one deterministic globally weighted sample stream across DDP ranks."""

    def __init__(
        self,
        weights: torch.Tensor,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ) -> None:
        weights = torch.as_tensor(weights, dtype=torch.double).flatten()
        if weights.numel() == 0 or not torch.isfinite(weights).all():
            raise ValueError("weights must be a non-empty finite vector")
        if bool((weights <= 0).any()):
            raise ValueError("all sampling weights must be positive")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.weights = weights
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = math.ceil(len(weights) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        indices = global_indices[
            self.rank : self.total_size : self.num_replicas
        ].tolist()
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _dataset_target_intensities(
    dataset: Dataset,
    *,
    quantile: float,
) -> torch.Tensor:
    """Read one robust target intensity per sample, including concatenated splits."""
    if isinstance(dataset, ConcatDataset):
        return torch.cat(
            [
                _dataset_target_intensities(child, quantile=quantile)
                for child in dataset.datasets
            ]
        )
    if isinstance(dataset, Subset):
        values = _dataset_target_intensities(dataset.dataset, quantile=quantile)
        return values[torch.as_tensor(dataset.indices, dtype=torch.long)]
    samples = getattr(dataset, "samples", None)
    root = getattr(dataset, "root", None)
    if samples is None or root is None:
        raise TypeError(
            "intensity-balanced sampling requires a PairedImageDataset-like "
            "dataset with samples and root"
        )
    values = []
    for _, row in samples.iterrows():
        relative_path = row.get("target_path") or row.get("sar_path")
        if not relative_path:
            raise ValueError("dataset sample has no target_path or sar_path")
        with rasterio.open(Path(root) / str(relative_path)) as source:
            target = source.read(1, masked=True).compressed()
        target = target[np.isfinite(target)]
        values.append(
            float(np.quantile(target, quantile)) if target.size else float("nan")
        )
    result = torch.as_tensor(values, dtype=torch.float64)
    if not torch.isfinite(result).all():
        bad = int((~torch.isfinite(result)).sum())
        raise ValueError(f"{bad} training targets have no finite intensity statistic")
    return result


def _balanced_intensity_weights(
    intensities: torch.Tensor,
    *,
    bins: tuple[float, ...],
    power: float,
    max_weight_ratio: float,
) -> tuple[torch.Tensor, list[int]]:
    """Return inverse-frequency sample weights for configured intensity bins."""
    values = torch.as_tensor(intensities, dtype=torch.float64).flatten()
    edges = torch.as_tensor(bins, dtype=torch.float64)
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("intensities must be a non-empty finite vector")
    bin_index = torch.bucketize(values, edges[1:-1], right=False)
    bin_count = len(bins) - 1
    counts_tensor = torch.bincount(bin_index, minlength=bin_count)
    nonzero = counts_tensor > 0
    inverse = torch.ones(bin_count, dtype=torch.float64)
    inverse[nonzero] = (values.numel() / counts_tensor[nonzero].to(torch.float64)).pow(
        float(power)
    )
    sample_weights = inverse[bin_index]
    sample_weights = sample_weights / sample_weights.min().clamp_min(1e-12)
    sample_weights = sample_weights.clamp_max(float(max_weight_ratio))
    return sample_weights, [int(value) for value in counts_tensor.tolist()]


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
