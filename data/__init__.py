__all__ = ["PairedDataModule", "PairedImageDataset"]


def __getattr__(name: str):
    if name == "PairedDataModule":
        from .datamodule import PairedDataModule

        return PairedDataModule
    if name == "PairedImageDataset":
        from .dataset import PairedImageDataset

        return PairedImageDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
