from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.utils.data import Dataset


class PairedImageDataset(Dataset):
    """Barebones paired dataset that returns (x, y)."""

    def __init__(
        self,
        x_data: Optional[list[torch.Tensor]] = None,
        y_data: Optional[list[torch.Tensor]] = None,
        length: int = 1,
        x_channels: int = 3,
        y_channels: int = 3,
        image_size: int = 64,
        transform_x: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        transform_y: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.x_data = x_data or []
        self.y_data = y_data or []
        self.length = max(1, int(length))
        self.x_channels = int(x_channels)
        self.y_channels = int(y_channels)
        self.image_size = int(image_size)
        self.transform_x = transform_x
        self.transform_y = transform_y

        if self.y_data and len(self.x_data) != len(self.y_data):
            raise ValueError("x_data and y_data must have the same length.")

    def __len__(self) -> int:
        return len(self.x_data) if self.x_data else self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.x_data) == 0:
            # Placeholder sample so the interface works before real data is wired.
            x = torch.zeros(self.x_channels, self.image_size, self.image_size)
            y = torch.zeros(self.y_channels, self.image_size, self.image_size)
            return x, y

        x = self.x_data[idx]
        y = self.y_data[idx] if self.y_data else torch.zeros_like(x)

        if self.transform_x is not None:
            x = self.transform_x(x)
        if self.transform_y is not None:
            y = self.transform_y(y)

        return x, y
