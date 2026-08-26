from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectConv2d(nn.Conv2d):
    """Reflect-padded convolution that also accepts degenerate feature maps.

    PyTorch requires each reflected axis to be larger than its padding. Deep
    U-Nets can reach a 1-pixel axis for tiny inputs, where reflection is not
    defined. In that case only, repeat the edge value along the degenerate axis.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["padding_mode"] = "reflect"
        super().__init__(*args, **kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pad_height, pad_width = self.padding
        if inputs.shape[-2] > pad_height and inputs.shape[-1] > pad_width:
            return super().forward(inputs)

        padded = inputs
        if pad_width:
            mode = "reflect" if inputs.shape[-1] > pad_width else "replicate"
            padded = F.pad(padded, (pad_width, pad_width, 0, 0), mode=mode)
        if pad_height:
            mode = "reflect" if inputs.shape[-2] > pad_height else "replicate"
            padded = F.pad(padded, (0, 0, pad_height, pad_height), mode=mode)
        return F.conv2d(
            padded,
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )


__all__ = ["ReflectConv2d"]
