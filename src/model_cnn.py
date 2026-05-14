from __future__ import annotations

import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class SpectrogramCNNMaskDenoiser(nn.Module):
    """Small residual CNN that predicts a magnitude mask on noisy spectrograms."""

    def __init__(self, channels: int = 32, layers: int = 8, mask_scale: float = 1.0) -> None:
        super().__init__()
        self.mask_scale = mask_scale
        self.in_conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        dilations = [1, 2, 4, 8]
        self.blocks = nn.Sequential(
            *[ResidualConvBlock(channels, dilation=dilations[idx % len(dilations)]) for idx in range(layers)]
        )
        self.out_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x)
        h = self.blocks(h)
        return torch.sigmoid(self.out_conv(h)) * self.mask_scale
