from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSpectrogramDenoiser(nn.Module):
    def __init__(self, base_channels: int = 16, mask_scale: float = 1.0) -> None:
        super().__init__()
        c = base_channels
        self.mask_scale = mask_scale
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.bottleneck = ConvBlock(c * 4, c * 8)

        self.dec3 = ConvBlock(c * 8 + c * 4, c * 4)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = ConvBlock(c * 2 + c, c)
        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (h, w)

    @staticmethod
    def _crop(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        h, w = shape
        return x[:, :, :h, :w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, original_shape = self._pad_to_multiple(x, multiple=16)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        mask = torch.sigmoid(self.out_conv(d1)) * self.mask_scale
        return self._crop(mask, original_shape)
