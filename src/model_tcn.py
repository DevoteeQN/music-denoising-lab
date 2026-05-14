from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TCNResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.05) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(1, channels),
            nn.PReLU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
            ),
            nn.GroupNorm(1, channels),
            nn.PReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LightweightTCNDenoiser(nn.Module):
    """Small Conv-TasNet-style encoder-TCN-decoder denoiser.

    This is intentionally compact for course experiments. It is not a full
    Conv-TasNet reproduction, but it follows the same time-domain idea:
    encode waveform, estimate a mask with dilated temporal convolutions, and
    decode the enhanced representation back to waveform.
    """

    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 3,
        blocks: int = 8,
        encoder_kernel: int = 16,
        encoder_stride: int = 8,
        mask_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.encoder_kernel = encoder_kernel
        self.encoder_stride = encoder_stride
        self.mask_scale = mask_scale

        self.encoder = nn.Conv1d(
            1,
            channels,
            kernel_size=encoder_kernel,
            stride=encoder_stride,
            padding=encoder_kernel // 2,
            bias=False,
        )
        dilations = [2 ** (idx % 4) for idx in range(blocks)]
        self.tcn = nn.Sequential(
            *[
                TCNResidualBlock(channels, kernel_size=kernel_size, dilation=dilation)
                for dilation in dilations
            ]
        )
        self.mask_head = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.PReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.decoder = nn.ConvTranspose1d(
            channels,
            1,
            kernel_size=encoder_kernel,
            stride=encoder_stride,
            padding=encoder_kernel // 2,
            bias=False,
        )

    @staticmethod
    def _fix_length(x: torch.Tensor, length: int) -> torch.Tensor:
        current = x.shape[-1]
        if current == length:
            return x
        if current > length:
            return x[..., :length]
        return F.pad(x, (0, length - current))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        original_shape = audio.shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        length = audio.shape[-1]
        encoded = F.relu(self.encoder(audio))
        features = self.tcn(encoded)
        mask = self.mask_head(features) * self.mask_scale
        decoded = self.decoder(encoded * mask)
        decoded = self._fix_length(decoded, length)
        if len(original_shape) == 2:
            decoded = decoded.squeeze(1)
        return decoded
