from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class TinyDiTSpectrogramDenoiser(nn.Module):
    """Tiny DiT-style patch Transformer for spectrogram mask prediction.

    This model borrows the patchified 2D-token Transformer idea from DiT, but
    keeps the training objective supervised for a lightweight course setting.
    It predicts a magnitude mask from a noisy log spectrogram.
    """

    def __init__(
        self,
        dim: int = 96,
        depth: int = 4,
        heads: int = 4,
        patch_size: int = 32,
        mask_scale: float = 1.0,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.mask_scale = mask_scale
        patch_dim = patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.to_patch = nn.Linear(dim, patch_dim)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _pad_to_patch(x: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, h, w = x.shape
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (h, w)

    @staticmethod
    def _crop(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        h, w = shape
        return x[:, :, :h, :w]

    def _patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        p = self.patch_size
        b, c, h, w = x.shape
        grid_h = h // p
        grid_w = w // p
        patches = x.reshape(b, c, grid_h, p, grid_w, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(b, grid_h * grid_w, c * p * p)
        return patches, grid_h, grid_w

    def _unpatchify(self, patches: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        p = self.patch_size
        b = patches.shape[0]
        x = patches.reshape(b, grid_h, grid_w, 1, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, 1, grid_h * p, grid_w * p)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, original_shape = self._pad_to_patch(x, self.patch_size)
        patches, grid_h, grid_w = self._patchify(x)
        tokens = self.patch_embed(patches)
        if tokens.shape[1] > self.pos_embed.shape[1]:
            raise RuntimeError(
                f"Too many patches ({tokens.shape[1]}). Increase max_tokens for this spectrogram size."
            )
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        mask_patches = self.to_patch(tokens)
        mask = self._unpatchify(mask_patches, grid_h, grid_w)
        mask = torch.sigmoid(mask) * self.mask_scale
        return self._crop(mask, original_shape)
