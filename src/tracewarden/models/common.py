from __future__ import annotations

import math

import torch
import torch.nn as nn


class PosEnc(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d: int, n: int = 256):
        super().__init__()
        pe = torch.zeros(n, d)
        pos = torch.arange(n).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


def mlp(d: int, out: int, drop: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(drop), nn.Linear(d // 2, out))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
