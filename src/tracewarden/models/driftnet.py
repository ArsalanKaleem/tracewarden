"""Re-implementation of DriftNet (arXiv 2609.10892): a bidirectional dual-head trajectory Transformer
over frozen step embeddings + 4 world features. Reads a FINISHED trajectory (post-hoc)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import PosEnc, mlp


class DriftNet(nn.Module):
    def __init__(self, d_in: int = 772, d: int = 256, layers: int = 2, heads: int = 4, ff: int = 512,
                 drop: float = 0.1):
        super().__init__()
        self.config = dict(d_in=d_in, d=d, layers=layers, heads=heads, ff=ff, drop=drop)
        self.proj, self.pos, self.drop = nn.Linear(d_in, d), PosEnc(d), nn.Dropout(drop)
        layer = nn.TransformerEncoderLayer(d, heads, ff, drop, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.traj_head, self.step_head = mlp(d, 1, drop), mlp(d, 4, drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """x: (B,T,d_in), mask: (B,T) True for real steps. Returns (traj_logit (B,), step_logits (B,T,4))."""
        h = self.drop(self.pos(self.proj(x)))
        h = self.enc(h, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.traj_head(pooled).squeeze(-1), self.step_head(h)
