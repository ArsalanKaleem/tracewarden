"""StreamGuard - novel addition N1: a CAUSAL model over interleaved ACTION/OBS events.

At each ACTION event it scores "is this proposed call serving an attacker?" using only the past,
which is the moment a firewall can still block the call. At each OBS event it scores "does this
observation carry an injection?". Whether a poisoned observation was resisted or obeyed is not
knowable when it arrives; it emerges from later ACTION scores.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import PosEnc, mlp


class StreamGuard(nn.Module):
    def __init__(self, d_in: int = 783, d: int = 256, layers: int = 3, heads: int = 4, ff: int = 512,
                 drop: float = 0.1, max_events: int = 256):
        super().__init__()
        self.config = dict(d_in=d_in, d=d, layers=layers, heads=heads, ff=ff, drop=drop, max_events=max_events)
        self.type_emb = nn.Embedding(2, d)
        self.proj, self.pos, self.drop = nn.Linear(d_in, d), PosEnc(d, max_events), nn.Dropout(drop)
        layer = nn.TransformerEncoderLayer(d, heads, ff, drop, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.hijack_head, self.poison_head = mlp(d, 1, drop), mlp(d, 1, drop)

    def forward(self, x: torch.Tensor, etype: torch.Tensor, mask: torch.Tensor):
        """x: (B,E,d_in), etype: (B,E), mask: (B,E). Returns (hijack_logits (B,E), poison_logits (B,E))."""
        E = x.size(1)
        h = self.drop(self.pos(self.proj(x) + self.type_emb(etype)))
        causal = torch.triu(torch.ones(E, E, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.enc(h, mask=causal, src_key_padding_mask=~mask)
        return self.hijack_head(h).squeeze(-1), self.poison_head(h).squeeze(-1)
