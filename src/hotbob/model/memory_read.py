from __future__ import annotations

import math

import torch
from torch import nn

from hotbob.model.memory_bank import MemoryBank


class MemoryRead(nn.Module):
    """Cross-attention read from scoped tensor memory into token hidden states."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(
        self, hidden: torch.Tensor, memory: MemoryBank, current_scope: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.wq(hidden)
        k = self.wk(memory.vectors)
        v = self.wv(memory.vectors)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hidden.shape[-1])
        mask = memory.active_mask(current_scope).unsqueeze(1)
        scores = scores + torch.log(memory.strength.clamp_min(1e-6)).unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(mask.any(dim=-1, keepdim=True), attn, torch.zeros_like(attn))
        ctx = torch.matmul(attn, v)
        return hidden + self.wo(ctx), attn
