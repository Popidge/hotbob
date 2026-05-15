from __future__ import annotations

import torch
from torch import nn

from hotbob.model.memory_bank import MemoryBank
from hotbob.model.memory_read import MemoryRead
from hotbob.model.transformer import TinyTransformer


class StatefulTransformer(nn.Module):
    def __init__(self, vocab_size: int, action_vocab_size: int, d_model: int = 128) -> None:
        super().__init__()
        self.transformer = TinyTransformer(vocab_size, d_model=d_model)
        self.memory_read = MemoryRead(d_model)
        self.action_head = nn.Linear(d_model, action_vocab_size)

    def forward(
        self, tokens: torch.Tensor, memory: MemoryBank, scope_ids: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.transformer(tokens)
        hidden, _ = self.memory_read(hidden, memory, scope_ids)
        return self.action_head(hidden[:, -1])
