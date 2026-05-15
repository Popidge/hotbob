from __future__ import annotations

import torch
from torch import nn

from hotbob.model.memory_bank import MemoryBank
from hotbob.model.memory_read import MemoryRead
from hotbob.model.memory_write import MemoryWrite
from hotbob.model.transformer import TinyTransformer


class StatefulTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        action_vocab_size: int,
        d_model: int = 128,
        num_memory_slots: int = 32,
        num_types: int = 6,
        num_scopes: int = 64,
        num_privacy: int = 3,
        num_authority: int = 4,
    ) -> None:
        super().__init__()
        self.transformer = TinyTransformer(vocab_size, d_model=d_model)
        self.memory_read = MemoryRead(d_model)
        self.memory_write = MemoryWrite(
            d_model=d_model,
            num_slots=num_memory_slots,
            num_types=num_types,
            num_scopes=num_scopes,
            num_privacy=num_privacy,
            num_authority=num_authority,
        )
        self.action_head = nn.Linear(d_model, action_vocab_size)

    def forward(
        self, tokens: torch.Tensor, memory: MemoryBank, scope_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        hidden = self.transformer(tokens)
        hidden, _ = self.memory_read(hidden, memory, scope_ids)
        boundary_hidden = hidden[:, -1]
        outputs = self.memory_write(boundary_hidden)
        outputs["action_logits"] = self.action_head(boundary_hidden)
        return outputs
