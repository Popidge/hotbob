from __future__ import annotations

import torch
from torch import nn

from hotbob.model.action_readout import ActionReadout
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
        num_value_classes: int = 1,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.transformer = TinyTransformer(vocab_size, d_model=d_model, max_seq_len=max_seq_len)
        self.memory_read = MemoryRead(d_model)
        self.memory_write = MemoryWrite(
            d_model=d_model,
            num_slots=num_memory_slots,
            num_types=num_types,
            num_scopes=num_scopes,
            num_privacy=num_privacy,
            num_authority=num_authority,
            num_value_classes=num_value_classes,
        )
        self.action_readout = ActionReadout(d_model, action_vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        memory: MemoryBank,
        scope_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        padding_mask = tokens == 0
        hidden = self.transformer(tokens, padding_mask=padding_mask)
        hidden, read_attention = self.memory_read(hidden, memory, scope_ids)
        if lengths is None:
            lengths = (~padding_mask).sum(dim=1)
        last_indices = (lengths - 1).clamp_min(0)
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        boundary_hidden = hidden[batch_indices, last_indices]
        boundary_read_attention = read_attention[batch_indices, last_indices]
        memory_context = torch.bmm(boundary_read_attention.unsqueeze(1), memory.vectors).squeeze(1)
        outputs = self.memory_write(boundary_hidden)
        action_logits, action_features = self.action_readout(boundary_hidden, memory_context)
        outputs["action_logits"] = action_logits
        outputs["action_features"] = action_features
        outputs["read_attention"] = read_attention
        outputs["boundary_indices"] = last_indices
        outputs["memory_context"] = memory_context
        return outputs
