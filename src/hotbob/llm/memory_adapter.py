from __future__ import annotations

import torch
from torch import nn

from hotbob.model.memory_bank import MemoryBank
from hotbob.model.memory_read import MemoryRead
from hotbob.model.memory_write import MemoryWrite
from hotbob.training.dataset import (
    AUTHORITY_TO_ID,
    PRIVACY_TO_ID,
    TYPE_TO_ID,
    normalize_scope,
)
from hotbob.types import MemoryOp, MemoryOpName


class MemoryPrefixAdapter(nn.Module):
    def __init__(self, hidden_size: int, memory_prefix_len: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_prefix_len = memory_prefix_len
        self.read = MemoryRead(hidden_size)
        self.to_prefix = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * memory_prefix_len),
        )

    def forward(
        self,
        query_hidden: torch.Tensor,
        memory: MemoryBank,
        current_scope_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        read_hidden, attention = self.read(query_hidden.unsqueeze(1), memory, current_scope_ids)
        prefix = self.to_prefix(read_hidden[:, 0]).view(
            query_hidden.shape[0], self.memory_prefix_len, self.hidden_size
        )
        return prefix, attention[:, 0]


class LQwenMemoryHeads(nn.Module):
    """Trainable memory heads around a frozen decoder hidden state."""

    def __init__(
        self,
        hidden_size: int,
        num_memory_slots: int,
        num_scopes: int,
        num_value_classes: int,
        memory_prefix_len: int = 4,
    ) -> None:
        super().__init__()
        self.writer = MemoryWrite(
            hidden_size,
            num_memory_slots,
            len(TYPE_TO_ID),
            num_scopes,
            len(PRIVACY_TO_ID),
            len(AUTHORITY_TO_ID),
            num_value_classes,
        )
        self.value_projector = nn.Linear(hidden_size, hidden_size)
        self.prefix = MemoryPrefixAdapter(hidden_size, memory_prefix_len)


def scope_id(scope: str, scope_vocab: dict[str, int]) -> int:
    return scope_vocab.get(normalize_scope(scope), 0)


def apply_teacher_memory_op(
    memory: MemoryBank,
    op: MemoryOp,
    vector: torch.Tensor,
    *,
    slot_idx: int,
    scope_vocab: dict[str, int],
) -> None:
    slot_idx = min(slot_idx, memory.num_slots - 1)
    if op.op == MemoryOpName.DELETE:
        target_scope = scope_id(op.scope, scope_vocab)
        target_type = TYPE_TO_ID[op.type]
        for slot in range(memory.num_slots):
            if (
                bool(memory.occupied[0, slot].item())
                and int(memory.scope_ids[0, slot].item()) == target_scope
                and int(memory.type_ids[0, slot].item()) == target_type
            ):
                memory.apply_delete(0, slot)
        return
    if op.op == MemoryOpName.UPDATE and bool(memory.occupied[0, slot_idx].item()):
        memory.apply_update(0, slot_idx, vector)
        return
    memory.apply_write(
        0,
        slot_idx,
        vector,
        type_id=TYPE_TO_ID[op.type],
        scope_id=scope_id(op.scope, scope_vocab),
        privacy_id=PRIVACY_TO_ID[op.privacy],
        authority_id=AUTHORITY_TO_ID[op.authority],
    )
