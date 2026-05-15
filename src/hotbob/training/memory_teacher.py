from __future__ import annotations

import torch

from hotbob.model import MemoryBank


def build_teacher_forced_memory(
    *,
    model_embed: torch.nn.Embedding,
    tokens: torch.Tensor,
    slot_ids: torch.Tensor,
    type_ids: torch.Tensor,
    scope_ids: torch.Tensor,
    privacy_ids: torch.Tensor,
    authority_ids: torch.Tensor,
    num_memory_slots: int,
    d_model: int,
    device: torch.device | str,
) -> MemoryBank:
    """Create a tensor memory bank from labelled memory ops without prompt text injection."""

    memory = MemoryBank(num_slots=num_memory_slots, d_model=d_model, device=device)
    memory.reset(tokens.shape[0])
    for row in range(tokens.shape[0]):
        vector = model_embed(tokens[row, -1]).detach()
        memory.apply_write(
            row,
            int(slot_ids[row].item()),
            vector,
            type_id=int(type_ids[row].item()),
            scope_id=int(scope_ids[row].item()),
            privacy_id=int(privacy_ids[row].item()),
            authority_id=int(authority_ids[row].item()),
        )
    return memory
