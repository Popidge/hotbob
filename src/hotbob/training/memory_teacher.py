from __future__ import annotations

import torch

from hotbob.model import MemoryBank


def mean_value_embedding(
    model_embed: torch.nn.Embedding,
    memory_value_tokens: torch.Tensor,
    memory_value_mask: torch.Tensor,
) -> torch.Tensor:
    embedded = model_embed(memory_value_tokens)
    mask = memory_value_mask.unsqueeze(-1)
    summed = (embedded * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1)
    return summed / counts


def build_teacher_forced_memory(
    *,
    model_embed: torch.nn.Embedding,
    memory_value_tokens: torch.Tensor,
    memory_value_mask: torch.Tensor,
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
    memory.reset(memory_value_tokens.shape[0])
    value_vectors = mean_value_embedding(
        model_embed,
        memory_value_tokens,
        memory_value_mask,
    ).detach()
    for row in range(memory_value_tokens.shape[0]):
        memory.apply_write(
            row,
            int(slot_ids[row].item()),
            value_vectors[row],
            type_id=int(type_ids[row].item()),
            scope_id=int(scope_ids[row].item()),
            privacy_id=int(privacy_ids[row].item()),
            authority_id=int(authority_ids[row].item()),
        )
    return memory


def build_predicted_memory(
    *,
    outputs: dict[str, torch.Tensor],
    batch_size: int,
    num_memory_slots: int,
    d_model: int,
    device: torch.device | str,
) -> MemoryBank:
    """Create a memory bank from model-predicted write heads for evaluation."""

    memory = MemoryBank(num_slots=num_memory_slots, d_model=d_model, device=device)
    memory.reset(batch_size)
    slot_ids = outputs["slot_logits"].argmax(dim=-1)
    type_ids = outputs["type_logits"].argmax(dim=-1)
    scope_ids = outputs["scope_logits"].argmax(dim=-1)
    privacy_ids = outputs["privacy_logits"].argmax(dim=-1)
    authority_ids = outputs["authority_logits"].argmax(dim=-1)
    gates = outputs["write_gate"].squeeze(-1)
    for row in range(batch_size):
        memory.apply_write(
            row,
            int(slot_ids[row].item()),
            outputs["value_vector"][row].detach(),
            type_id=int(type_ids[row].item()),
            scope_id=int(scope_ids[row].item()),
            privacy_id=int(privacy_ids[row].item()),
            authority_id=int(authority_ids[row].item()),
            strength=float(gates[row].item()),
        )
    return memory
