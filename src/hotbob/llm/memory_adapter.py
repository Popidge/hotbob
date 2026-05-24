from __future__ import annotations

from enum import StrEnum

import torch
from torch import nn

from hotbob.model.memory_bank import MemoryBank
from hotbob.model.memory_read import MemoryRead
from hotbob.model.memory_write import MemoryWrite
from hotbob.training.dataset import (
    AUTHORITY_LEVEL_TO_ID,
    AUTHORITY_TO_ID,
    EXPIRY_POLICY_TO_ID,
    PAYLOAD_KIND_TO_ID,
    POLICY_ACTION_TO_ID,
    POLICY_TRIGGER_TO_ID,
    PRIVACY_TO_ID,
    TYPE_TO_ID,
    normalize_scope,
)
from hotbob.types import MemoryOp, MemoryOpName


class MemoryStateMode(StrEnum):
    SHARED = "shared"
    BY_TYPE = "by_type"


class MemoryPrefixAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        memory_prefix_len: int = 4,
        *,
        num_types: int = 1,
        num_scopes: int = 1,
        num_privacy: int = 1,
        num_authority: int = 1,
        num_payload_kinds: int = 1,
        num_policy_actions: int = 1,
        num_authority_levels: int = 1,
        metadata_mode: str = "none",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_prefix_len = memory_prefix_len
        self.metadata_mode = metadata_mode
        if metadata_mode not in {"none", "metadata"}:
            raise ValueError(f"Unknown prefix metadata mode: {metadata_mode}")
        if metadata_mode == "metadata":
            self.type_embedding = nn.Embedding(num_types, hidden_size)
            self.scope_embedding = nn.Embedding(num_scopes, hidden_size)
            self.privacy_embedding = nn.Embedding(num_privacy, hidden_size)
            self.authority_embedding = nn.Embedding(num_authority, hidden_size)
            self.payload_kind_embedding = nn.Embedding(num_payload_kinds, hidden_size)
            self.payload_default_action_embedding = nn.Embedding(
                num_policy_actions, hidden_size
            )
            self.payload_winning_authority_level_embedding = nn.Embedding(
                num_authority_levels, hidden_size
            )
            self.payload_losing_authority_level_embedding = nn.Embedding(
                num_authority_levels, hidden_size
            )
        self.read = MemoryRead(hidden_size)
        self.to_prefix = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * memory_prefix_len),
        )

    def _enriched_memory(self, memory: MemoryBank) -> MemoryBank:
        if self.metadata_mode == "none":
            return memory
        enriched = memory.clone_empty_like()
        type_ids = memory.type_ids.clamp(0, self.type_embedding.num_embeddings - 1)
        scope_ids = memory.scope_ids.clamp(0, self.scope_embedding.num_embeddings - 1)
        privacy_ids = memory.privacy_ids.clamp(0, self.privacy_embedding.num_embeddings - 1)
        authority_ids = memory.authority_ids.clamp(0, self.authority_embedding.num_embeddings - 1)
        payload_kind_ids = memory.payload_kind_ids.clamp(
            0, self.payload_kind_embedding.num_embeddings - 1
        )
        payload_default_action_ids = memory.payload_default_action_ids.clamp(
            0, self.payload_default_action_embedding.num_embeddings - 1
        )
        payload_winning_authority_level_ids = memory.payload_winning_authority_level_ids.clamp(
            0, self.payload_winning_authority_level_embedding.num_embeddings - 1
        )
        payload_losing_authority_level_ids = memory.payload_losing_authority_level_ids.clamp(
            0, self.payload_losing_authority_level_embedding.num_embeddings - 1
        )
        enriched.vectors = (
            memory.vectors
            + self.type_embedding(type_ids)
            + self.scope_embedding(scope_ids)
            + self.privacy_embedding(privacy_ids)
            + self.authority_embedding(authority_ids)
            + self.payload_kind_embedding(payload_kind_ids)
            + self.payload_default_action_embedding(payload_default_action_ids)
            + self.payload_winning_authority_level_embedding(payload_winning_authority_level_ids)
            + self.payload_losing_authority_level_embedding(payload_losing_authority_level_ids)
        )
        enriched.occupied = memory.occupied
        enriched.strength = memory.strength
        enriched.type_ids = memory.type_ids
        enriched.scope_ids = memory.scope_ids
        enriched.privacy_ids = memory.privacy_ids
        enriched.authority_ids = memory.authority_ids
        enriched.payload_kind_ids = memory.payload_kind_ids
        enriched.payload_default_action_ids = memory.payload_default_action_ids
        enriched.payload_winning_authority_level_ids = (
            memory.payload_winning_authority_level_ids
        )
        enriched.payload_losing_authority_level_ids = memory.payload_losing_authority_level_ids
        return enriched

    def forward(
        self,
        query_hidden: torch.Tensor,
        memory: MemoryBank,
        current_scope_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        read_hidden, attention = self.read(
            query_hidden.unsqueeze(1),
            self._enriched_memory(memory),
            current_scope_ids,
        )
        prefix = self.to_prefix(read_hidden[:, 0]).view(
            query_hidden.shape[0], self.memory_prefix_len, self.hidden_size
        )
        return prefix, attention[:, 0]


class LowRankMemoryCorrectionAdapter(nn.Module):
    """Memory-conditioned residual q/o correction over decoder hidden states.

    This is the first delta-HotBob integration path. It is intentionally shaped
    like q/o low-rank correction while staying outside Qwen attention internals:
    q-side correction is added to token inputs before the frozen decoder, and
    o-side correction is added to decoder hidden states before the LM head.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        rank: int = 16,
        num_types: int,
        num_scopes: int,
        num_privacy: int,
        num_authority: int,
        state_mode: MemoryStateMode | str = MemoryStateMode.SHARED,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = min(rank, hidden_size)
        self.state_mode = MemoryStateMode(state_mode)
        self.type_embedding = nn.Embedding(num_types, hidden_size)
        self.scope_embedding = nn.Embedding(num_scopes, hidden_size)
        self.privacy_embedding = nn.Embedding(num_privacy, hidden_size)
        self.authority_embedding = nn.Embedding(num_authority, hidden_size)
        self.read = MemoryRead(hidden_size)
        self.q_down = nn.Linear(hidden_size, self.rank, bias=False)
        self.q_up = nn.Linear(self.rank, hidden_size, bias=False)
        self.o_down = nn.Linear(hidden_size, self.rank, bias=False)
        self.o_up = nn.Linear(self.rank, hidden_size, bias=False)
        self.q_gate = nn.Parameter(torch.tensor(0.1))
        self.o_gate = nn.Parameter(torch.tensor(0.1))

    def _enriched_memory(self, memory: MemoryBank) -> MemoryBank:
        enriched = memory.clone_empty_like()
        type_ids = memory.type_ids.clamp(0, self.type_embedding.num_embeddings - 1)
        scope_ids = memory.scope_ids.clamp(0, self.scope_embedding.num_embeddings - 1)
        privacy_ids = memory.privacy_ids.clamp(0, self.privacy_embedding.num_embeddings - 1)
        authority_ids = memory.authority_ids.clamp(0, self.authority_embedding.num_embeddings - 1)
        enriched.vectors = (
            memory.vectors
            + self.type_embedding(type_ids)
            + self.scope_embedding(scope_ids)
            + self.privacy_embedding(privacy_ids)
            + self.authority_embedding(authority_ids)
        )
        enriched.occupied = memory.occupied
        enriched.strength = memory.strength
        enriched.type_ids = memory.type_ids
        enriched.scope_ids = memory.scope_ids
        enriched.privacy_ids = memory.privacy_ids
        enriched.authority_ids = memory.authority_ids
        return enriched

    def _typed_grouped_memory(
        self, memory: MemoryBank, current_scope_ids: torch.Tensor
    ) -> MemoryBank:
        active = memory.active_mask(current_scope_ids)
        num_types = self.type_embedding.num_embeddings
        grouped = MemoryBank(num_types, memory.d_model, device=memory.device)
        grouped.reset(memory.vectors.shape[0])
        grouped.type_ids = torch.arange(num_types, device=memory.device).unsqueeze(0).expand_as(
            grouped.type_ids
        )
        grouped.scope_ids = current_scope_ids.to(memory.device).unsqueeze(-1).expand_as(
            grouped.scope_ids
        )
        for type_id in range(num_types):
            type_mask = active & (memory.type_ids == type_id)
            weights = torch.where(type_mask, memory.strength, torch.zeros_like(memory.strength))
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            grouped.vectors[:, type_id] = (memory.vectors * weights.unsqueeze(-1)).sum(
                dim=1
            ) / denom
            grouped.strength[:, type_id] = weights.sum(dim=1).clamp_max(1.0)
            grouped.occupied[:, type_id] = type_mask.any(dim=1)
            grouped.privacy_ids[:, type_id] = torch.where(
                type_mask,
                memory.privacy_ids,
                torch.zeros_like(memory.privacy_ids),
            ).amax(dim=1)
            grouped.authority_ids[:, type_id] = torch.where(
                type_mask,
                memory.authority_ids,
                torch.zeros_like(memory.authority_ids),
            ).amax(dim=1)
        return grouped

    def _readout(
        self,
        hidden: torch.Tensor,
        memory: MemoryBank,
        current_scope_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = (
            self._typed_grouped_memory(memory, current_scope_ids)
            if self.state_mode == MemoryStateMode.BY_TYPE
            else memory
        )
        enriched = self._enriched_memory(source)
        read_hidden, attention = self.read(hidden, enriched, current_scope_ids)
        has_active = source.active_mask(current_scope_ids).any(dim=-1).view(hidden.shape[0], 1, 1)
        readout = torch.where(has_active, read_hidden - hidden, torch.zeros_like(hidden))
        return readout, attention

    def forward(
        self,
        hidden: torch.Tensor,
        memory: MemoryBank,
        current_scope_ids: torch.Tensor,
        *,
        apply_q: bool,
        apply_o: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        q_attention = None
        o_attention = None
        if apply_q:
            readout, q_attention = self._readout(hidden, memory, current_scope_ids)
            hidden = hidden + self.q_gate.to(hidden.dtype) * self.q_up(self.q_down(readout))
        if apply_o:
            readout, o_attention = self._readout(hidden, memory, current_scope_ids)
            hidden = hidden + self.o_gate.to(hidden.dtype) * self.o_up(self.o_down(readout))
        return hidden, q_attention, o_attention


class LQwenMemoryHeads(nn.Module):
    """Trainable memory heads around a frozen decoder hidden state."""

    def __init__(
        self,
        hidden_size: int,
        num_memory_slots: int,
        num_scopes: int,
        num_value_classes: int,
        num_tool_names: int = 1,
        num_route_steps: int = 8,
        memory_prefix_len: int = 4,
        correction_rank: int = 16,
        memory_state_mode: MemoryStateMode | str = MemoryStateMode.SHARED,
        memory_prefix_metadata_mode: str = "none",
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
            max(PAYLOAD_KIND_TO_ID.values(), default=0) + 1,
            max(POLICY_ACTION_TO_ID.values(), default=0) + 1,
            max(POLICY_TRIGGER_TO_ID.values(), default=0) + 1,
            max(EXPIRY_POLICY_TO_ID.values(), default=0) + 1,
            max(AUTHORITY_LEVEL_TO_ID.values(), default=0) + 1,
            num_tool_names,
            num_route_steps,
        )
        self.value_projector = nn.Linear(hidden_size, hidden_size)
        self.prefix = MemoryPrefixAdapter(
            hidden_size,
            memory_prefix_len,
            num_types=len(TYPE_TO_ID),
            num_scopes=num_scopes,
            num_privacy=len(PRIVACY_TO_ID),
            num_authority=len(AUTHORITY_TO_ID),
            num_payload_kinds=max(PAYLOAD_KIND_TO_ID.values(), default=0) + 1,
            num_policy_actions=max(POLICY_ACTION_TO_ID.values(), default=0) + 1,
            num_authority_levels=max(AUTHORITY_LEVEL_TO_ID.values(), default=0) + 1,
            metadata_mode=memory_prefix_metadata_mode,
        )
        self.correction = LowRankMemoryCorrectionAdapter(
            hidden_size,
            rank=correction_rank,
            num_types=len(TYPE_TO_ID),
            num_scopes=num_scopes,
            num_privacy=len(PRIVACY_TO_ID),
            num_authority=len(AUTHORITY_TO_ID),
            state_mode=memory_state_mode,
        )


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
        structured = structured_memory_metadata(op)
        memory.payload_kind_ids[0, slot_idx] = structured["payload_kind_id"]
        memory.payload_default_action_ids[0, slot_idx] = structured["default_action_id"]
        memory.payload_winning_authority_level_ids[0, slot_idx] = structured[
            "winning_authority_level_id"
        ]
        memory.payload_losing_authority_level_ids[0, slot_idx] = structured[
            "losing_authority_level_id"
        ]
        return
    structured = structured_memory_metadata(op)
    memory.apply_write(
        0,
        slot_idx,
        vector,
        type_id=TYPE_TO_ID[op.type],
        scope_id=scope_id(op.scope, scope_vocab),
        privacy_id=PRIVACY_TO_ID[op.privacy],
        authority_id=AUTHORITY_TO_ID[op.authority],
        payload_kind_id=structured["payload_kind_id"],
        payload_default_action_id=structured["default_action_id"],
        payload_winning_authority_level_id=structured["winning_authority_level_id"],
        payload_losing_authority_level_id=structured["losing_authority_level_id"],
    )


def structured_memory_metadata(op: MemoryOp) -> dict[str, int]:
    from hotbob.training.dataset import structured_targets_from_payload

    structured = structured_targets_from_payload(op.payload)
    return {
        "payload_kind_id": int(structured["payload_kind_id"]),
        "default_action_id": int(structured["default_action_id"]),
        "winning_authority_level_id": int(structured["winning_authority_level_id"]),
        "losing_authority_level_id": int(structured["losing_authority_level_id"]),
    }
