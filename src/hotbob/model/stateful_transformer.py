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
        num_payload_kinds: int = 1,
        num_policy_actions: int = 1,
        num_policy_triggers: int = 1,
        num_expiry_policies: int = 1,
        num_authority_levels: int = 1,
        num_tool_names: int = 1,
        num_route_steps: int = 1,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.transformer = TinyTransformer(vocab_size, d_model=d_model, max_seq_len=max_seq_len)
        self.type_embedding = nn.Embedding(num_types, d_model)
        self.scope_embedding = nn.Embedding(num_scopes, d_model)
        self.privacy_embedding = nn.Embedding(num_privacy, d_model)
        self.authority_embedding = nn.Embedding(num_authority, d_model)
        self.payload_kind_embedding = nn.Embedding(num_payload_kinds, d_model)
        self.payload_default_action_embedding = nn.Embedding(num_policy_actions, d_model)
        self.payload_winning_authority_level_embedding = nn.Embedding(
            num_authority_levels,
            d_model,
        )
        self.payload_losing_authority_level_embedding = nn.Embedding(
            num_authority_levels,
            d_model,
        )
        self.memory_metadata_norm = nn.LayerNorm(d_model)
        self.memory_read = MemoryRead(d_model)
        self.memory_write = MemoryWrite(
            d_model=d_model,
            num_slots=num_memory_slots,
            num_types=num_types,
            num_scopes=num_scopes,
            num_privacy=num_privacy,
            num_authority=num_authority,
            num_value_classes=num_value_classes,
            num_payload_kinds=num_payload_kinds,
            num_policy_actions=num_policy_actions,
            num_policy_triggers=num_policy_triggers,
            num_expiry_policies=num_expiry_policies,
            num_authority_levels=num_authority_levels,
            num_tool_names=num_tool_names,
            num_route_steps=num_route_steps,
        )
        self.action_readout = ActionReadout(d_model, action_vocab_size)

    def _memory_for_read(self, memory: MemoryBank) -> MemoryBank:
        enriched = memory.clone_empty_like()
        type_ids = memory.type_ids.clamp(0, self.type_embedding.num_embeddings - 1)
        scope_ids = memory.scope_ids.clamp(0, self.scope_embedding.num_embeddings - 1)
        privacy_ids = memory.privacy_ids.clamp(0, self.privacy_embedding.num_embeddings - 1)
        authority_ids = memory.authority_ids.clamp(0, self.authority_embedding.num_embeddings - 1)
        payload_kind_ids = memory.payload_kind_ids.clamp(
            0,
            self.payload_kind_embedding.num_embeddings - 1,
        )
        payload_default_action_ids = memory.payload_default_action_ids.clamp(
            0,
            self.payload_default_action_embedding.num_embeddings - 1,
        )
        payload_winning_authority_level_ids = memory.payload_winning_authority_level_ids.clamp(
            0,
            self.payload_winning_authority_level_embedding.num_embeddings - 1,
        )
        payload_losing_authority_level_ids = memory.payload_losing_authority_level_ids.clamp(
            0,
            self.payload_losing_authority_level_embedding.num_embeddings - 1,
        )
        enriched.vectors = self.memory_metadata_norm(
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
        enriched.payload_winning_authority_level_ids = memory.payload_winning_authority_level_ids
        enriched.payload_losing_authority_level_ids = memory.payload_losing_authority_level_ids
        return enriched

    def forward(
        self,
        tokens: torch.Tensor,
        memory: MemoryBank,
        scope_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        padding_mask = tokens == 0
        hidden = self.transformer(tokens, padding_mask=padding_mask)
        read_memory = self._memory_for_read(memory)
        hidden, read_attention = self.memory_read(hidden, read_memory, scope_ids)
        if lengths is None:
            lengths = (~padding_mask).sum(dim=1)
        last_indices = (lengths - 1).clamp_min(0)
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        boundary_hidden = hidden[batch_indices, last_indices]
        boundary_read_attention = read_attention[batch_indices, last_indices]
        memory_context = torch.bmm(boundary_read_attention.unsqueeze(1), read_memory.vectors).squeeze(1)
        outputs = self.memory_write(boundary_hidden)
        action_logits, action_features = self.action_readout(boundary_hidden, memory_context)
        outputs["action_logits"] = action_logits
        outputs["action_features"] = action_features
        outputs["read_attention"] = read_attention
        outputs["boundary_indices"] = last_indices
        outputs["memory_context"] = memory_context
        return outputs
