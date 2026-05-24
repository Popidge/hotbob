from __future__ import annotations

import torch

from hotbob.model.memory_bank import MemoryBank
from hotbob.model.stateful_transformer import StatefulTransformer


def test_memory_metadata_changes_action_features_for_same_value_vector() -> None:
    torch.manual_seed(0)
    model = StatefulTransformer(
        vocab_size=16,
        action_vocab_size=4,
        d_model=8,
        num_memory_slots=2,
        num_types=3,
        num_scopes=3,
        num_privacy=2,
        num_authority=3,
        num_payload_kinds=4,
        num_policy_actions=5,
        num_authority_levels=6,
        max_seq_len=8,
    )
    model.eval()
    tokens = torch.tensor([[2, 3, 4]], dtype=torch.long)
    lengths = torch.tensor([3], dtype=torch.long)
    scope_ids = torch.tensor([1], dtype=torch.long)
    vector = torch.ones(8)

    memory_a = MemoryBank(num_slots=2, d_model=8)
    memory_a.reset(1)
    memory_a.apply_write(
        0,
        0,
        vector,
        type_id=1,
        scope_id=1,
        privacy_id=1,
        authority_id=1,
        payload_kind_id=1,
        payload_default_action_id=1,
        payload_winning_authority_level_id=1,
        payload_losing_authority_level_id=2,
    )
    memory_b = MemoryBank(num_slots=2, d_model=8)
    memory_b.reset(1)
    memory_b.apply_write(
        0,
        0,
        vector,
        type_id=1,
        scope_id=1,
        privacy_id=1,
        authority_id=1,
        payload_kind_id=1,
        payload_default_action_id=3,
        payload_winning_authority_level_id=4,
        payload_losing_authority_level_id=5,
    )

    with torch.no_grad():
        outputs_a = model(tokens, memory_a, scope_ids, lengths)
        outputs_b = model(tokens, memory_b, scope_ids, lengths)

    assert not torch.allclose(outputs_a["memory_context"], outputs_b["memory_context"])
    assert not torch.allclose(outputs_a["action_features"], outputs_b["action_features"])
