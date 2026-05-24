from __future__ import annotations

from pathlib import Path

import torch

from hotbob.model import MemoryBank
from hotbob.model.stateful_transformer import StatefulTransformer
from hotbob.training.dataset import (
    AUTHORITY_TO_ID,
    PRIVACY_TO_ID,
    TYPE_TO_ID,
    TraceDataset,
    TraceVocab,
    memory_text_from_op,
    normalize_scope,
    tokenize_text,
)
from hotbob.training.memory_teacher import mean_value_embedding
from hotbob.types import MemoryOpName, TaskTrace, TraceEvent


def load_model_and_dataset(
    checkpoint_path: str,
    traces: list[TaskTrace],
    device: str,
) -> tuple[StatefulTransformer, TraceDataset, dict] | None:
    if not Path(checkpoint_path).exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = TraceVocab.from_token_to_id(checkpoint["vocab"])
    dataset = TraceDataset(
        traces,
        vocab=vocab,
        scope_vocab=checkpoint["scope_vocab"],
        value_vocab=checkpoint.get("value_vocab"),
        tool_name_vocab=checkpoint.get("tool_name_vocab"),
    )
    config = checkpoint["config"]
    model = StatefulTransformer(
        vocab_size=len(vocab),
        action_vocab_size=config["action_vocab_size"],
        d_model=config["d_model"],
        num_memory_slots=config["num_memory_slots"],
        num_types=len(TYPE_TO_ID),
        num_scopes=max(dataset.scope_vocab.values()) + 1,
        num_privacy=len(PRIVACY_TO_ID),
        num_authority=len(AUTHORITY_TO_ID),
        num_value_classes=config.get("num_value_classes", 1),
        num_payload_kinds=config.get("num_payload_kinds", 1),
        num_policy_actions=config.get("num_policy_actions", 1),
        num_policy_triggers=config.get("num_policy_triggers", 1),
        num_expiry_policies=config.get("num_expiry_policies", 1),
        num_authority_levels=config.get("num_authority_levels", 1),
        num_tool_names=config.get("num_tool_names", 1),
        num_route_steps=config.get("num_route_steps", 8),
        max_seq_len=config.get("max_seq_len", 256),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model, dataset, config


def encode_event_tensor(
    dataset: TraceDataset,
    event: TraceEvent,
    current_scope: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = dataset._encode_event(event, current_scope)[-dataset.max_seq_len :]
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=device)
    return tokens, lengths


def value_vector_for_op(
    model: StatefulTransformer,
    dataset: TraceDataset,
    op,
    device: str,
) -> torch.Tensor:
    value_ids = dataset.vocab.encode(tokenize_text(memory_text_from_op(op))) or [0]
    value_tokens = torch.tensor([value_ids], dtype=torch.long, device=device)
    value_mask = torch.ones_like(value_tokens, dtype=torch.bool)
    return mean_value_embedding(model.transformer.embed, value_tokens, value_mask)[0].detach()


def apply_teacher_op(
    memory: MemoryBank,
    model: StatefulTransformer,
    dataset: TraceDataset,
    trace: TaskTrace,
    op_index: int,
    device: str,
) -> None:
    op = trace.expected_memory_ops[op_index]
    slot_idx = min(op_index, memory.num_slots - 1)
    if op.op == MemoryOpName.DELETE:
        for slot in range(memory.num_slots):
            if (
                bool(memory.occupied[0, slot].item())
                and int(memory.scope_ids[0, slot].item())
                == dataset.scope_vocab.get(normalize_scope(op.scope), -1)
                and int(memory.type_ids[0, slot].item()) == TYPE_TO_ID[op.type]
            ):
                memory.apply_delete(0, slot)
                return
        return
    vector = value_vector_for_op(model, dataset, op, device)
    if op.op == MemoryOpName.UPDATE and bool(memory.occupied[0, slot_idx].item()):
        memory.apply_update(0, slot_idx, vector)
        return
    memory.apply_write(
        0,
        slot_idx,
        vector,
        type_id=TYPE_TO_ID[op.type],
        scope_id=dataset.scope_vocab.get(normalize_scope(op.scope), 0),
        privacy_id=PRIVACY_TO_ID[op.privacy],
        authority_id=AUTHORITY_TO_ID[op.authority],
    )


def apply_predicted_op(*args, **kwargs) -> None:
    from hotbob.training.evaluate import apply_predicted_op as _apply_predicted_op

    _apply_predicted_op(*args, **kwargs)
