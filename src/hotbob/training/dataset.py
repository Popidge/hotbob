from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from hotbob.types import (
    ActionLabel,
    MemoryAuthority,
    MemoryOpName,
    MemoryPrivacy,
    MemoryType,
    TaskTrace,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")

PAD = "<pad>"
UNK = "<unk>"


def tokenize_text(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class TraceVocab:
    def __init__(self, tokens: Sequence[str] = ()) -> None:
        self.token_to_id = {PAD: 0, UNK: 1}
        self.id_to_token = [PAD, UNK]
        for token in tokens:
            self.add(token)

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)
        return self.token_to_id[token]

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self.token_to_id.get(token, self.token_to_id[UNK]) for token in tokens]

    def __len__(self) -> int:
        return len(self.id_to_token)

    @classmethod
    def from_token_to_id(cls, token_to_id: dict[str, int]) -> TraceVocab:
        vocab = cls()
        vocab.token_to_id = dict(token_to_id)
        vocab.id_to_token = [""] * len(token_to_id)
        for token, idx in token_to_id.items():
            vocab.id_to_token[idx] = token
        return vocab


def build_vocab(traces: Sequence[TaskTrace]) -> TraceVocab:
    vocab = TraceVocab()
    for trace in traces:
        for event in trace.events:
            vocab.add(f"role_{event.role.lower()}")
            vocab.add(f"scope_{(event.scope or trace.current_scope).lower()}")
            for token in tokenize_text(event.content):
                vocab.add(token)
    return vocab


def build_scope_vocab(traces: Sequence[TaskTrace]) -> dict[str, int]:
    scopes = sorted(
        {event.scope or trace.current_scope for trace in traces for event in trace.events}
        | {trace.current_scope for trace in traces}
        | {op.scope for trace in traces for op in trace.expected_memory_ops}
    )
    return {scope: idx + 1 for idx, scope in enumerate(scopes)}


@dataclass(frozen=True)
class EncodedTrace:
    write_tokens: list[int]
    action_tokens: list[int]
    memory_value_tokens: list[int]
    current_scope_id: int
    action_id: int
    op_id: int
    type_id: int
    scope_id: int
    privacy_id: int
    authority_id: int
    slot_id: int


ACTION_TO_ID = {label: idx for idx, label in enumerate(ActionLabel)}
OP_TO_ID = {label: idx for idx, label in enumerate(MemoryOpName)}
TYPE_TO_ID = {label: idx for idx, label in enumerate(MemoryType)}
PRIVACY_TO_ID = {label: idx for idx, label in enumerate(MemoryPrivacy)}
AUTHORITY_TO_ID = {label: idx for idx, label in enumerate(MemoryAuthority)}


class TraceDataset(Dataset[EncodedTrace]):
    def __init__(
        self,
        traces: list[TaskTrace],
        vocab: TraceVocab | None = None,
        scope_vocab: dict[str, int] | None = None,
        max_seq_len: int = 256,
    ) -> None:
        self.traces = traces
        self.vocab = vocab or build_vocab(traces)
        self.scope_vocab = scope_vocab or build_scope_vocab(traces)
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> EncodedTrace:
        trace = self.traces[idx]
        first_op = self._select_supervised_op(trace)
        write_event = next(
            (
                event
                for event in trace.events
                if (event.scope or trace.current_scope) == first_op.scope
            ),
            trace.events[0],
        )
        action_event = trace.events[-1]
        write_tokens = self._encode_event(write_event, trace.current_scope)
        action_tokens = self._encode_event(action_event, trace.current_scope)
        memory_value_tokens = self.vocab.encode(tokenize_text(first_op.value))
        return EncodedTrace(
            write_tokens=write_tokens[-self.max_seq_len :],
            action_tokens=action_tokens[-self.max_seq_len :],
            memory_value_tokens=memory_value_tokens,
            current_scope_id=self.scope_vocab[trace.current_scope],
            action_id=ACTION_TO_ID[trace.expected_final_action],
            op_id=OP_TO_ID[first_op.op],
            type_id=TYPE_TO_ID[first_op.type],
            scope_id=self.scope_vocab[first_op.scope],
            privacy_id=PRIVACY_TO_ID[first_op.privacy],
            authority_id=AUTHORITY_TO_ID[first_op.authority],
            slot_id=0,
        )

    def _encode_event(self, event, current_scope: str) -> list[int]:
        tokens = [
            f"role_{event.role.lower()}",
            f"scope_{(event.scope or current_scope).lower()}",
            *tokenize_text(event.content),
        ]
        return self.vocab.encode(tokens)

    def _select_supervised_op(self, trace: TaskTrace):
        scoped_ops = [op for op in trace.expected_memory_ops if op.scope == trace.current_scope]
        if scoped_ops:
            return scoped_ops[0]
        return trace.expected_memory_ops[0]


def collate_traces(batch: Sequence[EncodedTrace]) -> dict[str, torch.Tensor]:
    max_action_len = max(len(item.action_tokens) for item in batch)
    max_write_len = max(len(item.write_tokens) for item in batch)
    max_value_len = max(len(item.memory_value_tokens) for item in batch)
    tokens = torch.zeros((len(batch), max_action_len), dtype=torch.long)
    write_tokens = torch.zeros((len(batch), max_write_len), dtype=torch.long)
    memory_value_tokens = torch.zeros((len(batch), max_value_len), dtype=torch.long)
    memory_value_mask = torch.zeros((len(batch), max_value_len), dtype=torch.bool)
    for i, item in enumerate(batch):
        tokens[i, : len(item.action_tokens)] = torch.tensor(item.action_tokens, dtype=torch.long)
        write_tokens[i, : len(item.write_tokens)] = torch.tensor(
            item.write_tokens, dtype=torch.long
        )
        memory_value_tokens[i, : len(item.memory_value_tokens)] = torch.tensor(
            item.memory_value_tokens, dtype=torch.long
        )
        memory_value_mask[i, : len(item.memory_value_tokens)] = True
    return {
        "tokens": tokens,
        "lengths": torch.tensor([len(item.action_tokens) for item in batch], dtype=torch.long),
        "write_tokens": write_tokens,
        "write_lengths": torch.tensor([len(item.write_tokens) for item in batch], dtype=torch.long),
        "memory_value_tokens": memory_value_tokens,
        "memory_value_mask": memory_value_mask,
        "current_scope_ids": torch.tensor(
            [item.current_scope_id for item in batch], dtype=torch.long
        ),
        "action_ids": torch.tensor([item.action_id for item in batch], dtype=torch.long),
        "op_ids": torch.tensor([item.op_id for item in batch], dtype=torch.long),
        "type_ids": torch.tensor([item.type_id for item in batch], dtype=torch.long),
        "scope_ids": torch.tensor([item.scope_id for item in batch], dtype=torch.long),
        "privacy_ids": torch.tensor([item.privacy_id for item in batch], dtype=torch.long),
        "authority_ids": torch.tensor([item.authority_id for item in batch], dtype=torch.long),
        "slot_ids": torch.tensor([item.slot_id for item in batch], dtype=torch.long),
    }
