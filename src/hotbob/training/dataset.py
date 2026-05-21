from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from hotbob.types import (
    ActionLabel,
    AuthorityLevel,
    AuthorityRulePayload,
    DisclosureRulePayload,
    ExpiryPolicy,
    ExpiryRulePayload,
    MemoryAuthority,
    MemoryOpName,
    MemoryPayload,
    MemoryPayloadKind,
    MemoryPrivacy,
    MemoryType,
    PolicyAction,
    PolicyTrigger,
    StandingOrderPayload,
    TaskStatePayload,
    TaskTrace,
    ToolFactPayload,
    ToolRoutingCommitmentPayload,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")

PAD = "<pad>"
UNK = "<unk>"


def tokenize_text(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize_scope(scope: str) -> str:
    """Map generated one-off scope ids to reusable scope classes."""

    lower = scope.lower()
    if lower.startswith("game_"):
        return "game"
    if lower.startswith("mission_") and lower.endswith("_old"):
        return "mission_old"
    if lower.startswith("mission_") and lower.endswith("_new"):
        return "mission_new"
    if lower.startswith("mission_"):
        return "mission"
    if lower.startswith("repo_") and lower.endswith("_a"):
        return "repo_a"
    if lower.startswith("repo_") and lower.endswith("_b"):
        return "repo_b"
    if lower.startswith("repo_"):
        return "repo"
    return lower


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
            vocab.add(f"scope_{normalize_scope(event.scope or trace.current_scope)}")
            for token in tokenize_text(event.content):
                vocab.add(token)
        for op in trace.expected_memory_ops:
            for token in tokenize_text(memory_text_from_op(op)):
                vocab.add(token)
    return vocab


def memory_text_from_op(op) -> str:
    payload = op.payload
    parts = [op.value, str(payload.kind)]
    for field in (
        "default_action",
        "trigger",
        "authority_level",
        "expiry_policy",
        "active_until",
        "expired_action",
        "winning_authority",
        "losing_authority",
        "conflict_action",
        "tool_name",
        "claim_key",
        "claim_value",
        "next_required_action",
        "safe_response",
        "next_tool",
        "final_action_after_tools",
    ):
        value = getattr(payload, field, None)
        if value is not None:
            parts.append(str(value))
    for field in (
        "allowed_responses",
        "forbidden_responses",
        "exceptions",
        "allowed_audience",
        "forbidden_audience",
        "committed_tool_sequence",
        "completed_tools",
    ):
        values = getattr(payload, field, None)
        if values:
            parts.extend(str(value) for value in values)
    return " ".join(parts)


def build_scope_vocab(traces: Sequence[TaskTrace]) -> dict[str, int]:
    scopes = sorted(
        {
            normalize_scope(event.scope or trace.current_scope)
            for trace in traces
            for event in trace.events
        }
        | {normalize_scope(trace.current_scope) for trace in traces}
        | {normalize_scope(op.scope) for trace in traces for op in trace.expected_memory_ops}
    )
    return {scope: idx + 1 for idx, scope in enumerate(scopes)}


def build_value_vocab(traces: Sequence[TaskTrace]) -> dict[str, int]:
    values = sorted({op.value for trace in traces for op in trace.expected_memory_ops})
    return {value: idx + 1 for idx, value in enumerate(values)}


def build_tool_name_vocab(traces: Sequence[TaskTrace]) -> dict[str, int]:
    names = sorted(
        {
            name
            for trace in traces
            for op in trace.expected_memory_ops
            for name in _tool_names_from_payload(op.payload)
        }
    )
    return {name: idx + 1 for idx, name in enumerate(names)}


def _tool_names_from_payload(payload: MemoryPayload) -> list[str]:
    if isinstance(payload, ToolFactPayload):
        return [payload.tool_name]
    if isinstance(payload, ToolRoutingCommitmentPayload):
        return list(payload.committed_tool_sequence)
    return []


@dataclass(frozen=True)
class EncodedTrace:
    write_tokens: list[int]
    action_tokens: list[int]
    event_tokens: list[list[int]]
    event_op_ids: list[int]
    event_type_ids: list[int]
    event_scope_ids: list[int]
    event_privacy_ids: list[int]
    event_authority_ids: list[int]
    event_slot_ids: list[int]
    event_value_class_ids: list[int]
    event_payload_kind_ids: list[int]
    event_default_action_ids: list[int]
    event_trigger_ids: list[int]
    event_exception_ids: list[int]
    event_expiry_policy_ids: list[int]
    event_authority_level_ids: list[int]
    event_tool_name_ids: list[int]
    event_route_step_ids: list[int]
    event_has_write: list[bool]
    event_has_payload: list[bool]
    event_has_trigger: list[bool]
    event_has_default_action: list[bool]
    event_has_expiry_policy: list[bool]
    event_has_authority_level: list[bool]
    event_has_tool_name: list[bool]
    event_has_route_step: list[bool]
    event_memory_value_tokens: list[list[int]]
    memory_value_tokens: list[int]
    current_scope_id: int
    action_id: int
    op_id: int
    type_id: int
    scope_id: int
    privacy_id: int
    authority_id: int
    slot_id: int
    value_class_id: int
    payload_kind_id: int
    default_action_id: int
    trigger_id: int
    exception_id: int
    expiry_policy_id: int
    authority_level_id: int
    tool_name_id: int
    route_step_id: int


ACTION_TO_ID = {label: idx for idx, label in enumerate(ActionLabel)}
OP_TO_ID = {label: idx for idx, label in enumerate(MemoryOpName)}
TYPE_TO_ID = {label: idx for idx, label in enumerate(MemoryType)}
PRIVACY_TO_ID = {label: idx for idx, label in enumerate(MemoryPrivacy)}
AUTHORITY_TO_ID = {label: idx for idx, label in enumerate(MemoryAuthority)}
PAYLOAD_KIND_TO_ID = {label: idx + 1 for idx, label in enumerate(MemoryPayloadKind)}
POLICY_ACTION_TO_ID = {label: idx + 1 for idx, label in enumerate(PolicyAction)}
POLICY_TRIGGER_TO_ID = {label: idx + 1 for idx, label in enumerate(PolicyTrigger)}
EXPIRY_POLICY_TO_ID = {label: idx + 1 for idx, label in enumerate(ExpiryPolicy)}
AUTHORITY_LEVEL_TO_ID = {label: idx + 1 for idx, label in enumerate(AuthorityLevel)}


def structured_targets_from_payload(
    payload: MemoryPayload, tool_name_vocab: dict[str, int] | None = None
) -> dict[str, int | bool]:
    tool_name_vocab = tool_name_vocab or {}
    targets: dict[str, int | bool] = {
        "payload_kind_id": PAYLOAD_KIND_TO_ID[payload.kind],
        "default_action_id": 0,
        "trigger_id": 0,
        "exception_id": 0,
        "expiry_policy_id": 0,
        "authority_level_id": 0,
        "tool_name_id": 0,
        "route_step_id": 0,
        "has_payload": True,
        "has_trigger": False,
        "has_default_action": False,
        "has_expiry_policy": False,
        "has_authority_level": False,
        "has_tool_name": False,
        "has_route_step": False,
    }
    if isinstance(payload, StandingOrderPayload):
        targets.update(
            default_action_id=POLICY_ACTION_TO_ID[payload.default_action],
            trigger_id=POLICY_TRIGGER_TO_ID[payload.trigger],
            exception_id=POLICY_TRIGGER_TO_ID[payload.exceptions[0]]
            if payload.exceptions
            else 0,
            expiry_policy_id=EXPIRY_POLICY_TO_ID[payload.expiry_policy],
            authority_level_id=AUTHORITY_LEVEL_TO_ID[payload.authority_level],
            has_trigger=True,
            has_default_action=True,
            has_expiry_policy=True,
            has_authority_level=True,
        )
    elif isinstance(payload, ExpiryRulePayload):
        targets.update(
            default_action_id=POLICY_ACTION_TO_ID[payload.expired_action],
            expiry_policy_id=EXPIRY_POLICY_TO_ID[payload.active_until],
            has_default_action=True,
            has_expiry_policy=True,
        )
    elif isinstance(payload, AuthorityRulePayload):
        targets.update(
            default_action_id=POLICY_ACTION_TO_ID[payload.conflict_action],
            authority_level_id=AUTHORITY_LEVEL_TO_ID[payload.winning_authority],
            has_default_action=True,
            has_authority_level=True,
        )
    elif isinstance(payload, ToolFactPayload):
        targets.update(
            tool_name_id=tool_name_vocab.get(payload.tool_name, 0),
            authority_level_id=AUTHORITY_LEVEL_TO_ID[
                AuthorityLevel.TOOL_VERIFIED if payload.verified else AuthorityLevel.TOOL_UNVERIFIED
            ],
            has_tool_name=True,
            has_authority_level=True,
        )
    elif isinstance(payload, TaskStatePayload):
        targets.update(
            default_action_id=POLICY_ACTION_TO_ID[payload.next_required_action],
            route_step_id=payload.step_index,
            has_default_action=True,
            has_route_step=True,
        )
    elif isinstance(payload, DisclosureRulePayload):
        audience = (
            payload.forbidden_audience[0]
            if payload.forbidden_audience
            else payload.allowed_audience[0]
        )
        targets.update(
            default_action_id=POLICY_ACTION_TO_ID[payload.safe_response],
            authority_level_id=AUTHORITY_LEVEL_TO_ID[audience],
            has_default_action=True,
            has_authority_level=True,
        )
    elif isinstance(payload, ToolRoutingCommitmentPayload):
        targets.update(
            default_action_id=POLICY_ACTION_TO_ID[payload.final_action_after_tools],
            tool_name_id=tool_name_vocab.get(payload.next_tool, 0),
            route_step_id=len(payload.completed_tools),
            has_default_action=True,
            has_tool_name=True,
            has_route_step=True,
        )
    return targets


class TraceDataset(Dataset[EncodedTrace]):
    def __init__(
        self,
        traces: list[TaskTrace],
        vocab: TraceVocab | None = None,
        scope_vocab: dict[str, int] | None = None,
        value_vocab: dict[str, int] | None = None,
        tool_name_vocab: dict[str, int] | None = None,
        max_seq_len: int = 256,
    ) -> None:
        self.traces = traces
        self.vocab = vocab or build_vocab(traces)
        self.scope_vocab = scope_vocab or build_scope_vocab(traces)
        self.value_vocab = value_vocab or build_value_vocab(traces)
        self.tool_name_vocab = tool_name_vocab or build_tool_name_vocab(traces)
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
        memory_value_tokens = self.vocab.encode(tokenize_text(memory_text_from_op(first_op)))
        event_targets = self._encode_event_write_targets(trace)
        payload_targets = structured_targets_from_payload(first_op.payload, self.tool_name_vocab)
        return EncodedTrace(
            write_tokens=write_tokens[-self.max_seq_len :],
            action_tokens=action_tokens[-self.max_seq_len :],
            event_tokens=event_targets["tokens"],
            event_op_ids=event_targets["op_ids"],
            event_type_ids=event_targets["type_ids"],
            event_scope_ids=event_targets["scope_ids"],
            event_privacy_ids=event_targets["privacy_ids"],
            event_authority_ids=event_targets["authority_ids"],
            event_slot_ids=event_targets["slot_ids"],
            event_value_class_ids=event_targets["value_class_ids"],
            event_payload_kind_ids=event_targets["payload_kind_ids"],
            event_default_action_ids=event_targets["default_action_ids"],
            event_trigger_ids=event_targets["trigger_ids"],
            event_exception_ids=event_targets["exception_ids"],
            event_expiry_policy_ids=event_targets["expiry_policy_ids"],
            event_authority_level_ids=event_targets["authority_level_ids"],
            event_tool_name_ids=event_targets["tool_name_ids"],
            event_route_step_ids=event_targets["route_step_ids"],
            event_has_write=event_targets["has_write"],
            event_has_payload=event_targets["has_payload"],
            event_has_trigger=event_targets["has_trigger"],
            event_has_default_action=event_targets["has_default_action"],
            event_has_expiry_policy=event_targets["has_expiry_policy"],
            event_has_authority_level=event_targets["has_authority_level"],
            event_has_tool_name=event_targets["has_tool_name"],
            event_has_route_step=event_targets["has_route_step"],
            event_memory_value_tokens=event_targets["memory_value_tokens"],
            memory_value_tokens=memory_value_tokens,
            current_scope_id=self.scope_vocab.get(normalize_scope(trace.current_scope), 0),
            action_id=ACTION_TO_ID[trace.expected_final_action],
            op_id=OP_TO_ID[first_op.op],
            type_id=TYPE_TO_ID[first_op.type],
            scope_id=self.scope_vocab.get(normalize_scope(first_op.scope), 0),
            privacy_id=PRIVACY_TO_ID[first_op.privacy],
            authority_id=AUTHORITY_TO_ID[first_op.authority],
            slot_id=0,
            value_class_id=self.value_vocab.get(first_op.value, 0),
            payload_kind_id=int(payload_targets["payload_kind_id"]),
            default_action_id=int(payload_targets["default_action_id"]),
            trigger_id=int(payload_targets["trigger_id"]),
            exception_id=int(payload_targets["exception_id"]),
            expiry_policy_id=int(payload_targets["expiry_policy_id"]),
            authority_level_id=int(payload_targets["authority_level_id"]),
            tool_name_id=int(payload_targets["tool_name_id"]),
            route_step_id=int(payload_targets["route_step_id"]),
        )

    def _encode_event(self, event, current_scope: str) -> list[int]:
        tokens = [
            f"role_{event.role.lower()}",
            f"scope_{normalize_scope(event.scope or current_scope)}",
            *tokenize_text(event.content),
        ]
        return self.vocab.encode(tokens)

    def _select_supervised_op(self, trace: TaskTrace):
        scoped_ops = [op for op in trace.expected_memory_ops if op.scope == trace.current_scope]
        if scoped_ops:
            return scoped_ops[0]
        return trace.expected_memory_ops[0]

    def _encode_event_write_targets(self, trace: TaskTrace) -> dict[str, list]:
        tokens: list[list[int]] = []
        op_ids: list[int] = []
        type_ids: list[int] = []
        scope_ids: list[int] = []
        privacy_ids: list[int] = []
        authority_ids: list[int] = []
        slot_ids: list[int] = []
        value_class_ids: list[int] = []
        has_write: list[bool] = []
        has_payload: list[bool] = []
        has_trigger: list[bool] = []
        has_default_action: list[bool] = []
        has_expiry_policy: list[bool] = []
        has_authority_level: list[bool] = []
        has_tool_name: list[bool] = []
        has_route_step: list[bool] = []
        memory_value_tokens: list[list[int]] = []
        payload_kind_ids: list[int] = []
        default_action_ids: list[int] = []
        trigger_ids: list[int] = []
        exception_ids: list[int] = []
        expiry_policy_ids: list[int] = []
        authority_level_ids: list[int] = []
        tool_name_ids: list[int] = []
        route_step_ids: list[int] = []
        op_index = 0
        default_op = trace.expected_memory_ops[0]
        for event in trace.events[:-1]:
            tokens.append(self._encode_event(event, trace.current_scope)[-self.max_seq_len :])
            target_op = (
                trace.expected_memory_ops[op_index]
                if op_index < len(trace.expected_memory_ops)
                else None
            )
            if target_op is not None and (event.scope or trace.current_scope) == target_op.scope:
                op_ids.append(OP_TO_ID[target_op.op])
                type_ids.append(TYPE_TO_ID[target_op.type])
                scope_ids.append(self.scope_vocab.get(normalize_scope(target_op.scope), 0))
                privacy_ids.append(PRIVACY_TO_ID[target_op.privacy])
                authority_ids.append(AUTHORITY_TO_ID[target_op.authority])
                slot_ids.append(min(op_index, 31))
                value_class_ids.append(self.value_vocab.get(target_op.value, 0))
                has_write.append(True)
                memory_value_tokens.append(
                    self.vocab.encode(tokenize_text(memory_text_from_op(target_op)))
                )
                structured = structured_targets_from_payload(
                    target_op.payload,
                    self.tool_name_vocab,
                )
                payload_kind_ids.append(int(structured["payload_kind_id"]))
                default_action_ids.append(int(structured["default_action_id"]))
                trigger_ids.append(int(structured["trigger_id"]))
                exception_ids.append(int(structured["exception_id"]))
                expiry_policy_ids.append(int(structured["expiry_policy_id"]))
                authority_level_ids.append(int(structured["authority_level_id"]))
                tool_name_ids.append(int(structured["tool_name_id"]))
                route_step_ids.append(int(structured["route_step_id"]))
                has_payload.append(bool(structured["has_payload"]))
                has_trigger.append(bool(structured["has_trigger"]))
                has_default_action.append(bool(structured["has_default_action"]))
                has_expiry_policy.append(bool(structured["has_expiry_policy"]))
                has_authority_level.append(bool(structured["has_authority_level"]))
                has_tool_name.append(bool(structured["has_tool_name"]))
                has_route_step.append(bool(structured["has_route_step"]))
                op_index += 1
            else:
                op_ids.append(OP_TO_ID[MemoryOpName.NOOP])
                type_ids.append(TYPE_TO_ID[default_op.type])
                scope_ids.append(
                    self.scope_vocab.get(normalize_scope(event.scope or trace.current_scope), 0)
                )
                privacy_ids.append(PRIVACY_TO_ID[default_op.privacy])
                authority_ids.append(AUTHORITY_TO_ID[default_op.authority])
                slot_ids.append(0)
                value_class_ids.append(0)
                has_write.append(False)
                memory_value_tokens.append([0])
                payload_kind_ids.append(0)
                default_action_ids.append(0)
                trigger_ids.append(0)
                exception_ids.append(0)
                expiry_policy_ids.append(0)
                authority_level_ids.append(0)
                tool_name_ids.append(0)
                route_step_ids.append(0)
                has_payload.append(False)
                has_trigger.append(False)
                has_default_action.append(False)
                has_expiry_policy.append(False)
                has_authority_level.append(False)
                has_tool_name.append(False)
                has_route_step.append(False)
        return {
            "tokens": tokens,
            "op_ids": op_ids,
            "type_ids": type_ids,
            "scope_ids": scope_ids,
            "privacy_ids": privacy_ids,
            "authority_ids": authority_ids,
            "slot_ids": slot_ids,
            "value_class_ids": value_class_ids,
            "payload_kind_ids": payload_kind_ids,
            "default_action_ids": default_action_ids,
            "trigger_ids": trigger_ids,
            "exception_ids": exception_ids,
            "expiry_policy_ids": expiry_policy_ids,
            "authority_level_ids": authority_level_ids,
            "tool_name_ids": tool_name_ids,
            "route_step_ids": route_step_ids,
            "has_write": has_write,
            "has_payload": has_payload,
            "has_trigger": has_trigger,
            "has_default_action": has_default_action,
            "has_expiry_policy": has_expiry_policy,
            "has_authority_level": has_authority_level,
            "has_tool_name": has_tool_name,
            "has_route_step": has_route_step,
            "memory_value_tokens": memory_value_tokens,
        }


def collate_traces(batch: Sequence[EncodedTrace]) -> dict[str, torch.Tensor]:
    max_action_len = max(len(item.action_tokens) for item in batch)
    max_write_len = max(len(item.write_tokens) for item in batch)
    max_value_len = max(len(item.memory_value_tokens) for item in batch)
    max_events = max(len(item.event_tokens) for item in batch)
    max_event_len = max(len(tokens) for item in batch for tokens in item.event_tokens)
    max_event_value_len = max(
        len(tokens) for item in batch for tokens in item.event_memory_value_tokens
    )
    tokens = torch.zeros((len(batch), max_action_len), dtype=torch.long)
    write_tokens = torch.zeros((len(batch), max_write_len), dtype=torch.long)
    memory_value_tokens = torch.zeros((len(batch), max_value_len), dtype=torch.long)
    memory_value_mask = torch.zeros((len(batch), max_value_len), dtype=torch.bool)
    event_tokens = torch.zeros((len(batch), max_events, max_event_len), dtype=torch.long)
    event_lengths = torch.zeros((len(batch), max_events), dtype=torch.long)
    event_mask = torch.zeros((len(batch), max_events), dtype=torch.bool)
    event_value_tokens = torch.zeros(
        (len(batch), max_events, max_event_value_len), dtype=torch.long
    )
    event_value_mask = torch.zeros((len(batch), max_events, max_event_value_len), dtype=torch.bool)
    for i, item in enumerate(batch):
        tokens[i, : len(item.action_tokens)] = torch.tensor(item.action_tokens, dtype=torch.long)
        write_tokens[i, : len(item.write_tokens)] = torch.tensor(
            item.write_tokens, dtype=torch.long
        )
        memory_value_tokens[i, : len(item.memory_value_tokens)] = torch.tensor(
            item.memory_value_tokens, dtype=torch.long
        )
        memory_value_mask[i, : len(item.memory_value_tokens)] = True
        for j, event in enumerate(item.event_tokens):
            event_tokens[i, j, : len(event)] = torch.tensor(event, dtype=torch.long)
            event_lengths[i, j] = len(event)
            event_mask[i, j] = True
            value_tokens = item.event_memory_value_tokens[j]
            event_value_tokens[i, j, : len(value_tokens)] = torch.tensor(
                value_tokens, dtype=torch.long
            )
            event_value_mask[i, j, : len(value_tokens)] = True
    return {
        "tokens": tokens,
        "lengths": torch.tensor([len(item.action_tokens) for item in batch], dtype=torch.long),
        "write_tokens": write_tokens,
        "write_lengths": torch.tensor([len(item.write_tokens) for item in batch], dtype=torch.long),
        "memory_value_tokens": memory_value_tokens,
        "memory_value_mask": memory_value_mask,
        "event_tokens": event_tokens,
        "event_lengths": event_lengths,
        "event_mask": event_mask,
        "event_value_tokens": event_value_tokens,
        "event_value_mask": event_value_mask,
        "event_op_ids": _pad_event_labels(batch, "event_op_ids", max_events),
        "event_type_ids": _pad_event_labels(batch, "event_type_ids", max_events),
        "event_scope_ids": _pad_event_labels(batch, "event_scope_ids", max_events),
        "event_privacy_ids": _pad_event_labels(batch, "event_privacy_ids", max_events),
        "event_authority_ids": _pad_event_labels(batch, "event_authority_ids", max_events),
        "event_slot_ids": _pad_event_labels(batch, "event_slot_ids", max_events),
        "event_value_class_ids": _pad_event_labels(batch, "event_value_class_ids", max_events),
        "event_payload_kind_ids": _pad_event_labels(batch, "event_payload_kind_ids", max_events),
        "event_default_action_ids": _pad_event_labels(
            batch, "event_default_action_ids", max_events
        ),
        "event_trigger_ids": _pad_event_labels(batch, "event_trigger_ids", max_events),
        "event_exception_ids": _pad_event_labels(batch, "event_exception_ids", max_events),
        "event_expiry_policy_ids": _pad_event_labels(
            batch, "event_expiry_policy_ids", max_events
        ),
        "event_authority_level_ids": _pad_event_labels(
            batch, "event_authority_level_ids", max_events
        ),
        "event_tool_name_ids": _pad_event_labels(batch, "event_tool_name_ids", max_events),
        "event_route_step_ids": _pad_event_labels(batch, "event_route_step_ids", max_events),
        "event_has_write": _pad_event_bools(batch, "event_has_write", max_events),
        "event_has_payload": _pad_event_bools(batch, "event_has_payload", max_events),
        "event_has_trigger": _pad_event_bools(batch, "event_has_trigger", max_events),
        "event_has_default_action": _pad_event_bools(
            batch, "event_has_default_action", max_events
        ),
        "event_has_expiry_policy": _pad_event_bools(
            batch, "event_has_expiry_policy", max_events
        ),
        "event_has_authority_level": _pad_event_bools(
            batch, "event_has_authority_level", max_events
        ),
        "event_has_tool_name": _pad_event_bools(batch, "event_has_tool_name", max_events),
        "event_has_route_step": _pad_event_bools(batch, "event_has_route_step", max_events),
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
        "value_class_ids": torch.tensor([item.value_class_id for item in batch], dtype=torch.long),
        "payload_kind_ids": torch.tensor(
            [item.payload_kind_id for item in batch], dtype=torch.long
        ),
        "default_action_ids": torch.tensor(
            [item.default_action_id for item in batch], dtype=torch.long
        ),
        "trigger_ids": torch.tensor([item.trigger_id for item in batch], dtype=torch.long),
        "exception_ids": torch.tensor([item.exception_id for item in batch], dtype=torch.long),
        "expiry_policy_ids": torch.tensor(
            [item.expiry_policy_id for item in batch], dtype=torch.long
        ),
        "authority_level_ids": torch.tensor(
            [item.authority_level_id for item in batch], dtype=torch.long
        ),
        "tool_name_ids": torch.tensor([item.tool_name_id for item in batch], dtype=torch.long),
        "route_step_ids": torch.tensor([item.route_step_id for item in batch], dtype=torch.long),
    }


def _pad_event_labels(batch: Sequence[EncodedTrace], attr: str, max_events: int) -> torch.Tensor:
    labels = torch.zeros((len(batch), max_events), dtype=torch.long)
    for i, item in enumerate(batch):
        values = getattr(item, attr)
        labels[i, : len(values)] = torch.tensor(values, dtype=torch.long)
    return labels


def _pad_event_bools(batch: Sequence[EncodedTrace], attr: str, max_events: int) -> torch.Tensor:
    labels = torch.zeros((len(batch), max_events), dtype=torch.bool)
    for i, item in enumerate(batch):
        values = getattr(item, attr)
        labels[i, : len(values)] = torch.tensor(values, dtype=torch.bool)
    return labels
