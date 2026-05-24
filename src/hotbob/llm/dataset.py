from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from hotbob.data.traces import generate_traces, generate_weighted_traces
from hotbob.llm.prompts import final_prompt_from_trace, target_text_from_trace
from hotbob.training.dataset import normalize_scope
from hotbob.types import MemoryOp, TaskTrace, TraceEvent


class LLMTraceEvent(BaseModel):
    role: str
    content: str
    scope: str | None = None


class LLMTrace(BaseModel):
    events: list[LLMTraceEvent]
    final_prompt: str
    target_text: str
    target_action: str
    current_scope: str
    expected_memory_ops: list[MemoryOp] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class TracePrivacyReport:
    final_prompt_contains_hidden_value: bool
    hidden_values: tuple[str, ...]


def task_trace_to_llm_trace(trace: TaskTrace) -> LLMTrace:
    return LLMTrace(
        events=[
            LLMTraceEvent(role=event.role, content=event.content, scope=event.scope)
            for event in trace.events[:-1]
        ],
        final_prompt=final_prompt_from_trace(trace),
        target_text=target_text_from_trace(trace),
        target_action=str(trace.expected_final_action),
        current_scope=trace.current_scope,
        expected_memory_ops=trace.expected_memory_ops,
        metadata=dict(
            trace.metadata,
            task_family=trace.task_family,
            memory_required=True,
            structured_payload_required=True,
            final_event_hides_memory_value=trace.metadata.get(
                "final_event_hides_memory_value", True
            ),
        ),
    )


def privacy_report(trace: LLMTrace) -> TracePrivacyReport:
    hidden_values = tuple(
        op.value for op in trace.expected_memory_ops if "HIDDEN_FROM_USER" in str(op.privacy)
    )
    contains_hidden_value = any(value in trace.final_prompt for value in hidden_values)
    return TracePrivacyReport(
        final_prompt_contains_hidden_value=contains_hidden_value,
        hidden_values=hidden_values,
    )


def build_llm_tool_name_vocab(traces: list[LLMTrace]) -> dict[str, int]:
    names: set[str] = set()
    for trace in traces:
        for op in trace.expected_memory_ops:
            payload = op.payload
            if hasattr(payload, "tool_name"):
                names.add(str(payload.tool_name))
            if hasattr(payload, "committed_tool_sequence"):
                names.update(str(name) for name in payload.committed_tool_sequence)
    return {name: idx + 1 for idx, name in enumerate(sorted(names))}


def write_llm_jsonl(traces: list[LLMTrace], out: str | Path) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace.model_dump(mode="json")) + "\n")


def read_llm_jsonl(path: str | Path) -> list[LLMTrace]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [LLMTrace.model_validate_json(line) for line in f if line.strip()]


def generate_llm_traces(n: int, seed: int) -> list[LLMTrace]:
    return [task_trace_to_llm_trace(trace) for trace in generate_traces(n, seed)]


def generate_weighted_llm_traces(
    n: int,
    seed: int,
    family_weights: dict[str, float] | None = None,
) -> list[LLMTrace]:
    return [
        task_trace_to_llm_trace(trace)
        for trace in generate_weighted_traces(n, seed, family_weights)
    ]


def build_llm_scope_vocab(traces: list[LLMTrace]) -> dict[str, int]:
    scopes = sorted(
        {
            normalize_scope(scope)
            for trace in traces
            for scope in (
                [trace.current_scope]
                + [event.scope for event in trace.events if event.scope is not None]
                + [op.scope for op in trace.expected_memory_ops]
            )
        }
    )
    return {scope: idx + 1 for idx, scope in enumerate(scopes)}


def trace_event_from_llm(event: LLMTraceEvent) -> TraceEvent:
    return TraceEvent(role=event.role, content=event.content, scope=event.scope)
