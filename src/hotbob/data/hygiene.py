from __future__ import annotations

from hotbob.training.dataset import tokenize_text
from hotbob.types import TaskTrace

GENERIC_MEMORY_TOKENS = {
    "function",
    "script",
    "priority",
    "policy",
    "target",
    "order",
    "secret",
    "colour",
    "color",
    "speed",
    "stealth",
    "weapons",
    "fire",
    "hold",
    "raise",
    "shields",
    "route",
    "inspection",
}


def final_event_memory_leaks(trace: TaskTrace) -> list[str]:
    """Return memory key/value tokens that appear in the final action event."""

    final_tokens = set(tokenize_text(trace.events[-1].content))
    leaks: list[str] = []
    for op in trace.expected_memory_ops:
        for token in tokenize_text(op.key) + tokenize_text(op.value):
            if token in GENERIC_MEMORY_TOKENS:
                continue
            if token in final_tokens:
                leaks.append(token)
    return sorted(set(leaks))


def is_memory_required_trace(trace: TaskTrace) -> bool:
    return bool(trace.metadata.get("memory_required"))
