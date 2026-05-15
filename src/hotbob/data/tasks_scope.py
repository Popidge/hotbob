from __future__ import annotations

import random

from hotbob.types import (
    ActionLabel,
    MemoryAuthority,
    MemoryOp,
    MemoryOpName,
    MemoryPrivacy,
    MemoryType,
    TaskTrace,
    TraceEvent,
)


def make_scope_isolation_trace(rng: random.Random, idx: int) -> TaskTrace:
    scope_a = f"repo_{idx}_a"
    scope_b = f"repo_{idx}_b"
    current = rng.choice([scope_a, scope_b])
    a_value, a_action = rng.choice(
        [
            ("dave", ActionLabel.INSPECT_FUNCTION_DAVE),
            ("calculate_final_score", ActionLabel.INSPECT_FUNCTION_CALCULATE_FINAL_SCORE),
        ]
    )
    b_value, b_action = (
        ("calculate_final_score", ActionLabel.INSPECT_FUNCTION_CALCULATE_FINAL_SCORE)
        if a_value == "dave"
        else ("dave", ActionLabel.INSPECT_FUNCTION_DAVE)
    )
    target = a_action if current == scope_a else b_action
    return TaskTrace(
        events=[
            TraceEvent(
                role="USER",
                content=f"In this scope, the inspection target is {a_value}.",
                scope=scope_a,
            ),
            TraceEvent(
                role="USER",
                content=f"In this scope, the inspection target is {b_value}.",
                scope=scope_b,
            ),
            TraceEvent(role="CURRENT_SCOPE", content="selected scope", scope=current),
            TraceEvent(role="USER", content="Inspect the scoped target.", scope=current),
        ],
        expected_memory_ops=[
            MemoryOp(
                op=MemoryOpName.WRITE,
                type=MemoryType.SYMBOL_BINDING,
                key="inspection_target",
                value=a_value,
                scope=scope_a,
                privacy=MemoryPrivacy.VISIBLE,
                authority=MemoryAuthority.USER,
            ),
            MemoryOp(
                op=MemoryOpName.WRITE,
                type=MemoryType.SYMBOL_BINDING,
                key="inspection_target",
                value=b_value,
                scope=scope_b,
                privacy=MemoryPrivacy.VISIBLE,
                authority=MemoryAuthority.USER,
            ),
        ],
        expected_final_action=target,
        current_scope=current,
        task_family="scope_isolation",
        metadata={"memory_required": True, "final_event_hides_memory_value": True},
    )
