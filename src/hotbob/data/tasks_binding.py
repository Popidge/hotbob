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


def make_symbol_binding_trace(rng: random.Random, idx: int) -> TaskTrace:
    scope = f"repo_{idx % 11}"
    distractor = rng.choice(["alice handles parsing", "maria owns tests", "sam wrote docs"])
    target_is_dave = rng.random() < 0.5
    symbol = "dave" if target_is_dave else "calculate_final_score"
    action = (
        ActionLabel.INSPECT_FUNCTION_DAVE
        if target_is_dave
        else ActionLabel.INSPECT_FUNCTION_CALCULATE_FINAL_SCORE
    )
    return TaskTrace(
        events=[
            TraceEvent(
                role="USER",
                content=f"In this repo, the remembered target function is {symbol}.",
                scope=scope,
            ),
            TraceEvent(role="USER", content=distractor, scope=scope),
            TraceEvent(role="USER", content="Inspect the remembered target.", scope=scope),
        ],
        expected_memory_ops=[
            MemoryOp(
                op=MemoryOpName.WRITE,
                type=MemoryType.SYMBOL_BINDING,
                key="remembered_target",
                value=symbol,
                scope=scope,
                privacy=MemoryPrivacy.VISIBLE,
                authority=MemoryAuthority.USER,
            )
        ],
        expected_final_action=action,
        current_scope=scope,
        task_family="symbol_binding",
        metadata={"memory_required": True, "final_event_hides_memory_value": True},
    )
