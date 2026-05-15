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
    return TaskTrace(
        events=[
            TraceEvent(
                role="USER",
                content="In this repo, dave is the function that calculates the final result.",
                scope=scope,
            ),
            TraceEvent(role="USER", content=distractor, scope=scope),
            TraceEvent(role="USER", content="Fix the final result bug.", scope=scope),
        ],
        expected_memory_ops=[
            MemoryOp(
                op=MemoryOpName.WRITE,
                type=MemoryType.SYMBOL_BINDING,
                key="dave",
                value="final_result_function",
                scope=scope,
                privacy=MemoryPrivacy.VISIBLE,
                authority=MemoryAuthority.USER,
            )
        ],
        expected_final_action=ActionLabel.INSPECT_FUNCTION_DAVE,
        current_scope=scope,
        task_family="symbol_binding",
        metadata={},
    )
