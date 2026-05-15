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
    current = rng.choice(["repo_a", "repo_b"])
    target = (
        ActionLabel.INSPECT_FUNCTION_DAVE
        if current == "repo_a"
        else ActionLabel.INSPECT_FUNCTION_CALCULATE_FINAL_SCORE
    )
    return TaskTrace(
        events=[
            TraceEvent(
                role="USER", content="In REPO_A, dave = final scoring function.", scope="repo_a"
            ),
            TraceEvent(
                role="USER", content="In REPO_B, dave = database migration script.", scope="repo_b"
            ),
            TraceEvent(role="CURRENT_SCOPE", content=current.upper(), scope=current),
            TraceEvent(role="USER", content="inspect final scoring function", scope=current),
        ],
        expected_memory_ops=[
            MemoryOp(
                op=MemoryOpName.WRITE,
                type=MemoryType.SYMBOL_BINDING,
                key="dave",
                value="final_scoring_function",
                scope="repo_a",
                privacy=MemoryPrivacy.VISIBLE,
                authority=MemoryAuthority.USER,
            ),
            MemoryOp(
                op=MemoryOpName.WRITE,
                type=MemoryType.SYMBOL_BINDING,
                key="dave",
                value="database_migration_script",
                scope="repo_b",
                privacy=MemoryPrivacy.VISIBLE,
                authority=MemoryAuthority.USER,
            ),
        ],
        expected_final_action=target,
        current_scope=current,
        task_family="scope_isolation",
        metadata={},
    )
