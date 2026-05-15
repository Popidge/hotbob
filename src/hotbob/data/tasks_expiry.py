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


def make_expiry_trace(rng: random.Random, idx: int) -> TaskTrace:
    old_scope = f"mission_{idx}_old"
    new_scope = f"mission_{idx}_new"
    return TaskTrace(
        events=[
            TraceEvent(
                role="USER",
                content="For this mission only, prioritise stealth over speed.",
                scope=old_scope,
            ),
            TraceEvent(role="MISSION_END", content=old_scope, scope=old_scope),
            TraceEvent(role="NEW_MISSION", content=new_scope, scope=new_scope),
            TraceEvent(role="USER", content="Plot route", scope=new_scope),
        ],
        expected_memory_ops=[
            MemoryOp(
                op=MemoryOpName.WRITE,
                type=MemoryType.PREFERENCE,
                key="route_priority",
                value="stealth_over_speed",
                scope=old_scope,
                privacy=MemoryPrivacy.VISIBLE,
                authority=MemoryAuthority.USER,
                ttl=2,
            ),
            MemoryOp(
                op=MemoryOpName.DELETE,
                type=MemoryType.PREFERENCE,
                key="route_priority",
                value="stealth_over_speed",
                scope=old_scope,
                privacy=MemoryPrivacy.VISIBLE,
                authority=MemoryAuthority.SIM,
            ),
        ],
        expected_final_action=ActionLabel.IGNORE_EXPIRED_ORDER,
        current_scope=new_scope,
        task_family="expiry",
        metadata={"expired_scope": old_scope},
    )
