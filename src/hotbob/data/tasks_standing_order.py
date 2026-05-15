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


def make_standing_order_trace(rng: random.Random, idx: int) -> TaskTrace:
    scope = f"mission_{idx % 13}"
    civilians = rng.random() < 0.45
    event = (
        "Enemy targets civilian transport."
        if civilians
        else "Enemy powers weapons. Civilians absent."
    )
    action = ActionLabel.FIRE_WEAPONS if civilians else ActionLabel.HOLD_FIRE
    return TaskTrace(
        events=[
            TraceEvent(
                role="CAPTAIN",
                content="Do not fire first unless civilians are at risk.",
                scope=scope,
            ),
            TraceEvent(role="SIM_EVENT", content=event, scope=scope),
        ],
        expected_memory_ops=[
            MemoryOp(
                op=MemoryOpName.WRITE,
                type=MemoryType.STANDING_ORDER,
                key="weapons_policy",
                value="no_fire_first_unless_civilians_at_risk",
                scope=scope,
                privacy=MemoryPrivacy.VISIBLE,
                authority=MemoryAuthority.USER,
            )
        ],
        expected_final_action=action,
        current_scope=scope,
        task_family="standing_order",
        metadata={"civilians_at_risk": civilians},
    )
