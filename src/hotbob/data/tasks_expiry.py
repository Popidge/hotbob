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
    active = rng.random() < 0.5
    priority, action = rng.choice(
        [
            ("stealth_over_speed", ActionLabel.APPLY_STEALTH_PRIORITY),
            ("speed_over_stealth", ActionLabel.APPLY_SPEED_PRIORITY),
        ]
    )
    final_scope = old_scope if active else new_scope
    events = [
        TraceEvent(
            role="USER",
            content=f"For this mission only, prioritise {priority.replace('_over_', ' over ')}.",
            scope=old_scope,
        )
    ]
    expected_ops = [
        MemoryOp(
            op=MemoryOpName.WRITE,
            type=MemoryType.PREFERENCE,
            key="route_priority",
            value=priority,
            scope=old_scope,
            privacy=MemoryPrivacy.VISIBLE,
            authority=MemoryAuthority.USER,
            ttl=2,
        )
    ]
    if active:
        events.append(TraceEvent(role="USER", content="Plot route", scope=old_scope))
        final_action = action
    else:
        events.extend(
            [
                TraceEvent(role="MISSION_END", content=old_scope, scope=old_scope),
                TraceEvent(role="NEW_MISSION", content=new_scope, scope=new_scope),
                TraceEvent(role="USER", content="Plot route", scope=new_scope),
            ]
        )
        expected_ops.append(
            MemoryOp(
                op=MemoryOpName.DELETE,
                type=MemoryType.PREFERENCE,
                key="route_priority",
                value=priority,
                scope=old_scope,
                privacy=MemoryPrivacy.VISIBLE,
                authority=MemoryAuthority.SIM,
            )
        )
        final_action = ActionLabel.IGNORE_EXPIRED_ORDER
    return TaskTrace(
        events=events,
        expected_memory_ops=expected_ops,
        expected_final_action=final_action,
        current_scope=final_scope,
        task_family="expiry",
        metadata={
            "expired_scope": old_scope,
            "active_preference": active,
            "memory_required": True,
            "final_event_hides_memory_value": True,
        },
    )
