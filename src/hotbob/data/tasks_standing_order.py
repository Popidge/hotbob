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
    policy, value, action = rng.choice(
        [
            (
                "Hold fire unless civilians are at risk.",
                "hold_fire_unless_civilians_at_risk",
                ActionLabel.HOLD_FIRE,
            ),
            (
                "Raise shields when weapons are powered.",
                "raise_shields_on_powered_weapons",
                ActionLabel.RAISE_SHIELDS,
            ),
            (
                "Fire weapons when hostile lock is confirmed.",
                "fire_on_hostile_lock",
                ActionLabel.FIRE_WEAPONS,
            ),
        ]
    )
    event = "A tactical trigger is present. Apply standing order."
    return TaskTrace(
        events=[
            TraceEvent(
                role="CAPTAIN",
                content=policy,
                scope=scope,
            ),
            TraceEvent(role="SIM_EVENT", content=event, scope=scope),
        ],
        expected_memory_ops=[
            MemoryOp(
                op=MemoryOpName.WRITE,
                type=MemoryType.STANDING_ORDER,
                key="weapons_policy",
                value=value,
                scope=scope,
                privacy=MemoryPrivacy.VISIBLE,
                authority=MemoryAuthority.USER,
            )
        ],
        expected_final_action=action,
        current_scope=scope,
        task_family="standing_order",
        metadata={
            "memory_required": True,
            "final_event_hides_memory_value": True,
        },
    )
