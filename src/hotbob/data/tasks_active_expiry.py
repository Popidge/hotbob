from __future__ import annotations

import random

from hotbob.types import (
    ActionLabel,
    AuthorityLevel,
    ExpiryPolicy,
    ExpiryRulePayload,
    MemoryAuthority,
    MemoryOp,
    MemoryOpName,
    MemoryPayloadKind,
    MemoryPrivacy,
    MemoryType,
    PolicyAction,
    PolicyTrigger,
    StandingOrderPayload,
    TaskTrace,
    TraceEvent,
)


def make_active_expiry_trace(rng: random.Random, idx: int) -> TaskTrace:
    scope = f"mission_{idx % 19}"
    scenario = (idx // 8) % 4
    order = StandingOrderPayload(
        kind=MemoryPayloadKind.STANDING_ORDER,
        default_action=PolicyAction.RAISE_SHIELDS,
        trigger=PolicyTrigger.WEAPONS_POWERED,
        allowed_responses=[PolicyAction.RAISE_SHIELDS, PolicyAction.EVADE],
        forbidden_responses=[PolicyAction.FIRE_WEAPONS],
        authority_level=AuthorityLevel.CAPTAIN,
        expiry_policy=ExpiryPolicy.MISSION_END if scenario == 0 else ExpiryPolicy.EVENT_COUNT,
    )
    ops = [
        MemoryOp(
            op=MemoryOpName.WRITE,
            type=MemoryType.STANDING_ORDER,
            key="temporary_priority",
            value="raise_shields_when_weapons_powered",
            payload=order,
            scope=scope,
            privacy=MemoryPrivacy.VISIBLE if scenario != 3 else MemoryPrivacy.HIDDEN_FROM_USER,
            authority=MemoryAuthority.USER,
            ttl=1 if scenario in {1, 2, 3} else None,
        )
    ]
    if scenario == 0:
        final, action = "Temporary priority is still active.", ActionLabel.RAISE_SHIELDS
    elif scenario == 1:
        expiry = ExpiryRulePayload(
            kind=MemoryPayloadKind.EXPIRY_RULE,
            target_key="temporary_priority",
            active_until=ExpiryPolicy.EVENT_COUNT,
            expired_action=PolicyAction.HOLD_FIRE,
        )
        ops.append(
            MemoryOp(
                op=MemoryOpName.DELETE,
                type=MemoryType.EXPIRY_RULE,
                key="temporary_priority_expired",
                value="delete_expired_priority",
                payload=expiry,
                scope=scope,
                privacy=MemoryPrivacy.VISIBLE,
                authority=MemoryAuthority.SIM,
            )
        )
        final, action = "Temporary priority has expired.", ActionLabel.HOLD_FIRE
    elif scenario == 2:
        expiry = ExpiryRulePayload(
            kind=MemoryPayloadKind.EXPIRY_RULE,
            target_key="temporary_priority",
            active_until=ExpiryPolicy.EVENT_COUNT,
            expired_action=PolicyAction.HAIL,
            replacement_key="replacement_priority",
        )
        ops.append(
            MemoryOp(
                op=MemoryOpName.UPDATE,
                type=MemoryType.EXPIRY_RULE,
                key="replacement_priority",
                value="replace_expired_priority",
                payload=expiry,
                scope=scope,
                privacy=MemoryPrivacy.VISIBLE,
                authority=MemoryAuthority.SIM,
            )
        )
        final, action = "Expired priority has a replacement instruction.", ActionLabel.HAIL
    else:
        expiry = ExpiryRulePayload(
            kind=MemoryPayloadKind.EXPIRY_RULE,
            target_key="temporary_priority",
            active_until=ExpiryPolicy.EVENT_COUNT,
            expired_action=PolicyAction.HOLD_FIRE,
        )
        ops.append(
            MemoryOp(
                op=MemoryOpName.DELETE,
                type=MemoryType.EXPIRY_RULE,
                key="hidden_priority_expired",
                value="hidden_expired_priority",
                payload=expiry,
                scope=scope,
                privacy=MemoryPrivacy.HIDDEN_FROM_USER,
                authority=MemoryAuthority.SIM,
            )
        )
        final, action = "A private temporary priority has expired.", ActionLabel.HOLD_FIRE
    return TaskTrace(
        events=[
            TraceEvent(role="CAPTAIN", content="Temporary operational rule recorded.", scope=scope),
            TraceEvent(role="SIM_EVENT", content=final, scope=scope),
        ],
        expected_memory_ops=ops,
        expected_final_action=action,
        current_scope=scope,
        task_family="active_expiry",
        metadata={"scenario": f"active_expiry_{scenario}"},
    )
