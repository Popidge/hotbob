from __future__ import annotations

import random

from hotbob.types import (
    ActionLabel,
    AuthorityLevel,
    ExpiryPolicy,
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


def _op(scope: str, payload: StandingOrderPayload, key: str = "weapons_policy") -> MemoryOp:
    return MemoryOp(
        op=MemoryOpName.WRITE,
        type=MemoryType.STANDING_ORDER,
        key=key,
        value=f"{payload.default_action}_{payload.trigger}",
        payload=payload,
        scope=scope,
        privacy=MemoryPrivacy.VISIBLE,
        authority=MemoryAuthority.USER,
    )


def make_rich_standing_order_trace(rng: random.Random, idx: int) -> TaskTrace:
    scope = f"mission_{idx % 17}"
    scenario = (idx // 8) % 4
    if scenario == 0:
        payload = StandingOrderPayload(
            kind=MemoryPayloadKind.STANDING_ORDER,
            default_action=PolicyAction.HOLD_FIRE,
            trigger=PolicyTrigger.HOSTILE_POSTURE,
            allowed_responses=[PolicyAction.HOLD_FIRE, PolicyAction.RAISE_SHIELDS],
            forbidden_responses=[PolicyAction.FIRE_WEAPONS],
            exceptions=[PolicyTrigger.CIVILIANS_AT_RISK],
            authority_level=AuthorityLevel.CAPTAIN,
            expiry_policy=ExpiryPolicy.MISSION_END,
        )
        final = "Hostile posture detected. No civilian risk is present."
        action = ActionLabel.HOLD_FIRE
    elif scenario == 1:
        payload = StandingOrderPayload(
            kind=MemoryPayloadKind.STANDING_ORDER,
            default_action=PolicyAction.HOLD_FIRE,
            trigger=PolicyTrigger.HOSTILE_POSTURE,
            allowed_responses=[PolicyAction.RAISE_SHIELDS, PolicyAction.EVADE],
            forbidden_responses=[PolicyAction.FIRE_WEAPONS],
            exceptions=[PolicyTrigger.CIVILIANS_AT_RISK],
            authority_level=AuthorityLevel.CAPTAIN,
            expiry_policy=ExpiryPolicy.MISSION_END,
        )
        final = "Hostile posture detected. Civilians are at risk."
        action = ActionLabel.RAISE_SHIELDS
    elif scenario == 2:
        payload = StandingOrderPayload(
            kind=MemoryPayloadKind.STANDING_ORDER,
            default_action=PolicyAction.EVADE,
            trigger=PolicyTrigger.HOSTILE_LOCK,
            allowed_responses=[PolicyAction.EVADE, PolicyAction.HAIL],
            forbidden_responses=[PolicyAction.FIRE_WEAPONS],
            authority_level=AuthorityLevel.CAPTAIN,
            expiry_policy=ExpiryPolicy.MISSION_END,
        )
        final = "Unverified hostile lock detected. Weapons are tempting but restricted."
        action = ActionLabel.EVADE
    else:
        old_scope = f"{scope}_old"
        payload = StandingOrderPayload(
            kind=MemoryPayloadKind.STANDING_ORDER,
            default_action=PolicyAction.HAIL,
            trigger=PolicyTrigger.WEAPONS_POWERED,
            allowed_responses=[PolicyAction.HAIL, PolicyAction.RAISE_SHIELDS],
            forbidden_responses=[PolicyAction.FIRE_WEAPONS],
            authority_level=AuthorityLevel.CAPTAIN,
            expiry_policy=ExpiryPolicy.MISSION_END,
        )
        other = StandingOrderPayload(
            kind=MemoryPayloadKind.STANDING_ORDER,
            default_action=PolicyAction.FIRE_WEAPONS,
            trigger=PolicyTrigger.WEAPONS_POWERED,
            allowed_responses=[PolicyAction.FIRE_WEAPONS],
            forbidden_responses=[PolicyAction.HAIL],
            authority_level=AuthorityLevel.USER,
            expiry_policy=ExpiryPolicy.MISSION_END,
        )
        return TaskTrace(
            events=[
                TraceEvent(
                    role="CAPTAIN",
                    content="Current mission standing order recorded.",
                    scope=scope,
                ),
                TraceEvent(
                    role="USER",
                    content="Old mission standing order recorded.",
                    scope=old_scope,
                ),
                TraceEvent(
                    role="SIM_EVENT",
                    content="Weapons powered in the current mission.",
                    scope=scope,
                ),
            ],
            expected_memory_ops=[_op(scope, payload), _op(old_scope, other)],
            expected_final_action=ActionLabel.HAIL,
            current_scope=scope,
            task_family="standing_order",
            metadata={"scenario": "scope_selects_current_order"},
        )
    return TaskTrace(
        events=[
            TraceEvent(role="CAPTAIN", content="Standing tactical order recorded.", scope=scope),
            TraceEvent(role="SIM_EVENT", content=final, scope=scope),
        ],
        expected_memory_ops=[_op(scope, payload)],
        expected_final_action=action,
        current_scope=scope,
        task_family="standing_order",
        metadata={"scenario": f"standing_order_{scenario}"},
    )
