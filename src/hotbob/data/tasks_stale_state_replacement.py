from __future__ import annotations

import random

from hotbob.types import (
    ActionLabel,
    MemoryAuthority,
    MemoryOp,
    MemoryOpName,
    MemoryPayloadKind,
    MemoryPrivacy,
    MemoryType,
    PolicyAction,
    TaskStatePayload,
    TaskTrace,
    TraceEvent,
)


def make_stale_state_replacement_trace(rng: random.Random, idx: int) -> TaskTrace:
    scope = f"mission_{idx % 37}"
    scenario = (idx // 8) % 4
    op_scope = f"{scope}_old" if scenario == 2 else scope
    payload = TaskStatePayload(
        kind=MemoryPayloadKind.TASK_STATE,
        task_id=f"route_{idx % 13}",
        step_index=scenario,
        next_required_action=PolicyAction.CONTINUE_COMMITTED_ROUTE,
        interrupted=False,
    )
    final_action = [
        ActionLabel.REPLACE_STALE_STATE,
        ActionLabel.REPLACE_STALE_STATE,
        ActionLabel.KEEP_CURRENT_STATE,
        ActionLabel.REPLACE_STALE_STATE,
    ][scenario]
    final = [
        "Old target is stale; use the new target.",
        "Verified route replaces the stale route.",
        "Stale state belongs to an old scope.",
        "Replacement should keep authority and privacy metadata.",
    ][scenario]
    return TaskTrace(
        events=[
            TraceEvent(role="SIM_EVENT", content="State replacement recorded.", scope=op_scope),
            TraceEvent(role="USER", content=final, scope=scope),
        ],
        expected_memory_ops=[
            MemoryOp(
                op=MemoryOpName.UPDATE,
                type=MemoryType.TASK_STATE,
                key=payload.task_id,
                value=f"replace_{payload.task_id}",
                payload=payload,
                scope=op_scope,
                privacy=MemoryPrivacy.INTERNAL_ONLY,
                authority=MemoryAuthority.TOOL if scenario == 1 else MemoryAuthority.MODEL,
            )
        ],
        expected_final_action=final_action,
        current_scope=scope,
        task_family="stale_state_replacement",
        metadata={"scenario": f"stale_replace_{scenario}"},
    )
