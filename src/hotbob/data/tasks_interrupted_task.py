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


def make_interrupted_task_trace(rng: random.Random, idx: int) -> TaskTrace:
    scope = f"mission_{idx % 31}"
    scenario = (idx // 8) % 4
    next_action = [
        PolicyAction.CONTINUE_COMMITTED_ROUTE,
        PolicyAction.ASK_CLARIFICATION,
        PolicyAction.INSPECT_TOOL,
        PolicyAction.CONTINUE_COMMITTED_ROUTE,
    ][scenario]
    payload = TaskStatePayload(
        kind=MemoryPayloadKind.TASK_STATE,
        task_id=f"task_{idx % 11}",
        step_index=scenario + 1,
        next_required_action=next_action,
        interrupted=scenario in {0, 1},
        resume_policy=next_action,
    )
    final_action = [
        ActionLabel.RESUME_INTERRUPTED_TASK,
        ActionLabel.ABANDON_INTERRUPTED_TASK,
        ActionLabel.CALL_SENSOR_TOOL,
        ActionLabel.CONTINUE_TOOL_ROUTE,
    ][scenario]
    final = [
        "Interrupt cleared; continue the saved task.",
        "Higher authority interrupt supersedes saved task.",
        "Continue with the next required tool call.",
        "Do not repeat the completed step.",
    ][scenario]
    return TaskTrace(
        events=[
            TraceEvent(role="ASSISTANT", content="Task progress recorded.", scope=scope),
            TraceEvent(role="USER", content=final, scope=scope),
        ],
        expected_memory_ops=[
            MemoryOp(
                op=MemoryOpName.WRITE,
                type=MemoryType.TASK_STATE,
                key=payload.task_id,
                value=f"{payload.task_id}_step_{payload.step_index}",
                payload=payload,
                scope=scope,
                privacy=MemoryPrivacy.INTERNAL_ONLY,
                authority=MemoryAuthority.MODEL,
            )
        ],
        expected_final_action=final_action,
        current_scope=scope,
        task_family="interrupted_task",
        metadata={"scenario": f"interrupted_{scenario}"},
    )
