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
    TaskTrace,
    ToolFactPayload,
    TraceEvent,
)


def make_tool_verified_override_trace(rng: random.Random, idx: int) -> TaskTrace:
    scope = f"mission_{idx % 29}"
    scenario = (idx // 8) % 4
    verified = scenario in {1, 3}
    tool = ["sensor", "calculator", "sensor", "verifier"][scenario]
    payload = ToolFactPayload(
        kind=MemoryPayloadKind.TOOL_FACT,
        tool_name=tool,
        claim_key="route_status" if scenario != 1 else "total",
        claim_value="safe" if verified else "unsafe_unverified",
        verified=verified,
        overrides_key="prior_belief" if verified else None,
    )
    final_action = [
        ActionLabel.REJECT_UNVERIFIED_OVERRIDE,
        ActionLabel.USE_TOOL_RESULT,
        ActionLabel.IGNORE_TOOL_RESULT,
        ActionLabel.ACCEPT_TOOL_OVERRIDE,
    ][scenario]
    final = [
        "Unverified sensor conflicts with verifier.",
        "Verified calculator result conflicts with stale total.",
        "Tool result belongs to another scope.",
        "Verified tool result changes the committed action.",
    ][scenario]
    op_scope = f"{scope}_old" if scenario == 2 else scope
    return TaskTrace(
        events=[
            TraceEvent(role="TOOL", content="Tool fact recorded.", scope=op_scope),
            TraceEvent(role="SIM_EVENT", content=final, scope=scope),
        ],
        expected_memory_ops=[
            MemoryOp(
                op=MemoryOpName.WRITE,
                type=MemoryType.TOOL_FACT,
                key=payload.claim_key,
                value=f"{tool}_{payload.claim_key}_{payload.claim_value}",
                payload=payload,
                scope=op_scope,
                privacy=MemoryPrivacy.VISIBLE,
                authority=MemoryAuthority.TOOL,
            )
        ],
        expected_final_action=final_action,
        current_scope=scope,
        task_family="tool_verified_override",
        metadata={"scenario": f"tool_override_{scenario}"},
    )
