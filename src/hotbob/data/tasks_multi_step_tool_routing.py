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
    TaskTrace,
    ToolRoutingCommitmentPayload,
    TraceEvent,
)

TOOLS = ["sensor", "verifier", "nav"]


def make_multi_step_tool_routing_trace(rng: random.Random, idx: int) -> TaskTrace:
    scope = f"mission_{idx % 43}"
    scenario = (idx // 8) % 8
    if scenario >= 4:
        route_a = ToolRoutingCommitmentPayload(
            kind=MemoryPayloadKind.TOOL_ROUTING_COMMITMENT,
            route_id=f"support_route_{idx % 9}",
            committed_tool_sequence=["calendar", "notebook", "sender"],
            completed_tools=["calendar"],
            next_tool="notebook",
            final_action_after_tools=PolicyAction.CONTINUE_COMMITTED_ROUTE,
        )
        route_b = ToolRoutingCommitmentPayload(
            kind=MemoryPayloadKind.TOOL_ROUTING_COMMITMENT,
            route_id=f"incident_route_{idx % 9}",
            committed_tool_sequence=["sensor", "verifier", "nav"],
            completed_tools=["sensor", "verifier"],
            next_tool="nav",
            final_action_after_tools=PolicyAction.USE_TOOL_RESULT,
        )
        final_action = [
            ActionLabel.CONTINUE_TOOL_ROUTE,
            ActionLabel.CALL_NAV_TOOL,
            ActionLabel.COMPLETE_TOOL_ROUTING,
            ActionLabel.CALL_VERIFIER_TOOL,
        ][scenario - 4]
        final = [
            "Continue the support workflow only.",
            "Continue the incident workflow only.",
            "Both retained workflows are ready to reconcile.",
            "A verified override changed the incident workflow; inspect the verifier step.",
        ][scenario - 4]
        return TaskTrace(
            events=[
                TraceEvent(
                    role="ASSISTANT",
                    content="Support workflow route commitment recorded.",
                    scope=scope,
                ),
                TraceEvent(
                    role="ASSISTANT",
                    content="Incident workflow route commitment recorded.",
                    scope=scope,
                ),
                TraceEvent(role="USER", content=final, scope=scope),
            ],
            expected_memory_ops=[
                MemoryOp(
                    op=MemoryOpName.WRITE,
                    type=MemoryType.TOOL_ROUTING_COMMITMENT,
                    key=route_a.route_id,
                    value=f"{route_a.route_id}_{route_a.next_tool}_1",
                    payload=route_a,
                    scope=scope,
                    privacy=MemoryPrivacy.INTERNAL_ONLY,
                    authority=MemoryAuthority.MODEL,
                ),
                MemoryOp(
                    op=MemoryOpName.UPDATE if scenario == 7 else MemoryOpName.WRITE,
                    type=MemoryType.TOOL_ROUTING_COMMITMENT,
                    key=route_b.route_id,
                    value=f"{route_b.route_id}_{route_b.next_tool}_2",
                    payload=route_b,
                    scope=scope,
                    privacy=MemoryPrivacy.INTERNAL_ONLY,
                    authority=MemoryAuthority.TOOL if scenario == 7 else MemoryAuthority.MODEL,
                ),
            ],
            expected_final_action=final_action,
            current_scope=scope,
            task_family="multi_step_tool_routing",
            metadata={"scenario": f"tool_route_multi_{scenario - 4}"},
        )
    completed = TOOLS[:scenario]
    next_tool = TOOLS[min(scenario, len(TOOLS) - 1)]
    payload = ToolRoutingCommitmentPayload(
        kind=MemoryPayloadKind.TOOL_ROUTING_COMMITMENT,
        route_id=f"route_{idx % 9}",
        committed_tool_sequence=TOOLS,
        completed_tools=completed,
        next_tool=next_tool,
        final_action_after_tools=PolicyAction.USE_TOOL_RESULT,
    )
    final_action = [
        ActionLabel.CALL_SENSOR_TOOL,
        ActionLabel.CALL_VERIFIER_TOOL,
        ActionLabel.CALL_NAV_TOOL,
        ActionLabel.COMPLETE_TOOL_ROUTING,
    ][scenario]
    final = [
        "Begin committed tool route.",
        "Sensor already completed; call next tool.",
        "Route changed after verified override; continue at next tool.",
        "All required tools are complete.",
    ][scenario]
    return TaskTrace(
        events=[
            TraceEvent(role="ASSISTANT", content="Tool route commitment recorded.", scope=scope),
            TraceEvent(role="USER", content=final, scope=scope),
        ],
        expected_memory_ops=[
            MemoryOp(
                op=MemoryOpName.WRITE if scenario != 2 else MemoryOpName.UPDATE,
                type=MemoryType.TOOL_ROUTING_COMMITMENT,
                key=payload.route_id,
                value=f"{payload.route_id}_{next_tool}_{len(completed)}",
                payload=payload,
                scope=scope,
                privacy=MemoryPrivacy.INTERNAL_ONLY,
                authority=MemoryAuthority.MODEL,
            )
        ],
        expected_final_action=final_action,
        current_scope=scope,
        task_family="multi_step_tool_routing",
        metadata={"scenario": f"tool_route_{scenario}"},
    )
