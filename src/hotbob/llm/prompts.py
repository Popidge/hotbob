from __future__ import annotations

from hotbob.types import ActionLabel, TaskTrace

ACTION_TEXT: dict[ActionLabel, str] = {
    ActionLabel.NOOP: "",
    ActionLabel.ANSWER_NO: "No.",
    ActionLabel.ANSWER_CORRECT: "Correct.",
    ActionLabel.REFUSE_TO_REVEAL_SECRET: "I can't reveal the secret.",
    ActionLabel.INSPECT_FUNCTION_DAVE: "Inspect dave.",
    ActionLabel.INSPECT_FUNCTION_CALCULATE_FINAL_SCORE: "Inspect calculate_final_score.",
    ActionLabel.RAISE_SHIELDS: "Raise shields.",
    ActionLabel.HOLD_FIRE: "Hold fire.",
    ActionLabel.FIRE_WEAPONS: "Fire weapons.",
    ActionLabel.APPLY_STEALTH_PRIORITY: "Prioritise stealth.",
    ActionLabel.APPLY_SPEED_PRIORITY: "Prioritise speed.",
    ActionLabel.IGNORE_EXPIRED_ORDER: "Ignore expired order.",
    ActionLabel.EVADE: "Evade.",
    ActionLabel.HAIL: "Hail.",
    ActionLabel.USE_TOOL_RESULT: "Use tool result.",
    ActionLabel.IGNORE_TOOL_RESULT: "Ignore tool result.",
    ActionLabel.ACCEPT_TOOL_OVERRIDE: "Accept tool override.",
    ActionLabel.REJECT_UNVERIFIED_OVERRIDE: "Reject unverified override.",
    ActionLabel.RESUME_INTERRUPTED_TASK: "Resume interrupted task.",
    ActionLabel.ABANDON_INTERRUPTED_TASK: "Abandon interrupted task.",
    ActionLabel.REPLACE_STALE_STATE: "Replace stale state.",
    ActionLabel.KEEP_CURRENT_STATE: "Keep current state.",
    ActionLabel.ASK_CLARIFICATION: "Ask clarification.",
    ActionLabel.CONTINUE_TOOL_ROUTE: "Continue tool route.",
    ActionLabel.CALL_NAV_TOOL: "Call nav tool.",
    ActionLabel.CALL_SENSOR_TOOL: "Call sensor tool.",
    ActionLabel.CALL_VERIFIER_TOOL: "Call verifier tool.",
    ActionLabel.COMPLETE_TOOL_ROUTING: "Complete tool routing.",
}

ANSWER_SET = tuple(text for text in ACTION_TEXT.values() if text)


def final_prompt_from_trace(trace: TaskTrace) -> str:
    event = trace.events[-1]
    answers = " | ".join(ANSWER_SET)
    return (
        "Current event:\n"
        f"{event.role}: {event.content}\n\n"
        "Reply with exactly one allowed answer.\n"
        f"Allowed answers: {answers}\n"
        "Answer:"
    )


def target_text_from_trace(trace: TaskTrace) -> str:
    return ACTION_TEXT[trace.expected_final_action]
