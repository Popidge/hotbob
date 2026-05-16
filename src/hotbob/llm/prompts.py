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
}


def final_prompt_from_trace(trace: TaskTrace) -> str:
    event = trace.events[-1]
    return f"{event.role}: {event.content}"


def target_text_from_trace(trace: TaskTrace) -> str:
    return ACTION_TEXT[trace.expected_final_action]
