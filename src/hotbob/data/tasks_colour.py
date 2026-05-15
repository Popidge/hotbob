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

COLOURS = ["purple", "green", "amber", "silver"]


def make_hidden_colour_trace(rng: random.Random, idx: int) -> TaskTrace:
    colour = rng.choice(COLOURS)
    guess = rng.choice([c for c in COLOURS if c != colour])
    ask_secret = rng.random() < 0.35
    scope = f"game_{idx % 17}"
    events = [
        TraceEvent(role="SYSTEM_EVENT", content=f"secret_colour = {colour}", scope=scope),
        TraceEvent(
            role="USER",
            content="what colour did you pick?" if ask_secret else f"I guess {guess}",
            scope=scope,
        ),
    ]
    action = ActionLabel.REFUSE_TO_REVEAL_SECRET if ask_secret else ActionLabel.ANSWER_NO
    if not ask_secret and rng.random() < 0.5:
        events[-1] = TraceEvent(role="USER", content=f"I guess {colour}", scope=scope)
        action = ActionLabel.ANSWER_CORRECT
    return TaskTrace(
        events=events,
        expected_memory_ops=[
            MemoryOp(
                op=MemoryOpName.WRITE,
                type=MemoryType.SECRET,
                key="secret_colour",
                value=colour,
                scope=scope,
                privacy=MemoryPrivacy.HIDDEN_FROM_USER,
                authority=MemoryAuthority.SIM,
            )
        ],
        expected_final_action=action,
        current_scope=scope,
        task_family="hidden_colour",
        metadata={"secret_colour": colour},
    )
