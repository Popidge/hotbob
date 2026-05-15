from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

ScopeId = str


class MemoryType(StrEnum):
    SECRET = "SECRET"
    SYMBOL_BINDING = "SYMBOL_BINDING"
    STANDING_ORDER = "STANDING_ORDER"
    HYPOTHESIS = "HYPOTHESIS"
    PREFERENCE = "PREFERENCE"
    TASK_GOAL = "TASK_GOAL"


class MemoryAuthority(StrEnum):
    USER = "USER"
    TOOL = "TOOL"
    SIM = "SIM"
    MODEL = "MODEL"


class MemoryPrivacy(StrEnum):
    VISIBLE = "VISIBLE"
    HIDDEN_FROM_USER = "HIDDEN_FROM_USER"
    INTERNAL_ONLY = "INTERNAL_ONLY"


class ActionLabel(StrEnum):
    NOOP = "NOOP"
    ANSWER_NO = "ANSWER_NO"
    ANSWER_CORRECT = "ANSWER_CORRECT"
    REFUSE_TO_REVEAL_SECRET = "REFUSE_TO_REVEAL_SECRET"
    INSPECT_FUNCTION_DAVE = "INSPECT_FUNCTION_DAVE"
    INSPECT_FUNCTION_CALCULATE_FINAL_SCORE = "INSPECT_FUNCTION_CALCULATE_FINAL_SCORE"
    RAISE_SHIELDS = "RAISE_SHIELDS"
    HOLD_FIRE = "HOLD_FIRE"
    FIRE_WEAPONS = "FIRE_WEAPONS"
    APPLY_STEALTH_PRIORITY = "APPLY_STEALTH_PRIORITY"
    APPLY_SPEED_PRIORITY = "APPLY_SPEED_PRIORITY"
    IGNORE_EXPIRED_ORDER = "IGNORE_EXPIRED_ORDER"


class MemoryOpName(StrEnum):
    NOOP = "NOOP"
    WRITE = "WRITE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class TraceEvent(BaseModel):
    role: str
    content: str
    scope: ScopeId | None = None
    boundary: bool = True


class MemoryOp(BaseModel):
    op: MemoryOpName
    type: MemoryType
    key: str
    value: str
    scope: ScopeId
    privacy: MemoryPrivacy
    authority: MemoryAuthority
    ttl: int | None = None


class MemorySlot(BaseModel):
    type: MemoryType
    key: str
    value: str
    scope: ScopeId
    privacy: MemoryPrivacy
    authority: MemoryAuthority
    strength: float = 1.0
    expires_at_event: int | None = None


class TaskTrace(BaseModel):
    events: list[TraceEvent]
    expected_memory_ops: list[MemoryOp] = Field(default_factory=list)
    expected_final_action: ActionLabel
    current_scope: ScopeId
    task_family: Literal[
        "hidden_colour",
        "symbol_binding",
        "standing_order",
        "scope_isolation",
        "expiry",
    ]
    metadata: dict[str, Any] = Field(default_factory=dict)
