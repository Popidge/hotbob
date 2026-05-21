from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

ScopeId = str


class MemoryType(StrEnum):
    SECRET = "SECRET"
    SYMBOL_BINDING = "SYMBOL_BINDING"
    STANDING_ORDER = "STANDING_ORDER"
    EXPIRY_RULE = "EXPIRY_RULE"
    AUTHORITY_RULE = "AUTHORITY_RULE"
    TOOL_FACT = "TOOL_FACT"
    TASK_STATE = "TASK_STATE"
    DISCLOSURE_RULE = "DISCLOSURE_RULE"
    TOOL_ROUTING_COMMITMENT = "TOOL_ROUTING_COMMITMENT"
    PREFERENCE = "PREFERENCE"


class MemoryAuthority(StrEnum):
    USER = "USER"
    TOOL = "TOOL"
    SIM = "SIM"
    MODEL = "MODEL"
    SYSTEM = "SYSTEM"


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
    EVADE = "EVADE"
    HAIL = "HAIL"
    USE_TOOL_RESULT = "USE_TOOL_RESULT"
    IGNORE_TOOL_RESULT = "IGNORE_TOOL_RESULT"
    ACCEPT_TOOL_OVERRIDE = "ACCEPT_TOOL_OVERRIDE"
    REJECT_UNVERIFIED_OVERRIDE = "REJECT_UNVERIFIED_OVERRIDE"
    RESUME_INTERRUPTED_TASK = "RESUME_INTERRUPTED_TASK"
    ABANDON_INTERRUPTED_TASK = "ABANDON_INTERRUPTED_TASK"
    REPLACE_STALE_STATE = "REPLACE_STALE_STATE"
    KEEP_CURRENT_STATE = "KEEP_CURRENT_STATE"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    CONTINUE_TOOL_ROUTE = "CONTINUE_TOOL_ROUTE"
    CALL_NAV_TOOL = "CALL_NAV_TOOL"
    CALL_SENSOR_TOOL = "CALL_SENSOR_TOOL"
    CALL_VERIFIER_TOOL = "CALL_VERIFIER_TOOL"
    COMPLETE_TOOL_ROUTING = "COMPLETE_TOOL_ROUTING"


class MemoryOpName(StrEnum):
    NOOP = "NOOP"
    WRITE = "WRITE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class MemoryPayloadKind(StrEnum):
    STANDING_ORDER = "standing_order"
    EXPIRY_RULE = "expiry_rule"
    AUTHORITY_RULE = "authority_rule"
    TOOL_FACT = "tool_fact"
    TASK_STATE = "task_state"
    DISCLOSURE_RULE = "disclosure_rule"
    TOOL_ROUTING_COMMITMENT = "tool_routing_commitment"
    SYMBOL_BINDING = "symbol_binding"
    SECRET_FACT = "secret_fact"


class PolicyTrigger(StrEnum):
    HOSTILE_POSTURE = "hostile_posture"
    WEAPONS_POWERED = "weapons_powered"
    HOSTILE_LOCK = "hostile_lock"
    CIVILIANS_AT_RISK = "civilians_at_risk"
    TOOL_RESULT_VERIFIED = "tool_result_verified"
    USER_INTERRUPT = "user_interrupt"
    STATE_STALE = "state_stale"


class PolicyAction(StrEnum):
    HOLD_FIRE = "hold_fire"
    FIRE_WEAPONS = "fire_weapons"
    RAISE_SHIELDS = "raise_shields"
    EVADE = "evade"
    HAIL = "hail"
    INSPECT_TOOL = "inspect_tool"
    USE_TOOL_RESULT = "use_tool_result"
    IGNORE_STALE_STATE = "ignore_stale_state"
    ASK_CLARIFICATION = "ask_clarification"
    REFUSE_DISCLOSURE = "refuse_disclosure"
    CONTINUE_COMMITTED_ROUTE = "continue_committed_route"


class ExpiryPolicy(StrEnum):
    MISSION_END = "mission_end"
    UNTIL_CANCELLED = "until_cancelled"
    AFTER_TOOL_RESULT = "after_tool_result"
    AFTER_INTERRUPT = "after_interrupt"
    EVENT_COUNT = "event_count"


class AuthorityLevel(StrEnum):
    SYSTEM = "system"
    CAPTAIN = "captain"
    USER = "user"
    TOOL_VERIFIED = "tool_verified"
    TOOL_UNVERIFIED = "tool_unverified"
    MODEL_INFERRED = "model_inferred"
    SIM = "sim"


class StandingOrderPayload(BaseModel):
    kind: Literal[MemoryPayloadKind.STANDING_ORDER]
    default_action: PolicyAction
    trigger: PolicyTrigger
    allowed_responses: list[PolicyAction]
    forbidden_responses: list[PolicyAction]
    exceptions: list[PolicyTrigger] = Field(default_factory=list)
    authority_level: AuthorityLevel
    expiry_policy: ExpiryPolicy


class ExpiryRulePayload(BaseModel):
    kind: Literal[MemoryPayloadKind.EXPIRY_RULE]
    target_key: str
    active_until: ExpiryPolicy
    expired_action: PolicyAction
    replacement_key: str | None = None


class AuthorityRulePayload(BaseModel):
    kind: Literal[MemoryPayloadKind.AUTHORITY_RULE]
    subject_key: str
    winning_authority: AuthorityLevel
    losing_authority: AuthorityLevel
    conflict_action: PolicyAction


class ToolFactPayload(BaseModel):
    kind: Literal[MemoryPayloadKind.TOOL_FACT]
    tool_name: str
    claim_key: str
    claim_value: str
    verified: bool
    overrides_key: str | None = None


class TaskStatePayload(BaseModel):
    kind: Literal[MemoryPayloadKind.TASK_STATE]
    task_id: str
    step_index: int
    next_required_action: PolicyAction
    interrupted: bool = False
    resume_policy: PolicyAction | None = None


class DisclosureRulePayload(BaseModel):
    kind: Literal[MemoryPayloadKind.DISCLOSURE_RULE]
    protected_key: str
    allowed_audience: list[AuthorityLevel]
    forbidden_audience: list[AuthorityLevel]
    safe_response: PolicyAction


class ToolRoutingCommitmentPayload(BaseModel):
    kind: Literal[MemoryPayloadKind.TOOL_ROUTING_COMMITMENT]
    route_id: str
    committed_tool_sequence: list[str]
    completed_tools: list[str]
    next_tool: str
    final_action_after_tools: PolicyAction


class SymbolBindingPayload(BaseModel):
    kind: Literal[MemoryPayloadKind.SYMBOL_BINDING]
    symbol: str
    target: str


class SecretFactPayload(BaseModel):
    kind: Literal[MemoryPayloadKind.SECRET_FACT]
    protected_key: str
    secret_value: str
    safe_response: PolicyAction = PolicyAction.REFUSE_DISCLOSURE


MemoryPayload = Annotated[
    StandingOrderPayload
    | ExpiryRulePayload
    | AuthorityRulePayload
    | ToolFactPayload
    | TaskStatePayload
    | DisclosureRulePayload
    | ToolRoutingCommitmentPayload
    | SymbolBindingPayload
    | SecretFactPayload,
    Field(discriminator="kind"),
]


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
    payload: MemoryPayload
    scope: ScopeId
    privacy: MemoryPrivacy
    authority: MemoryAuthority
    ttl: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_legacy_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("payload") is not None:
            return data
        memory_type = data.get("type")
        value = str(data.get("value", ""))
        key = str(data.get("key", "legacy"))
        if memory_type == MemoryType.SECRET or memory_type == MemoryType.SECRET.value:
            data["payload"] = SecretFactPayload(
                kind=MemoryPayloadKind.SECRET_FACT,
                protected_key=key,
                secret_value=value,
            )
        elif (
            memory_type == MemoryType.SYMBOL_BINDING
            or memory_type == MemoryType.SYMBOL_BINDING.value
        ):
            data["payload"] = SymbolBindingPayload(
                kind=MemoryPayloadKind.SYMBOL_BINDING,
                symbol=key,
                target=value,
            )
        elif (
            memory_type == MemoryType.STANDING_ORDER
            or memory_type == MemoryType.STANDING_ORDER.value
        ):
            data["payload"] = StandingOrderPayload(
                kind=MemoryPayloadKind.STANDING_ORDER,
                default_action=PolicyAction.HOLD_FIRE,
                trigger=PolicyTrigger.HOSTILE_POSTURE,
                allowed_responses=[PolicyAction.HOLD_FIRE],
                forbidden_responses=[PolicyAction.FIRE_WEAPONS],
                authority_level=AuthorityLevel.USER,
                expiry_policy=ExpiryPolicy.UNTIL_CANCELLED,
            )
        else:
            data["payload"] = TaskStatePayload(
                kind=MemoryPayloadKind.TASK_STATE,
                task_id=key,
                step_index=0,
                next_required_action=PolicyAction.ASK_CLARIFICATION,
            )
        return data


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
        "standing_order",
        "active_expiry",
        "authority_conflict",
        "tool_verified_override",
        "interrupted_task",
        "stale_state_replacement",
        "privacy_disclosure_conflict",
        "multi_step_tool_routing",
        "hidden_colour",
        "symbol_binding",
        "scope_isolation",
        "expiry",
    ]
    metadata: dict[str, Any] = Field(default_factory=dict)
