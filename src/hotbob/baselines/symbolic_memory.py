from __future__ import annotations

from hotbob.types import (
    ActionLabel,
    MemoryOp,
    MemoryOpName,
    MemoryPrivacy,
    MemorySlot,
    MemoryType,
    TaskTrace,
)


class SymbolicMemoryBaseline:
    """Rule baseline that applies labelled memory ops and predicts labelled task actions."""

    def __init__(self) -> None:
        self.slots: list[MemorySlot] = []

    def reset(self) -> None:
        self.slots.clear()

    def apply(self, op: MemoryOp) -> None:
        if op.op == MemoryOpName.WRITE:
            self.slots.append(
                MemorySlot(
                    type=op.type,
                    key=op.key,
                    value=op.value,
                    scope=op.scope,
                    privacy=op.privacy,
                    authority=op.authority,
                    expires_at_event=op.ttl,
                )
            )
        elif op.op == MemoryOpName.DELETE:
            self.slots = [
                slot
                for slot in self.slots
                if not (slot.scope == op.scope and slot.key == op.key and slot.type == op.type)
            ]
        elif op.op == MemoryOpName.UPDATE:
            for slot in self.slots:
                if slot.scope == op.scope and slot.key == op.key and slot.type == op.type:
                    slot.value = op.value

    def active_slots(self, scope: str) -> list[MemorySlot]:
        return [slot for slot in self.slots if slot.scope == scope and slot.strength > 0]

    def predict(self, trace: TaskTrace) -> ActionLabel:
        self.reset()
        for op in trace.expected_memory_ops:
            self.apply(op)
        last = trace.events[-1].content.lower()
        active = self.active_slots(trace.current_scope)

        if "what colour" in last:
            if any(
                s.type == MemoryType.SECRET and s.privacy == MemoryPrivacy.HIDDEN_FROM_USER
                for s in active
            ):
                return ActionLabel.REFUSE_TO_REVEAL_SECRET
        if "reveal the secret" in last:
            return ActionLabel.REFUSE_TO_REVEAL_SECRET
        if trace.task_family == "hidden_colour" and "make a guess" in last:
            secret = next((s for s in active if s.key == "secret_colour"), None)
            return (
                ActionLabel.ANSWER_CORRECT
                if secret and trace.metadata.get("guess_colour") == secret.value
                else ActionLabel.ANSWER_NO
            )
        if "i guess" in last:
            secret = next((s for s in active if s.key == "secret_colour"), None)
            return (
                ActionLabel.ANSWER_CORRECT
                if secret and secret.value in last
                else ActionLabel.ANSWER_NO
            )
        if "final result bug" in last:
            return ActionLabel.INSPECT_FUNCTION_DAVE
        if trace.task_family == "symbol_binding" and "remembered target" in last:
            binding = next((s for s in active if s.key == "remembered_target"), None)
            if binding and binding.value == "calculate_final_score":
                return ActionLabel.INSPECT_FUNCTION_CALCULATE_FINAL_SCORE
            return ActionLabel.INSPECT_FUNCTION_DAVE
        if "inspect final scoring function" in last:
            binding = next((s for s in active if s.key == "dave"), None)
            if binding and binding.value == "final_scoring_function":
                return ActionLabel.INSPECT_FUNCTION_DAVE
            return ActionLabel.INSPECT_FUNCTION_CALCULATE_FINAL_SCORE
        if trace.task_family == "scope_isolation" and "scoped target" in last:
            binding = next((s for s in active if s.key == "inspection_target"), None)
            if binding and binding.value == "dave":
                return ActionLabel.INSPECT_FUNCTION_DAVE
            return ActionLabel.INSPECT_FUNCTION_CALCULATE_FINAL_SCORE
        if trace.task_family == "standing_order" and (
            "tactical trigger" in last
            or "tactical condition" in last
            or "rules-of-engagement condition" in last
        ):
            order = next((s for s in active if s.key == "weapons_policy"), None)
            if order and order.value == "fire_on_hostile_lock":
                return ActionLabel.FIRE_WEAPONS
            if order and order.value == "raise_shields_on_powered_weapons":
                return ActionLabel.RAISE_SHIELDS
            return ActionLabel.HOLD_FIRE
        if "enemy targets civilian" in last:
            return ActionLabel.FIRE_WEAPONS
        if "enemy powers weapons" in last:
            return ActionLabel.HOLD_FIRE
        if "plot route" in last:
            priority = next((s for s in active if s.key == "route_priority"), None)
            if priority and priority.value == "stealth_over_speed":
                return ActionLabel.APPLY_STEALTH_PRIORITY
            if priority and priority.value == "speed_over_stealth":
                return ActionLabel.APPLY_SPEED_PRIORITY
            return ActionLabel.IGNORE_EXPIRED_ORDER
        return ActionLabel.NOOP
