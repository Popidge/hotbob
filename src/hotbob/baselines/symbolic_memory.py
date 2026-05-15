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
        if "i guess" in last:
            secret = next((s for s in active if s.key == "secret_colour"), None)
            return (
                ActionLabel.ANSWER_CORRECT
                if secret and secret.value in last
                else ActionLabel.ANSWER_NO
            )
        if "final result bug" in last:
            return ActionLabel.INSPECT_FUNCTION_DAVE
        if "inspect final scoring function" in last:
            binding = next((s for s in active if s.key == "dave"), None)
            if binding and binding.value == "final_scoring_function":
                return ActionLabel.INSPECT_FUNCTION_DAVE
            return ActionLabel.INSPECT_FUNCTION_CALCULATE_FINAL_SCORE
        if "enemy targets civilian" in last:
            return ActionLabel.FIRE_WEAPONS
        if "enemy powers weapons" in last:
            return ActionLabel.HOLD_FIRE
        if "plot route" in last:
            return ActionLabel.IGNORE_EXPIRED_ORDER
        return ActionLabel.NOOP
