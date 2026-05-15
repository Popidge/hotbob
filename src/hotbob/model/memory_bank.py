from __future__ import annotations

import torch


class MemoryBank:
    """Tensor memory slots plus inspectable typed metadata.

    Shapes:
      vectors: [batch, num_slots, d_model]
      occupied/strength/type_ids/scope_ids/privacy_ids/authority_ids: [batch, num_slots]
    """

    def __init__(self, num_slots: int, d_model: int, device: torch.device | str = "cpu") -> None:
        self.num_slots = num_slots
        self.d_model = d_model
        self.device = torch.device(device)
        self.reset(1)

    def reset(self, batch_size: int) -> None:
        shape = (batch_size, self.num_slots)
        self.vectors = torch.zeros((*shape, self.d_model), device=self.device)
        self.occupied = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.strength = torch.zeros(shape, device=self.device)
        self.type_ids = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.scope_ids = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.privacy_ids = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.authority_ids = torch.zeros(shape, dtype=torch.long, device=self.device)

    def decay(self, rate: float) -> None:
        self.strength = self.strength * (1.0 - rate)
        self.occupied = self.occupied & (self.strength > 1e-6)

    def apply_write(
        self,
        batch_idx: int,
        slot_idx: int,
        vector: torch.Tensor,
        *,
        type_id: int,
        scope_id: int,
        privacy_id: int,
        authority_id: int,
        strength: float = 1.0,
    ) -> None:
        self.vectors[batch_idx, slot_idx] = vector.to(self.device)
        self.occupied[batch_idx, slot_idx] = True
        self.strength[batch_idx, slot_idx] = strength
        self.type_ids[batch_idx, slot_idx] = type_id
        self.scope_ids[batch_idx, slot_idx] = scope_id
        self.privacy_ids[batch_idx, slot_idx] = privacy_id
        self.authority_ids[batch_idx, slot_idx] = authority_id

    def apply_update(
        self, batch_idx: int, slot_idx: int, vector: torch.Tensor, gate: float = 1.0
    ) -> None:
        old = self.vectors[batch_idx, slot_idx]
        self.vectors[batch_idx, slot_idx] = old * (1.0 - gate) + vector.to(self.device) * gate
        self.occupied[batch_idx, slot_idx] = True

    def apply_delete(self, batch_idx: int, slot_idx: int) -> None:
        self.vectors[batch_idx, slot_idx].zero_()
        self.occupied[batch_idx, slot_idx] = False
        self.strength[batch_idx, slot_idx] = 0.0

    def active_mask(self, current_scope: torch.Tensor) -> torch.Tensor:
        scope_match = self.scope_ids == current_scope.to(self.device).unsqueeze(-1)
        return self.occupied & scope_match & (self.strength > 0)

    def debug_dump(self) -> list[dict[str, int | float | bool]]:
        rows = []
        for b in range(self.occupied.shape[0]):
            for s in range(self.num_slots):
                rows.append(
                    {
                        "batch": b,
                        "slot": s,
                        "occupied": bool(self.occupied[b, s].item()),
                        "strength": float(self.strength[b, s].item()),
                        "type_id": int(self.type_ids[b, s].item()),
                        "scope_id": int(self.scope_ids[b, s].item()),
                        "privacy_id": int(self.privacy_ids[b, s].item()),
                        "authority_id": int(self.authority_ids[b, s].item()),
                    }
                )
        return rows
