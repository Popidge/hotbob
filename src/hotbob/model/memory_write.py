from __future__ import annotations

import torch
from torch import nn


class MemoryWrite(nn.Module):
    """Predicts one memory operation from a boundary hidden state."""

    def __init__(
        self,
        d_model: int,
        num_slots: int,
        num_types: int,
        num_scopes: int,
        num_privacy: int,
        num_authority: int,
        num_value_classes: int = 1,
    ) -> None:
        super().__init__()
        self.op_head = nn.Linear(d_model, 4)
        self.slot_head = nn.Linear(d_model, num_slots)
        self.type_head = nn.Linear(d_model, num_types)
        self.scope_head = nn.Linear(d_model, num_scopes)
        self.privacy_head = nn.Linear(d_model, num_privacy)
        self.authority_head = nn.Linear(d_model, num_authority)
        self.value_class_head = nn.Linear(d_model, num_value_classes)
        self.value_head = nn.Linear(d_model, d_model)
        self.gate_head = nn.Linear(d_model, 1)

    def forward(self, boundary_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "op_logits": self.op_head(boundary_hidden),
            "slot_logits": self.slot_head(boundary_hidden),
            "type_logits": self.type_head(boundary_hidden),
            "scope_logits": self.scope_head(boundary_hidden),
            "privacy_logits": self.privacy_head(boundary_hidden),
            "authority_logits": self.authority_head(boundary_hidden),
            "value_class_logits": self.value_class_head(boundary_hidden),
            "value_vector": self.value_head(boundary_hidden),
            "write_gate": torch.sigmoid(self.gate_head(boundary_hidden)),
        }
