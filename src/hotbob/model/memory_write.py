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
    ) -> None:
        super().__init__()
        self.op = nn.Linear(d_model, 4)
        self.slot = nn.Linear(d_model, num_slots)
        self.type = nn.Linear(d_model, num_types)
        self.scope = nn.Linear(d_model, num_scopes)
        self.privacy = nn.Linear(d_model, num_privacy)
        self.authority = nn.Linear(d_model, num_authority)
        self.value = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, 1)

    def forward(self, boundary_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "op_logits": self.op(boundary_hidden),
            "slot_logits": self.slot(boundary_hidden),
            "type_logits": self.type(boundary_hidden),
            "scope_logits": self.scope(boundary_hidden),
            "privacy_logits": self.privacy(boundary_hidden),
            "authority_logits": self.authority(boundary_hidden),
            "value_vector": self.value(boundary_hidden),
            "write_gate": torch.sigmoid(self.gate(boundary_hidden)),
        }
