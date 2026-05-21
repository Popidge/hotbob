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
        num_payload_kinds: int = 1,
        num_policy_actions: int = 1,
        num_policy_triggers: int = 1,
        num_expiry_policies: int = 1,
        num_authority_levels: int = 1,
        num_tool_names: int = 1,
        num_route_steps: int = 1,
    ) -> None:
        super().__init__()
        self.op_head = nn.Linear(d_model, 4)
        self.slot_head = nn.Linear(d_model, num_slots)
        self.type_head = nn.Linear(d_model, num_types)
        self.scope_head = nn.Linear(d_model, num_scopes)
        self.privacy_head = nn.Linear(d_model, num_privacy)
        self.authority_head = nn.Linear(d_model, num_authority)
        self.value_class_head = nn.Linear(d_model, num_value_classes)
        self.payload_kind_head = nn.Linear(d_model, num_payload_kinds)
        self.payload_default_action_head = nn.Linear(d_model, num_policy_actions)
        self.payload_trigger_head = nn.Linear(d_model, num_policy_triggers)
        self.payload_exception_head = nn.Linear(d_model, num_policy_triggers)
        self.payload_expiry_policy_head = nn.Linear(d_model, num_expiry_policies)
        self.payload_authority_level_head = nn.Linear(d_model, num_authority_levels)
        self.payload_tool_name_head = nn.Linear(d_model, num_tool_names)
        self.payload_route_step_head = nn.Linear(d_model, num_route_steps)
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
            "payload_kind_logits": self.payload_kind_head(boundary_hidden),
            "payload_default_action_logits": self.payload_default_action_head(boundary_hidden),
            "payload_trigger_logits": self.payload_trigger_head(boundary_hidden),
            "payload_exception_logits": self.payload_exception_head(boundary_hidden),
            "payload_expiry_policy_logits": self.payload_expiry_policy_head(boundary_hidden),
            "payload_authority_level_logits": self.payload_authority_level_head(boundary_hidden),
            "payload_tool_name_logits": self.payload_tool_name_head(boundary_hidden),
            "payload_route_step_logits": self.payload_route_step_head(boundary_hidden),
            "value_vector": self.value_head(boundary_hidden),
            "write_gate": torch.sigmoid(self.gate_head(boundary_hidden)),
        }
