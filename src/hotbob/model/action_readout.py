from __future__ import annotations

import torch
from torch import nn


class ActionReadout(nn.Module):
    """Fuse boundary state and retrieved memory context for action prediction."""

    def __init__(self, d_model: int, action_vocab_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model * 4)
        self.hidden = nn.Linear(d_model * 4, d_model * 2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model * 2, action_vocab_size)

    def features(
        self,
        boundary_hidden: torch.Tensor,
        memory_context: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                boundary_hidden,
                memory_context,
                boundary_hidden * memory_context,
                torch.abs(boundary_hidden - memory_context),
            ],
            dim=-1,
        )

    def forward(
        self,
        boundary_hidden: torch.Tensor,
        memory_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(boundary_hidden, memory_context)
        hidden = self.hidden(self.norm(features))
        hidden = self.dropout(self.activation(hidden))
        return self.output(hidden), features
