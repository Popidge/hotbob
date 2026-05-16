from __future__ import annotations

import torch
from torch.nn import functional as F


def contrastive_retrieval_loss(
    memory_context: torch.Tensor,
    target_value: torch.Tensor,
    negatives: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE loss aligning retrieved memory contexts with target value vectors."""

    query = F.normalize(memory_context, dim=-1)
    positive = F.normalize(target_value, dim=-1)
    candidates = positive if negatives is None else F.normalize(negatives, dim=-1)
    logits = query @ candidates.T / temperature
    labels = torch.arange(query.shape[0], device=query.device)
    return F.cross_entropy(logits, labels)
