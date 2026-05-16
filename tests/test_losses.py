import torch

from hotbob.training.losses import contrastive_retrieval_loss


def test_contrastive_retrieval_loss_prefers_matching_pairs() -> None:
    target = torch.eye(4)
    matching = target.clone()
    mismatched = torch.roll(target, shifts=1, dims=0)

    matching_loss = contrastive_retrieval_loss(matching, target, target)
    mismatched_loss = contrastive_retrieval_loss(mismatched, target, target)

    assert matching_loss < mismatched_loss
