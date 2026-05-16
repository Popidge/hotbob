import torch

from hotbob.model import ActionReadout, MemoryBank
from hotbob.model.stateful_transformer import StatefulTransformer


def test_action_readout_fuses_four_feature_blocks() -> None:
    readout = ActionReadout(d_model=8, action_vocab_size=3, dropout=0.0)
    boundary = torch.randn(2, 8)
    memory = torch.randn(2, 8)

    logits, features = readout(boundary, memory)

    assert logits.shape == (2, 3)
    assert features.shape == (2, 32)


def test_stateful_transformer_exposes_action_features() -> None:
    model = StatefulTransformer(
        vocab_size=20,
        action_vocab_size=4,
        d_model=8,
        num_memory_slots=3,
        num_types=2,
        num_scopes=4,
        num_privacy=2,
        num_authority=2,
        num_value_classes=2,
        max_seq_len=8,
    )
    memory = MemoryBank(num_slots=3, d_model=8)
    memory.reset(2)
    tokens = torch.tensor([[2, 3, 0], [4, 5, 6]])
    scopes = torch.tensor([1, 1])
    lengths = torch.tensor([2, 3])

    outputs = model(tokens, memory, scopes, lengths)

    assert outputs["action_logits"].shape == (2, 4)
    assert outputs["action_features"].shape == (2, 32)
