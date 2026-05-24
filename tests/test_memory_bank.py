import torch

from hotbob.model import MemoryBank, MemoryRead


def test_memory_bank_write_update_decay_delete_and_scope_mask() -> None:
    bank = MemoryBank(num_slots=3, d_model=4)
    bank.reset(batch_size=1)
    bank.apply_write(
        0,
        0,
        torch.ones(4),
        type_id=1,
        scope_id=10,
        privacy_id=1,
        authority_id=2,
        payload_kind_id=3,
        payload_default_action_id=4,
        payload_winning_authority_level_id=5,
        payload_losing_authority_level_id=6,
    )
    bank.apply_write(0, 1, torch.ones(4) * 2, type_id=1, scope_id=20, privacy_id=1, authority_id=2)
    assert bank.active_mask(torch.tensor([10])).tolist() == [[True, False, False]]
    repeated = bank.repeat_batch(2)
    assert repeated.payload_kind_ids[:, 0].tolist() == [3, 3]
    assert repeated.payload_default_action_ids[:, 0].tolist() == [4, 4]
    assert repeated.payload_winning_authority_level_ids[:, 0].tolist() == [5, 5]
    assert repeated.payload_losing_authority_level_ids[:, 0].tolist() == [6, 6]
    bank.apply_update(0, 0, torch.zeros(4), gate=0.5)
    assert torch.allclose(bank.vectors[0, 0], torch.ones(4) * 0.5)
    bank.decay(0.5)
    assert bank.strength[0, 0].item() == 0.5
    bank.apply_delete(0, 0)
    assert not bank.occupied[0, 0].item()
    assert int(bank.payload_kind_ids[0, 0].item()) == 0


def test_memory_read_masks_wrong_scope() -> None:
    torch.manual_seed(0)
    bank = MemoryBank(num_slots=2, d_model=4)
    bank.reset(batch_size=1)
    bank.apply_write(0, 0, torch.ones(4), type_id=1, scope_id=1, privacy_id=1, authority_id=1)
    bank.apply_write(0, 1, torch.ones(4), type_id=1, scope_id=2, privacy_id=1, authority_id=1)
    read = MemoryRead(d_model=4)
    _, attn = read(torch.randn(1, 3, 4), bank, torch.tensor([1]))
    assert torch.all(attn[..., 1] < 1e-6)
