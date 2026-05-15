from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 256
    num_memory_slots: int = 32
