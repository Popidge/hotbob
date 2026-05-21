from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    integration_mode: str
    memory_state_mode: str
    correction_rank: int
    attention_patch_layers: str
    model_name: str
    seed: int | None = None
    train_traces: str | None = None
    eval_traces: str | None = None


def run_dir(base: str | Path, config: ExperimentConfig) -> Path:
    return Path(base) / config.name


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checkpoint_payload(
    *,
    config: ExperimentConfig | dict[str, Any],
    memory_heads_state: dict[str, Any],
    losses: list[float],
    training_args: dict[str, Any],
    lora_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_dict = asdict(config) if isinstance(config, ExperimentConfig) else dict(config)
    payload = {
        "memory_heads_state": memory_heads_state,
        "config": config_dict,
        "git_commit": _git_commit(),
        "losses": losses,
        "loss": losses[-1] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "training_args": training_args,
    }
    if lora_state is not None:
        payload["lora_state"] = lora_state
    return payload
