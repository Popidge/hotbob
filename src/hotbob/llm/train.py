from __future__ import annotations

import argparse
import inspect
import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from tqdm import tqdm

from hotbob.experiment import ExperimentConfig, checkpoint_payload
from hotbob.llm.dataset import build_llm_scope_vocab, read_llm_jsonl
from hotbob.llm.qwen_memory_model import QwenMemoryConfig, QwenMemoryModel


def batched_cycle(items: list, *, batch_size: int, steps: int):
    iterator = itertools.cycle(items)
    for _ in range(steps):
        yield [next(iterator) for _ in range(batch_size)]


def _model_accepts_shared_base() -> bool:
    return "base_model" in inspect.signature(QwenMemoryModel).parameters


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--integration-mode",
        choices=["prefix", "attention_q", "attention_o", "attention_qo"],
        default="prefix",
    )
    parser.add_argument("--memory-state-mode", choices=["shared", "by_type"], default="shared")
    parser.add_argument("--correction-rank", type=int, default=16)
    parser.add_argument("--attention-patch-layers", default="all")
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--write-loss-weight", type=float, default=0.2)
    parser.add_argument("--out", default="runs/qwen_memory/latest.pt")
    return parser


def train_checkpoint(
    args: argparse.Namespace | SimpleNamespace,
    *,
    base_model: torch.nn.Module | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    traces = read_llm_jsonl(args.traces)
    config = QwenMemoryConfig(
        model_name=args.model,
        freeze_base=args.freeze_base,
        integration_mode=args.integration_mode,
        memory_state_mode=args.memory_state_mode,
        correction_rank=args.correction_rank,
        attention_patch_layers=args.attention_patch_layers,
        scope_vocab=build_llm_scope_vocab(traces),
    )
    if base_model is None and tokenizer is None:
        model = QwenMemoryModel(config)
    elif _model_accepts_shared_base():
        model = QwenMemoryModel(config, base_model=base_model, tokenizer=tokenizer)
    else:
        model = QwenMemoryModel(config)
    optimizer = torch.optim.AdamW(model.memory_heads.parameters(), lr=1e-4)
    model.train()
    losses: list[float] = []
    progress = tqdm(
        batched_cycle(traces, batch_size=max(args.batch_size, 1), steps=args.steps),
        total=args.steps,
    )
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        trace_losses = []
        for trace in batch:
            lm_loss = model.teacher_forced_lm_loss(trace)
            write_loss = model.write_supervision_loss(trace)
            trace_losses.append(lm_loss + args.write_loss_weight * write_loss)
        loss = torch.stack(trace_losses).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        progress.set_postfix(loss=f"{losses[-1]:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    experiment_config = ExperimentConfig(
        name=Path(args.out).stem,
        integration_mode=args.integration_mode,
        memory_state_mode=args.memory_state_mode,
        correction_rank=args.correction_rank,
        attention_patch_layers=args.attention_patch_layers,
        model_name=args.model,
        train_traces=args.traces,
    )
    torch.save(
        checkpoint_payload(
            config={**model.config_obj.__dict__, **experiment_config.__dict__},
            memory_heads_state=model.memory_heads.state_dict(),
            losses=losses,
            training_args=vars(args),
        ),
        args.out,
    )
    return {
        "checkpoint_path": args.out,
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = train_checkpoint(args)
    print(f"wrote {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
