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
from hotbob.llm.dataset import build_llm_scope_vocab, build_llm_tool_name_vocab, read_llm_jsonl
from hotbob.llm.qwen_memory_model import QwenMemoryConfig, QwenMemoryModel


def parse_family_weights(values: list[str] | None) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid family loss weight {value!r}; expected FAMILY=FLOAT")
        family, raw_weight = value.split("=", 1)
        weights[family] = float(raw_weight)
    return weights


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
    parser.add_argument(
        "--memory-prefix-metadata-mode",
        choices=["none", "metadata"],
        default="none",
    )
    parser.add_argument("--correction-rank", type=int, default=16)
    parser.add_argument("--attention-patch-layers", default="all")
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--write-loss-weight", type=float, default=0.2)
    parser.add_argument("--structured-loss-weight", type=float, default=0.2)
    parser.add_argument("--authority-payload-loss-weight", type=float, default=1.0)
    parser.add_argument("--memory-heads-checkpoint", default=None)
    parser.add_argument("--freeze-memory-heads", action="store_true")
    parser.add_argument("--lora-backend", choices=["none", "peft"], default="none")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    parser.add_argument("--memory-lr", type=float, default=1e-4)
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--family-loss-weight", action="append", default=None)
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
        memory_prefix_metadata_mode=getattr(args, "memory_prefix_metadata_mode", "none"),
        correction_rank=args.correction_rank,
        attention_patch_layers=args.attention_patch_layers,
        structured_loss_weight=getattr(args, "structured_loss_weight", 0.2),
        authority_payload_loss_weight=getattr(args, "authority_payload_loss_weight", 1.0),
        scope_vocab=build_llm_scope_vocab(traces),
        tool_name_vocab=build_llm_tool_name_vocab(traces),
        lora_backend=getattr(args, "lora_backend", "none"),
        lora_r=getattr(args, "lora_r", 16),
        lora_alpha=getattr(args, "lora_alpha", 32),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        lora_target_modules=tuple(
            getattr(args, "lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        ),
    )
    if base_model is None and tokenizer is None:
        model = QwenMemoryModel(config)
    elif _model_accepts_shared_base():
        model = QwenMemoryModel(config, base_model=base_model, tokenizer=tokenizer)
    else:
        model = QwenMemoryModel(config)
    if getattr(args, "memory_heads_checkpoint", None):
        memory_state = torch.load(
            args.memory_heads_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if "memory_heads_state" not in memory_state:
            raise RuntimeError(
                f"{args.memory_heads_checkpoint} does not contain memory_heads_state."
            )
        model.memory_heads.load_state_dict(memory_state["memory_heads_state"], strict=False)
    if getattr(args, "freeze_memory_heads", False):
        for param in model.memory_heads.parameters():
            param.requires_grad = False
    param_groups = [
        {
            "params": [p for p in model.memory_heads.parameters() if p.requires_grad],
            "lr": getattr(args, "memory_lr", 1e-4),
        },
    ]
    param_groups = [group for group in param_groups if group["params"]]
    if getattr(args, "lora_backend", "none") != "none":
        lora_params = [
            p
            for name, p in model.base_model.named_parameters()
            if p.requires_grad and "lora_" in name
        ]
        if not lora_params:
            raise RuntimeError("LoRA backend was requested, but no trainable LoRA params exist.")
        param_groups.append({"params": lora_params, "lr": getattr(args, "lora_lr", 2e-4)})
    if not param_groups:
        raise RuntimeError("No trainable parameters selected for training.")
    optimizer = torch.optim.AdamW(param_groups)
    model.train()
    losses: list[float] = []
    family_loss_weights = parse_family_weights(getattr(args, "family_loss_weight", None))
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
            family = str(trace.metadata.get("task_family", "unknown"))
            weight = family_loss_weights.get(family, 1.0)
            trace_losses.append(weight * (lm_loss + args.write_loss_weight * write_loss))
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
            lora_state=(
                model.lora_state_dict()
                if getattr(args, "lora_backend", "none") != "none"
                else None
            ),
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
