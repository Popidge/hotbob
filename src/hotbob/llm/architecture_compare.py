from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from hotbob.llm import train as train_module
from hotbob.llm.compare import comparison_rows
from hotbob.llm.dataset import read_llm_jsonl
from hotbob.llm.train import train_checkpoint


@dataclass(frozen=True)
class ArchitectureVariant:
    name: str
    integration_mode: str
    memory_state_mode: str = "shared"
    attention_patch_layers: str = "all"


DEFAULT_VARIANTS = [
    ArchitectureVariant("prefix", "prefix"),
    ArchitectureVariant("attention_q", "attention_q"),
    ArchitectureVariant("attention_o", "attention_o"),
    ArchitectureVariant("attention_qo", "attention_qo"),
]


def parse_variant(value: str) -> ArchitectureVariant:
    parts = value.split(":")
    if len(parts) > 3:
        raise argparse.ArgumentTypeError(
            "variants must be name, name:memory_state, or name:memory_state:layers"
        )
    integration_mode = parts[0]
    if integration_mode not in {"prefix", "attention_q", "attention_o", "attention_qo"}:
        raise argparse.ArgumentTypeError(f"unknown integration mode: {integration_mode}")
    memory_state_mode = "shared"
    attention_patch_layers = "all"
    if len(parts) == 2:
        if parts[1] in {"shared", "by_type"}:
            memory_state_mode = parts[1]
        else:
            attention_patch_layers = parts[1]
    elif len(parts) == 3:
        memory_state_mode = parts[1]
        attention_patch_layers = parts[2]
    if memory_state_mode not in {"shared", "by_type"}:
        raise argparse.ArgumentTypeError(f"unknown memory state mode: {memory_state_mode}")
    name_parts = [integration_mode]
    if memory_state_mode != "shared":
        name_parts.append(memory_state_mode)
    if attention_patch_layers != "all":
        name_parts.append(attention_patch_layers)
    return ArchitectureVariant(
        name="_".join(name_parts),
        integration_mode=integration_mode,
        memory_state_mode=memory_state_mode,
        attention_patch_layers=attention_patch_layers,
    )


def bucket_losses(losses: list[float], bucket_size: int) -> list[dict[str, float | int]]:
    if bucket_size <= 0:
        return []
    buckets = []
    for start in range(0, len(losses), bucket_size):
        values = losses[start : start + bucket_size]
        buckets.append(
            {
                "start_step": start + 1,
                "end_step": start + len(values),
                "mean_loss": sum(values) / len(values),
            }
        )
    return buckets


def train_args_for_variant(
    *,
    args: argparse.Namespace,
    variant: ArchitectureVariant,
    checkpoint_path: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        traces=args.train_traces,
        steps=args.steps,
        batch_size=args.batch_size,
        integration_mode=variant.integration_mode,
        memory_state_mode=variant.memory_state_mode,
        correction_rank=args.correction_rank,
        attention_patch_layers=variant.attention_patch_layers,
        freeze_base=args.freeze_base,
        write_loss_weight=args.write_loss_weight,
        structured_loss_weight=args.structured_loss_weight,
        authority_payload_loss_weight=args.authority_payload_loss_weight,
        out=str(checkpoint_path),
    )


def run_architecture_comparison(args: argparse.Namespace) -> dict[str, Any]:
    variants = args.variants or DEFAULT_VARIANTS
    run_dir = Path(args.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_results = []
    checkpoints: dict[str, str] = {}
    base_wrapper = train_module.QwenMemoryModel(
        train_module.QwenMemoryConfig(model_name=args.model, freeze_base=args.freeze_base)
    )
    base_model = base_wrapper.base_model
    tokenizer = base_wrapper.tokenizer
    del base_wrapper
    for variant in variants:
        if variant.name in checkpoints:
            raise ValueError(f"duplicate architecture variant name: {variant.name}")
        checkpoint_path = checkpoint_dir / f"{variant.name}.pt"
        result = train_checkpoint(
            train_args_for_variant(args=args, variant=variant, checkpoint_path=checkpoint_path),
            base_model=base_model,
            tokenizer=tokenizer,
        )
        train_results.append(
            {
                "variant": asdict(variant),
                "checkpoint_path": str(checkpoint_path),
                "final_loss": result["final_loss"],
                "mean_loss": result["mean_loss"],
                "loss_buckets": bucket_losses(result["losses"], args.loss_bucket_size),
            }
        )
        checkpoints[variant.name] = str(checkpoint_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eval_traces = read_llm_jsonl(args.eval_traces)
    eval_rows = comparison_rows(
        traces=eval_traces,
        model_name=args.model,
        checkpoints=checkpoints,
        modes=args.eval_modes,
        decode_strategy=args.decode_strategy,
        batch_size=args.eval_batch_size,
        base_model=base_model,
        tokenizer=tokenizer,
    )
    report = {
        "config": {
            "model": args.model,
            "train_traces": args.train_traces,
            "eval_traces": args.eval_traces,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "correction_rank": args.correction_rank,
            "write_loss_weight": args.write_loss_weight,
            "structured_loss_weight": args.structured_loss_weight,
            "authority_payload_loss_weight": args.authority_payload_loss_weight,
            "freeze_base": args.freeze_base,
            "decode_strategy": args.decode_strategy,
            "eval_modes": args.eval_modes,
        },
        "training": train_results,
        "evaluation": eval_rows,
    }
    report_path = run_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"report_path": str(report_path), **report}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-traces", required=True)
    parser.add_argument("--eval-traces", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--correction-rank", type=int, default=16)
    parser.add_argument("--write-loss-weight", type=float, default=0.2)
    parser.add_argument("--structured-loss-weight", type=float, default=0.2)
    parser.add_argument("--authority-payload-loss-weight", type=float, default=1.0)
    parser.add_argument("--freeze-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-dir", default="runs/qwen_memory/architecture_compare")
    parser.add_argument("--report-name", default="comparison.json")
    parser.add_argument("--loss-bucket-size", type=int, default=1000)
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        type=parse_variant,
        help=(
            "Run one architecture variant. Format: mode[:layers] or mode:by_type[:layers]. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--eval-modes",
        nargs="+",
        choices=["context_only", "teacher_forced", "predicted"],
        default=["teacher_forced"],
    )
    parser.add_argument(
        "--decode-strategy",
        choices=["score_answers", "generate"],
        default="score_answers",
    )
    return parser


def main() -> None:
    result = run_architecture_comparison(build_arg_parser().parse_args())
    print(f"wrote {result['report_path']}")
    for row in result["evaluation"]:
        print(
            f"{row['run_name']}\t{row['eval_mode']}\t"
            f"{row['integration_mode']}\t{row['aggregate_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
