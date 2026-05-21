from __future__ import annotations

import argparse
import gc
import inspect
import json
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path

import torch

from hotbob.llm.dataset import build_llm_scope_vocab, build_llm_tool_name_vocab, read_llm_jsonl
from hotbob.llm.evaluate import evaluate_traces
from hotbob.llm.qwen_memory_model import QwenMemoryConfig, QwenMemoryModel

QWEN_CONFIG_FIELDS = {field.name for field in fields(QwenMemoryConfig)}


def _model_accepts_shared_base() -> bool:
    return "base_model" in inspect.signature(QwenMemoryModel).parameters


def parse_checkpoints(values: list[str]) -> dict[str, str | None]:
    if not values:
        return {"context_only": None}
    checkpoints: dict[str, str | None] = {}
    for value in values:
        if "=" not in value:
            checkpoints[Path(value).stem] = value
            continue
        name, path = value.split("=", 1)
        checkpoints[name] = path
    return checkpoints


def load_model(
    *,
    model_name: str,
    traces,
    checkpoint_path: str | None,
    base_model: torch.nn.Module | None = None,
    tokenizer=None,
) -> tuple[QwenMemoryModel, dict]:
    state = None
    config_kwargs = {
        "model_name": model_name,
        "scope_vocab": build_llm_scope_vocab(traces),
        "tool_name_vocab": build_llm_tool_name_vocab(traces),
    }
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config_kwargs.update(
            {
                key: value
                for key, value in state.get("config", {}).items()
                if key in QWEN_CONFIG_FIELDS
            }
        )
        config_kwargs["model_name"] = model_name
    config = QwenMemoryConfig(**config_kwargs)
    if base_model is None and tokenizer is None:
        model = QwenMemoryModel(config)
    elif _model_accepts_shared_base():
        model = QwenMemoryModel(config, base_model=base_model, tokenizer=tokenizer)
    else:
        model = QwenMemoryModel(config)
    if state is not None:
        if "memory_heads_state" in state:
            model.memory_heads.load_state_dict(state["memory_heads_state"], strict=False)
        else:
            model.load_state_dict(state["model_state"], strict=False)
    return model, state or {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument(
        "--modes",
        "--mode",
        nargs="+",
        choices=["context_only", "teacher_forced", "predicted"],
        default=["teacher_forced"],
    )
    parser.add_argument(
        "--decode-strategy",
        choices=["score_answers", "generate"],
        default="score_answers",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out", default="runs/qwen_memory/comparison.json")
    return parser


def comparison_rows(
    *,
    traces,
    model_name: str,
    checkpoints: dict[str, str | None],
    modes: Iterable[str],
    decode_strategy: str,
    batch_size: int,
    base_model: torch.nn.Module | None = None,
    tokenizer=None,
) -> list[dict]:
    rows: list[dict] = []
    for run_name, checkpoint_path in checkpoints.items():
        model, state = load_model(
            model_name=model_name,
            traces=traces,
            checkpoint_path=checkpoint_path,
            base_model=base_model,
            tokenizer=tokenizer,
        )
        for mode in modes:
            result = evaluate_traces(
                model,
                traces,
                mode=mode,
                decode_strategy=decode_strategy,
                batch_size=batch_size,
            )
            row = {
                "run_name": run_name,
                "integration_mode": model.config_obj.integration_mode,
                "checkpoint_path": checkpoint_path,
                "eval_mode": mode,
                "decode_strategy": decode_strategy,
                "aggregate_accuracy": result["exact_target_text_accuracy"],
                "secret_leak_failures": result["secret_leak_failures"],
                "wrong_scope_failures": result["wrong_scope_failures"],
                "expiry_failures": result["expiry_failures"],
                "authority_conflict_failures": result["authority_conflict_failures"],
                "tool_override_failures": result["tool_override_failures"],
                "interruption_failures": result["interruption_failures"],
                "stale_state_failures": result["stale_state_failures"],
                "privacy_disclosure_failures": result["privacy_disclosure_failures"],
                "tool_routing_failures": result["tool_routing_failures"],
                "structured_policy_failures": result["structured_policy_failures"],
            }
            row["family_accuracies"] = {
                key.removesuffix("_accuracy"): value
                for key, value in result.items()
                if key.endswith("_accuracy")
                and key not in {"exact_target_text_accuracy", "normalized_action_accuracy"}
            }
            if "git_commit" in state:
                row["git_commit"] = state["git_commit"]
            rows.append(row)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = build_arg_parser().parse_args()
    traces = read_llm_jsonl(args.traces)
    rows = comparison_rows(
        traces=traces,
        model_name=args.model,
        checkpoints=parse_checkpoints(args.checkpoints),
        modes=args.modes,
        decode_strategy=args.decode_strategy,
        batch_size=args.batch_size,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": rows}, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['run_name']}\t{row['eval_mode']}\t"
            f"{row['integration_mode']}\t{row['aggregate_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
