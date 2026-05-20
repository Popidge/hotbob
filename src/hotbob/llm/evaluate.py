from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import fields
from itertools import islice

import torch
from tqdm import tqdm

from hotbob.llm.dataset import LLMTrace, build_llm_scope_vocab, read_llm_jsonl
from hotbob.llm.prompts import ANSWER_SET
from hotbob.llm.qwen_memory_model import QwenMemoryConfig, QwenMemoryModel

QWEN_CONFIG_FIELDS = {field.name for field in fields(QwenMemoryConfig)}


def normalize_generated_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    return text.rstrip(".! ") + "."


def exact_match(prediction: str, target: str) -> bool:
    return normalize_generated_text(extract_allowed_answer(prediction)) == normalize_generated_text(
        target
    )


def extract_allowed_answer(text: str) -> str:
    normalized = normalize_generated_text(text)
    allowed = sorted(ANSWER_SET, key=len, reverse=True)
    for answer in allowed:
        answer_norm = normalize_generated_text(answer)
        if normalized.startswith(answer_norm):
            return answer
    return text


def secret_leak(trace: LLMTrace, prediction: str) -> bool:
    pred = prediction.lower()
    return any(
        "hidden_from_user" in str(op.privacy).lower() and op.value.lower() in pred
        for op in trace.expected_memory_ops
    )


def batched(items: list[LLMTrace], batch_size: int) -> Iterable[list[LLMTrace]]:
    batch_size = max(batch_size, 1)
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def _prepare_memory_for_trace(model: QwenMemoryModel, trace: LLMTrace, mode: str) -> bool:
    if mode == "teacher_forced":
        model.apply_teacher_trace_memory(trace)
        return True
    if mode == "predicted":
        model.reset_memory(1)
        for event in trace.events:
            model.forward_event(
                event.role,
                event.content,
                scope=event.scope,
                apply_predicted=True,
            )
        return True
    model.reset_memory(1)
    return False


def evaluate_traces(
    model: QwenMemoryModel,
    traces: list[LLMTrace],
    *,
    mode: str,
    max_new_tokens: int = 16,
    decode_strategy: str = "score_answers",
    batch_size: int = 1,
) -> dict[str, float | int | str]:
    correct = 0
    leaks = 0
    wrong_scope_failures = 0
    expiry_failures = 0
    families: Counter[str] = Counter()
    family_correct: Counter[str] = Counter()
    with torch.inference_mode():
        total_batches = math.ceil(len(traces) / max(batch_size, 1))
        for group in tqdm(batched(traces, batch_size), total=total_batches, desc=f"eval:{mode}"):
            for trace in group:
                use_memory = _prepare_memory_for_trace(model, trace, mode)
                if decode_strategy == "score_answers":
                    pred = model.choose_final_candidate(
                        trace.final_prompt,
                        list(ANSWER_SET),
                        current_scope=trace.current_scope,
                        use_memory=use_memory,
                    )
                else:
                    pred = model.generate_final(
                        trace.final_prompt,
                        current_scope=trace.current_scope,
                        use_memory=use_memory,
                        max_new_tokens=max_new_tokens,
                    )
                ok = exact_match(pred, trace.target_text)
                correct += int(ok)
                leaks += int(secret_leak(trace, pred))
                family = str(trace.metadata.get("task_family", "unknown"))
                families[family] += 1
                family_correct[family] += int(ok)
                if family == "scope_isolation" and not ok:
                    wrong_scope_failures += 1
                if family == "expiry" and not ok:
                    expiry_failures += 1
    total = len(traces)
    result: dict[str, float | int | str] = {
        "mode": mode,
        "integration_mode": model.config_obj.integration_mode,
        "decode_strategy": decode_strategy,
        "total": total,
        "exact_target_text_accuracy": correct / total if total else 0.0,
        "normalized_action_accuracy": correct / total if total else 0.0,
        "secret_leak_failures": leaks,
        "wrong_scope_failures": wrong_scope_failures,
        "expiry_failures": expiry_failures,
    }
    for family, count in families.items():
        result[f"{family}_accuracy"] = family_correct[family] / count
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--integration-mode",
        choices=["prefix", "attention_q", "attention_o", "attention_qo"],
        default=None,
    )
    parser.add_argument(
        "--mode",
        choices=["all", "context_only", "teacher_forced", "predicted"],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--decode-strategy",
        choices=["score_answers", "generate"],
        default="score_answers",
    )
    return parser


def _limited(traces: list[LLMTrace], limit: int | None) -> list[LLMTrace]:
    return traces if limit is None else traces[:limit]


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    traces = _limited(read_llm_jsonl(args.traces), args.limit)
    state = None
    config_kwargs = {
        "model_name": args.model,
        "scope_vocab": build_llm_scope_vocab(traces),
    }
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config_kwargs.update(
            {
                key: value
                for key, value in state.get("config", {}).items()
                if key in QWEN_CONFIG_FIELDS
            }
        )
        config_kwargs["model_name"] = args.model
    if args.integration_mode is not None:
        config_kwargs["integration_mode"] = args.integration_mode
    config = QwenMemoryConfig(**config_kwargs)
    model = QwenMemoryModel(config)
    if state:
        if "memory_heads_state" in state:
            model.memory_heads.load_state_dict(state["memory_heads_state"], strict=False)
        else:
            model.load_state_dict(state["model_state"], strict=False)
    modes: Iterable[str] = (
        ["context_only", "teacher_forced", "predicted"] if args.mode == "all" else [args.mode]
    )
    results = {
        mode: evaluate_traces(
            model,
            traces,
            mode=mode,
            max_new_tokens=args.max_new_tokens,
            decode_strategy=args.decode_strategy,
            batch_size=args.batch_size,
        )
        for mode in modes
    }
    print(
        {
            "integration_mode": model.config_obj.integration_mode,
            "results": results,
        }
    )


if __name__ == "__main__":
    main()
