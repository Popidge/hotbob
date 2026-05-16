from __future__ import annotations

import argparse
import re
from collections import Counter

import torch

from hotbob.llm.dataset import LLMTrace, build_llm_scope_vocab, read_llm_jsonl
from hotbob.llm.prompts import ANSWER_SET
from hotbob.llm.qwen_memory_model import QwenMemoryConfig, QwenMemoryModel


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


def evaluate_traces(
    model: QwenMemoryModel,
    traces: list[LLMTrace],
    *,
    mode: str,
) -> dict[str, float | int]:
    correct = 0
    leaks = 0
    families: Counter[str] = Counter()
    family_correct: Counter[str] = Counter()
    with torch.no_grad():
        for trace in traces:
            if mode == "teacher_forced":
                model.apply_teacher_trace_memory(trace)
                use_memory = True
            elif mode == "predicted":
                model.reset_memory(1)
                for event in trace.events:
                    model.forward_event(
                        event.role,
                        event.content,
                        scope=event.scope,
                        apply_predicted=True,
                    )
                use_memory = True
            else:
                model.reset_memory(1)
                use_memory = False
            pred = model.generate_final(
                trace.final_prompt,
                current_scope=trace.current_scope,
                use_memory=use_memory,
            )
            ok = exact_match(pred, trace.target_text)
            correct += int(ok)
            leaks += int(secret_leak(trace, pred))
            family = str(trace.metadata.get("task_family", "unknown"))
            families[family] += 1
            family_correct[family] += int(ok)
    total = len(traces)
    result: dict[str, float | int] = {
        "total": total,
        "exact_target_text_accuracy": correct / total if total else 0.0,
        "normalized_action_accuracy": correct / total if total else 0.0,
        "secret_leak_failures": leaks,
    }
    for family, count in families.items():
        result[f"{family}_accuracy"] = family_correct[family] / count
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--mode",
        choices=["context_only", "teacher_forced", "predicted"],
        default="teacher_forced",
    )
    args = parser.parse_args()
    traces = read_llm_jsonl(args.traces)
    state = None
    config_kwargs = {
        "model_name": args.model,
        "scope_vocab": build_llm_scope_vocab(traces),
    }
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config_kwargs.update(state.get("config", {}))
        config_kwargs["model_name"] = args.model
    config = QwenMemoryConfig(**config_kwargs)
    model = QwenMemoryModel(config)
    if state:
        if "memory_heads_state" in state:
            model.memory_heads.load_state_dict(state["memory_heads_state"])
        else:
            model.load_state_dict(state["model_state"], strict=False)
    print(evaluate_traces(model, traces, mode=args.mode))


if __name__ == "__main__":
    main()
