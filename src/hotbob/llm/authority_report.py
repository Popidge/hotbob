from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel
from tqdm import tqdm

from hotbob.llm.dataset import (
    LLMTrace,
    build_llm_scope_vocab,
    build_llm_tool_name_vocab,
    read_llm_jsonl,
)
from hotbob.llm.evaluate import _prepare_memory_for_trace, exact_match
from hotbob.llm.prompts import ANSWER_SET
from hotbob.llm.qwen_memory_model import QwenMemoryConfig, QwenMemoryModel
from hotbob.types import MemoryPrivacy

QWEN_CONFIG_FIELDS = {field.name for field in fields(QwenMemoryConfig)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _memory_ops(trace: LLMTrace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for op in trace.expected_memory_ops:
        row = {
            "op": str(op.op),
            "type": str(op.type),
            "key": op.key,
            "scope": op.scope,
            "privacy": str(op.privacy),
            "authority": str(op.authority),
            "payload": _jsonable(op.payload),
        }
        if op.privacy == MemoryPrivacy.VISIBLE:
            row["value"] = op.value
        else:
            row["value"] = "<redacted>"
        rows.append(row)
    return rows


def _events(trace: LLMTrace) -> list[dict[str, str | None]]:
    return [
        {"role": event.role, "scope": event.scope, "content": event.content}
        for event in trace.events
    ]


def load_model_for_report(
    *,
    checkpoint: str | None,
    model_name: str,
    traces: list[LLMTrace],
) -> QwenMemoryModel:
    state = None
    config_kwargs = {
        "model_name": model_name,
        "scope_vocab": build_llm_scope_vocab(traces),
        "tool_name_vocab": build_llm_tool_name_vocab(traces),
    }
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config_kwargs.update(
            {
                key: value
                for key, value in state.get("config", {}).items()
                if key in QWEN_CONFIG_FIELDS
            }
        )
        config_kwargs["model_name"] = model_name
    model = QwenMemoryModel(QwenMemoryConfig(**config_kwargs))
    if state is not None:
        if "memory_heads_state" in state:
            model.memory_heads.load_state_dict(state["memory_heads_state"], strict=False)
        else:
            model.load_state_dict(state["model_state"], strict=False)
        if state.get("lora_state"):
            model.load_lora_state_dict(state["lora_state"])
    return model


def build_authority_report(
    model: QwenMemoryModel,
    traces: list[LLMTrace],
    *,
    checkpoint: str | None,
    mode: str,
    decode_strategy: str,
    include_correct: bool = False,
) -> dict[str, Any]:
    authority = [
        (idx, trace)
        for idx, trace in enumerate(traces)
        if trace.metadata.get("task_family") == "authority_conflict"
    ]
    correct = 0
    failures: list[dict[str, Any]] = []
    by_scenario_total: Counter[str] = Counter()
    by_scenario_correct: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    with torch.inference_mode():
        for trace_index, trace in tqdm(authority, desc=f"authority:{mode}"):
            use_memory = _prepare_memory_for_trace(model, trace, mode)
            candidate_scores: list[tuple[str, float]] | None = None
            if decode_strategy == "score_answers":
                scores = model.score_final_candidates_dict(
                    trace.final_prompt,
                    list(ANSWER_SET),
                    current_scope=trace.current_scope,
                    use_memory=use_memory,
                )
                candidate_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                prediction = candidate_scores[0][0]
            else:
                prediction = model.generate_final(
                    trace.final_prompt,
                    current_scope=trace.current_scope,
                    use_memory=use_memory,
                )
            ok = exact_match(prediction, trace.target_text)
            correct += int(ok)
            scenario = str(trace.metadata.get("scenario", "unknown"))
            by_scenario_total[scenario] += 1
            by_scenario_correct[scenario] += int(ok)
            confusion[trace.target_text][prediction] += 1
            if ok and not include_correct:
                continue
            margin = None
            if candidate_scores and len(candidate_scores) > 1:
                margin = candidate_scores[0][1] - candidate_scores[1][1]
            failures.append(
                {
                    "trace_index": trace_index,
                    "scenario": scenario,
                    "current_scope": trace.current_scope,
                    "target_text": trace.target_text,
                    "prediction": prediction,
                    "target_action": trace.target_action,
                    "candidate_scores": candidate_scores,
                    "score_margin": margin,
                    "events": _events(trace),
                    "memory_ops": _memory_ops(trace),
                    "correct": ok,
                }
            )
    total = len(authority)
    return {
        "checkpoint": checkpoint,
        "mode": mode,
        "decode_strategy": decode_strategy,
        "total_authority_traces": total,
        "accuracy": correct / total if total else 0.0,
        "failure_count": total - correct,
        "by_scenario": {
            scenario: {
                "total": count,
                "accuracy": by_scenario_correct[scenario] / count if count else 0.0,
            }
            for scenario, count in sorted(by_scenario_total.items())
        },
        "confusion": {
            target: dict(predictions) for target, predictions in sorted(confusion.items())
        },
        "failures": failures,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--mode",
        choices=["context_only", "teacher_forced", "predicted"],
        default="predicted",
    )
    parser.add_argument(
        "--decode-strategy",
        choices=["score_answers", "generate"],
        default="score_answers",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="runs/qwen_memory/authority_report.json")
    parser.add_argument("--include-correct", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    traces = read_llm_jsonl(args.traces)
    if args.limit is not None:
        traces = traces[: args.limit]
    model = load_model_for_report(
        checkpoint=args.checkpoint,
        model_name=args.model,
        traces=traces,
    )
    report = build_authority_report(
        model,
        traces,
        checkpoint=args.checkpoint,
        mode=args.mode,
        decode_strategy=args.decode_strategy,
        include_correct=args.include_correct,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("total_authority_traces", "accuracy")}))


if __name__ == "__main__":
    main()
