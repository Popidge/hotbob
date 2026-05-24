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
from hotbob.llm.memory_adapter import scope_id
from hotbob.llm.prompts import ANSWER_SET
from hotbob.llm.qwen_memory_model import QwenMemoryConfig, QwenMemoryModel
from hotbob.training.dataset import (
    AUTHORITY_TO_ID,
    OP_TO_ID,
    PRIVACY_TO_ID,
    TYPE_TO_ID,
    structured_targets_from_payload,
)
from hotbob.types import MemoryOp, MemoryPrivacy

QWEN_CONFIG_FIELDS = {field.name for field in fields(QwenMemoryConfig)}

STRUCTURED_WRITE_HEAD_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("payload_kind", "payload_kind_logits", "payload_kind_id", "has_payload"),
    (
        "payload_default_action",
        "payload_default_action_logits",
        "default_action_id",
        "has_default_action",
    ),
    (
        "payload_winning_authority_level",
        "payload_winning_authority_level_logits",
        "winning_authority_level_id",
        "has_winning_authority_level",
    ),
    (
        "payload_losing_authority_level",
        "payload_losing_authority_level_logits",
        "losing_authority_level_id",
        "has_losing_authority_level",
    ),
)

BASIC_WRITE_HEAD_SPECS: tuple[tuple[str, str], ...] = (
    ("op", "op_logits"),
    ("type", "type_logits"),
    ("scope", "scope_logits"),
    ("privacy", "privacy_logits"),
    ("authority", "authority_logits"),
    ("slot", "slot_logits"),
)


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


def _accuracy_block(total: int, correct: int) -> dict[str, int | float]:
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }


def _prediction_distribution(counter: Counter[str], total: int) -> dict[str, float]:
    if total == 0:
        return {}
    return {label: count / total for label, count in sorted(counter.items())}


def _target_op_for_event(
    trace: LLMTrace,
    *,
    event_scope: str | None,
    op_index: int,
) -> tuple[MemoryOp | None, int]:
    next_op = (
        trace.expected_memory_ops[op_index] if op_index < len(trace.expected_memory_ops) else None
    )
    scope = event_scope or trace.current_scope
    if next_op is not None and next_op.scope == scope:
        return next_op, op_index + 1
    return None, op_index


def _write_head_matches(
    model: QwenMemoryModel,
    outputs: dict[str, torch.Tensor],
    target_op: MemoryOp,
    *,
    op_index: int,
) -> dict[str, bool]:
    matches: dict[str, bool] = {}
    matches["op"] = int(outputs["op_logits"].argmax(dim=-1).item()) == OP_TO_ID[target_op.op]
    matches["type"] = int(outputs["type_logits"].argmax(dim=-1).item()) == TYPE_TO_ID[
        target_op.type
    ]
    scope_vocab = getattr(model.config_obj, "scope_vocab", {"default": 1})
    matches["scope"] = int(outputs["scope_logits"].argmax(dim=-1).item()) == scope_id(
        target_op.scope,
        scope_vocab,
    )
    matches["privacy"] = int(outputs["privacy_logits"].argmax(dim=-1).item()) == PRIVACY_TO_ID[
        target_op.privacy
    ]
    matches["authority"] = int(outputs["authority_logits"].argmax(dim=-1).item()) == (
        AUTHORITY_TO_ID[target_op.authority]
    )
    num_slots = getattr(model.config_obj, "num_memory_slots", 32)
    expected_slot = min(op_index - 1, num_slots - 1)
    matches["slot"] = int(outputs["slot_logits"].argmax(dim=-1).item()) == expected_slot
    tool_vocab = getattr(model.config_obj, "tool_name_vocab", {})
    structured = structured_targets_from_payload(target_op.payload, tool_vocab)
    for name, logit_key, target_key, mask_key in STRUCTURED_WRITE_HEAD_SPECS:
        target_id = int(structured[target_key])
        has_target = (
            bool(structured[mask_key])
            if isinstance(structured.get(mask_key), bool)
            else target_id > 0
        )
        if not has_target:
            continue
        logits = outputs[logit_key]
        target_id = min(target_id, logits.shape[-1] - 1)
        matches[name] = int(logits.argmax(dim=-1).item()) == target_id
    return matches


def collect_authority_write_diagnostics(
    model: QwenMemoryModel,
    trace: LLMTrace,
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    write_events = 0
    op_index = 0
    with torch.inference_mode():
        for event in trace.events:
            outputs = model.forward_event(
                event.role,
                event.content,
                scope=event.scope,
                apply_predicted=False,
            )
            if "op_logits" not in outputs:
                continue
            target_op, op_index = _target_op_for_event(
                trace,
                event_scope=event.scope,
                op_index=op_index,
            )
            if target_op is None:
                continue
            write_events += 1
            for head_name, matched in _write_head_matches(
                model,
                outputs,
                target_op,
                op_index=op_index,
            ).items():
                totals[head_name] += 1
                correct[head_name] += int(matched)
    return {
        "write_events": write_events,
        "by_head": {
            head: _accuracy_block(totals[head], correct[head])
            for head in sorted(totals)
        },
    }


def _teacher_authority_slot(trace: LLMTrace) -> int | None:
    for idx, op in enumerate(trace.expected_memory_ops):
        if op.scope == trace.current_scope:
            return idx
    return None


def collect_authority_read_diagnostics(
    model: QwenMemoryModel,
    trace: LLMTrace,
) -> dict[str, Any] | None:
    if model.config_obj.integration_mode != "prefix":
        return None
    _, attention = model.memory_prefix_for_prompt(trace.final_prompt, trace.current_scope)
    attn = attention[0].detach().cpu()
    current_scope_tensor = model._scope_tensor(
        trace.current_scope,
        model.memory.device,
        batch_size=1,
    )
    active = model.memory.active_mask(current_scope_tensor)[0].cpu()
    authority_slot = _teacher_authority_slot(trace)
    distractor_slot = next(
        (
            idx
            for idx, op in enumerate(trace.expected_memory_ops)
            if op.scope != trace.current_scope
        ),
        None,
    )
    in_scope_slots = [slot for slot in range(model.memory.num_slots) if bool(active[slot].item())]
    authority_attention = (
        float(attn[authority_slot].item())
        if authority_slot is not None and bool(active[authority_slot].item())
        else 0.0
    )
    distractor_active = (
        distractor_slot is not None and bool(active[distractor_slot].item())
    )
    distractor_attention = 0.0
    if distractor_active and distractor_slot is not None:
        distractor_attention = float(attn[distractor_slot].item())
    max_slot = int(attn.argmax().item()) if in_scope_slots else None
    return {
        "active_in_scope_slots": len(in_scope_slots),
        "authority_slot": authority_slot,
        "authority_slot_attention": authority_attention,
        "distractor_slot": distractor_slot,
        "distractor_active_in_scope": distractor_active,
        "distractor_slot_attention": distractor_attention,
        "max_attention_slot": max_slot,
        "authority_slot_is_max": (
            max_slot == authority_slot
            if authority_slot is not None and in_scope_slots
            else None
        ),
    }


def compact_authority_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "failures"}


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
    include_write_diagnostics: bool = True,
    include_read_diagnostics: bool = True,
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
    by_target_action_total: Counter[str] = Counter()
    by_target_action_correct: Counter[str] = Counter()
    by_winning_authority_total: Counter[str] = Counter()
    by_winning_authority_correct: Counter[str] = Counter()
    by_losing_authority_total: Counter[str] = Counter()
    by_losing_authority_correct: Counter[str] = Counter()
    prediction_counter: Counter[str] = Counter()
    injection_total = 0
    injection_correct = 0
    non_injection_total = 0
    non_injection_correct = 0
    target_ranks: list[int] = []
    target_margins: list[float] = []
    target_rank_by_scenario: dict[str, list[int]] = defaultdict(list)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    write_totals: Counter[str] = Counter()
    write_correct: Counter[str] = Counter()
    read_authority_attention: list[float] = []
    read_authority_is_max = 0
    read_authority_is_max_total = 0
    read_active_slots: list[int] = []
    read_distractor_active = 0
    with torch.inference_mode():
        for trace_index, trace in tqdm(authority, desc=f"authority:{mode}"):
            if include_write_diagnostics:
                write_diag = collect_authority_write_diagnostics(model, trace)
                for head, block in write_diag["by_head"].items():
                    write_totals[head] += int(block["total"])
                    write_correct[head] += int(block["correct"])
            use_memory = _prepare_memory_for_trace(model, trace, mode)
            read_diag = (
                collect_authority_read_diagnostics(model, trace)
                if include_read_diagnostics
                else None
            )
            if read_diag is not None:
                read_authority_attention.append(read_diag["authority_slot_attention"])
                read_active_slots.append(read_diag["active_in_scope_slots"])
                if read_diag["distractor_active_in_scope"]:
                    read_distractor_active += 1
                if read_diag["authority_slot_is_max"] is not None:
                    read_authority_is_max_total += 1
                    read_authority_is_max += int(read_diag["authority_slot_is_max"])
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
            prediction_counter[prediction] += 1
            prompt_injection = bool(trace.metadata.get("authority_prompt_injection", False))
            if prompt_injection:
                injection_total += 1
                injection_correct += int(ok)
            else:
                non_injection_total += 1
                non_injection_correct += int(ok)
            scenario = str(trace.metadata.get("scenario", "unknown"))
            winning_authority = str(trace.metadata.get("winning_authority", "unknown"))
            losing_authority = str(trace.metadata.get("losing_authority", "unknown"))
            conflict_action = str(trace.metadata.get("conflict_action", "unknown"))
            by_scenario_total[scenario] += 1
            by_scenario_correct[scenario] += int(ok)
            by_target_action_total[trace.target_action] += 1
            by_target_action_correct[trace.target_action] += int(ok)
            by_winning_authority_total[winning_authority] += 1
            by_winning_authority_correct[winning_authority] += int(ok)
            by_losing_authority_total[losing_authority] += 1
            by_losing_authority_correct[losing_authority] += int(ok)
            confusion[trace.target_text][prediction] += 1
            target_rank = None
            target_score = None
            prediction_score = None
            target_margin = None
            if candidate_scores:
                scores_by_candidate = dict(candidate_scores)
                if trace.target_text in scores_by_candidate:
                    target_rank = (
                        [candidate for candidate, _ in candidate_scores].index(trace.target_text)
                        + 1
                    )
                    target_score = scores_by_candidate[trace.target_text]
                    non_target_scores = [
                        score
                        for candidate, score in candidate_scores
                        if candidate != trace.target_text
                    ]
                    best_non_target_score = (
                        max(non_target_scores) if non_target_scores else target_score
                    )
                    target_margin = target_score - best_non_target_score
                    target_ranks.append(target_rank)
                    target_margins.append(target_margin)
                    target_rank_by_scenario[scenario].append(target_rank)
                prediction_score = scores_by_candidate.get(prediction)
            if ok and not include_correct:
                continue
            margin = None
            if candidate_scores and len(candidate_scores) > 1:
                margin = candidate_scores[0][1] - candidate_scores[1][1]
            row: dict[str, Any] = {
                "trace_index": trace_index,
                "scenario": scenario,
                "current_scope": trace.current_scope,
                "target_text": trace.target_text,
                "prediction": prediction,
                "target_action": trace.target_action,
                "candidate_scores": candidate_scores,
                "score_margin": margin,
                "target_rank": target_rank,
                "target_score": target_score,
                "prediction_score": prediction_score,
                "target_margin": target_margin,
                "winning_authority": winning_authority,
                "losing_authority": losing_authority,
                "conflict_action": conflict_action,
                "authority_prompt_injection": prompt_injection,
                "events": _events(trace),
                "memory_ops": _memory_ops(trace),
                "correct": ok,
            }
            if read_diag is not None:
                row["read_diagnostics"] = read_diag
            failures.append(row)
    total = len(authority)
    report: dict[str, Any] = {
        "checkpoint": checkpoint,
        "mode": mode,
        "decode_strategy": decode_strategy,
        "total_authority_traces": total,
        "accuracy": correct / total if total else 0.0,
        "failure_count": total - correct,
        "prediction_distribution": _prediction_distribution(prediction_counter, total),
        "by_prompt_injection": {
            "true": _accuracy_block(injection_total, injection_correct),
            "false": _accuracy_block(non_injection_total, non_injection_correct),
        },
        "by_scenario": {
            scenario: {
                "total": count,
                "accuracy": by_scenario_correct[scenario] / count if count else 0.0,
            }
            for scenario, count in sorted(by_scenario_total.items())
        },
        "target_rank_mean": sum(target_ranks) / len(target_ranks) if target_ranks else 0.0,
        "target_rank_by_scenario": {
            scenario: sum(ranks) / len(ranks) if ranks else 0.0
            for scenario, ranks in sorted(target_rank_by_scenario.items())
        },
        "mean_target_margin": (
            sum(target_margins) / len(target_margins) if target_margins else 0.0
        ),
        "by_target_action": {
            action: {
                "total": count,
                "accuracy": by_target_action_correct[action] / count if count else 0.0,
            }
            for action, count in sorted(by_target_action_total.items())
        },
        "by_winning_authority": {
            authority: {
                "total": count,
                "accuracy": by_winning_authority_correct[authority] / count if count else 0.0,
            }
            for authority, count in sorted(by_winning_authority_total.items())
        },
        "by_losing_authority": {
            authority: {
                "total": count,
                "accuracy": by_losing_authority_correct[authority] / count if count else 0.0,
            }
            for authority, count in sorted(by_losing_authority_total.items())
        },
        "confusion": {
            target: dict(predictions) for target, predictions in sorted(confusion.items())
        },
        "failures": failures,
    }
    if include_write_diagnostics:
        report["write_diagnostics"] = {
            "by_head": {
                head: _accuracy_block(write_totals[head], write_correct[head])
                for head in sorted(write_totals)
            }
        }
    if include_read_diagnostics and read_authority_attention:
        report["read_diagnostics"] = {
            "mean_authority_slot_attention": sum(read_authority_attention)
            / len(read_authority_attention),
            "mean_active_in_scope_slots": sum(read_active_slots) / len(read_active_slots),
            "distractor_active_in_scope_rate": read_distractor_active / total if total else 0.0,
            "authority_slot_is_max_rate": (
                read_authority_is_max / read_authority_is_max_total
                if read_authority_is_max_total
                else 0.0
            ),
        }
    return report


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
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit per-trace failure rows from the saved JSON.",
    )
    parser.add_argument("--no-write-diagnostics", action="store_true")
    parser.add_argument("--no-read-diagnostics", action="store_true")
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
        include_write_diagnostics=not args.no_write_diagnostics,
        include_read_diagnostics=not args.no_read_diagnostics,
    )
    if args.compact:
        report = compact_authority_report(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary_keys = (
        "total_authority_traces",
        "accuracy",
        "write_diagnostics",
        "read_diagnostics",
        "by_prompt_injection",
    )
    print(json.dumps({key: report[key] for key in summary_keys if key in report}))


if __name__ == "__main__":
    main()
