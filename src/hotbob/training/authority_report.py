from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from hotbob.data.traces import read_jsonl
from hotbob.training.evaluate import (
    EvalResult,
    boundary_scores,
    diagnostic_accuracy,
    diagnostic_read_mass,
    diagnostic_total,
    evaluate_sequential_neural,
    memory_required_accuracy,
    read_target_mass,
    top_confusions,
)
from hotbob.types import TaskTrace

AUTHORITY_FAMILY = "authority_conflict"
AUTHORITY_PROBE_FAMILIES = (AUTHORITY_FAMILY, "tool_verified_override")


def _ratio(correct: int, total: int) -> float | None:
    if total == 0:
        return None
    return correct / total


def _mode_summary(result: EvalResult | None) -> dict[str, Any]:
    if result is None:
        return {"available": False}

    precision, recall, f1 = boundary_scores(result)
    scenario = AUTHORITY_FAMILY
    confusions = (
        result.family_action_confusion.get(scenario, Counter())
        if result.family_action_confusion is not None
        else Counter()
    )
    authority_total = result.totals[AUTHORITY_FAMILY]
    authority_correct = result.correct[AUTHORITY_FAMILY]
    probe_total = sum(result.totals[family] for family in AUTHORITY_PROBE_FAMILIES)
    probe_correct = sum(result.correct[family] for family in AUTHORITY_PROBE_FAMILIES)
    return {
        "available": True,
        "authority_accuracy": _ratio(authority_correct, authority_total),
        "authority_total": authority_total,
        "authority_correct": authority_correct,
        "authority_memory_required_accuracy": diagnostic_accuracy(result, AUTHORITY_FAMILY),
        "authority_target_slot_read_mass": diagnostic_read_mass(result, AUTHORITY_FAMILY),
        "authority_diagnostic_total": diagnostic_total(result, AUTHORITY_FAMILY),
        "authority_top_confusions": top_confusions(confusions, limit=8),
        "authority_probe_accuracy": _ratio(probe_correct, probe_total),
        "authority_probe_total": probe_total,
        "memory_required_accuracy": memory_required_accuracy(result),
        "target_slot_read_mass": read_target_mass(result),
        "boundary": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": result.boundary_tp,
            "fp": result.boundary_fp,
            "fn": result.boundary_fn,
            "tn": result.boundary_tn,
        },
        "write_accuracies": result.write_accuracies,
        "failures": {
            "secret_leaks": result.failures[0],
            "wrong_scope": result.failures[1],
            "expiry": result.failures[2],
        },
    }


def _metadata_slices(traces: list[TaskTrace]) -> dict[str, Any]:
    authority_traces = [trace for trace in traces if trace.task_family == AUTHORITY_FAMILY]
    by_scenario: Counter[str] = Counter()
    by_winning_authority: Counter[str] = Counter()
    by_losing_authority: Counter[str] = Counter()
    by_prompt_injection: Counter[str] = Counter()
    for trace in authority_traces:
        by_scenario[str(trace.metadata.get("scenario", "unknown"))] += 1
        by_winning_authority[str(trace.metadata.get("winning_authority", "unknown"))] += 1
        by_losing_authority[str(trace.metadata.get("losing_authority", "unknown"))] += 1
        by_prompt_injection[str(bool(trace.metadata.get("authority_prompt_injection", False)))] += 1
    return {
        "authority_total": len(authority_traces),
        "by_scenario": dict(sorted(by_scenario.items())),
        "by_winning_authority": dict(sorted(by_winning_authority.items())),
        "by_losing_authority": dict(sorted(by_losing_authority.items())),
        "by_prompt_injection": dict(sorted(by_prompt_injection.items())),
    }


def build_controller_authority_report(
    *,
    checkpoint: str,
    traces: list[TaskTrace],
    device: str,
) -> dict[str, Any]:
    modes = {
        "context_only": "context_only",
        "teacher_forced": "teacher_forced",
        "predicted": "predicted",
        "predicted_oracle_slot": "predicted_oracle_slot",
        "predicted_oracle_scope": "predicted_oracle_scope",
        "predicted_oracle_value": "predicted_oracle_value",
        "predicted_oracle_all": "predicted_oracle_slot_oracle_scope_oracle_value",
    }
    results = {
        name: evaluate_sequential_neural(checkpoint, traces, device, memory_mode=mode)
        for name, mode in modes.items()
    }
    return {
        "checkpoint": checkpoint,
        "num_traces": len(traces),
        "device": device,
        "metadata": _metadata_slices(traces),
        "modes": {name: _mode_summary(result) for name, result in results.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/latest.pt")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out", default="runs/controller_authority_report.json")
    args = parser.parse_args()

    traces = read_jsonl(args.traces)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = build_controller_authority_report(
        checkpoint=args.checkpoint,
        traces=traces,
        device=device,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "wrote controller authority report "
        f"{out_path} predicted_authority="
        f"{report['modes']['predicted'].get('authority_accuracy')}"
    )


if __name__ == "__main__":
    main()
