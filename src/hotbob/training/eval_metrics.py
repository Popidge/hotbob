from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from hotbob.data.hygiene import is_memory_required_trace
from hotbob.types import ActionLabel, TaskTrace


@dataclass(frozen=True)
class EvalResult:
    totals: Counter[str]
    correct: Counter[str]
    memory_required_total: int
    memory_required_correct: int
    failures: tuple[int, int, int]
    write_accuracies: dict[str, float]
    boundary_tp: int = 0
    boundary_fp: int = 0
    boundary_fn: int = 0
    boundary_tn: int = 0
    read_target_mass_sum: float = 0.0
    read_target_mass_count: int = 0
    family_action_confusion: dict[str, Counter[tuple[str, str]]] | None = None
    family_read_mass_sum: Counter[str] | None = None
    family_read_mass_count: Counter[str] | None = None
    family_memory_required_total: Counter[str] | None = None
    family_memory_required_correct: Counter[str] | None = None


def diagnostic_scenario(trace: TaskTrace) -> str:
    if trace.task_family == "hidden_colour":
        if trace.expected_final_action == ActionLabel.REFUSE_TO_REVEAL_SECRET:
            return "hidden_colour_reveal"
        return "hidden_colour_guess"
    if trace.task_family == "expiry":
        return "expiry_active" if trace.metadata.get("active_preference") else "expiry_expired"
    if trace.task_family == "standing_order":
        return "standing_order_action"
    return trace.task_family


def update_family_diagnostics(
    *,
    trace: TaskTrace,
    pred: ActionLabel,
    target: ActionLabel,
    target_read_mass: float,
    action_confusion: dict[str, Counter[tuple[str, str]]],
    read_mass_sum: Counter[str],
    read_mass_count: Counter[str],
    memory_required_total: Counter[str],
    memory_required_correct: Counter[str],
) -> None:
    scenario = diagnostic_scenario(trace)
    action_confusion.setdefault(scenario, Counter())
    action_confusion[scenario][(str(target), str(pred))] += 1
    read_mass_sum[scenario] += target_read_mass
    read_mass_count[scenario] += 1
    if is_memory_required_trace(trace):
        memory_required_total[scenario] += 1
        memory_required_correct[scenario] += int(pred == target)


def top_confusions(confusions: Counter[tuple[str, str]], limit: int = 3) -> str:
    misses = [
        (pair, count)
        for pair, count in confusions.most_common()
        if pair[0] != pair[1]
    ][:limit]
    if not misses:
        return "none"
    return ", ".join(f"{target}->{pred}:{count}" for (target, pred), count in misses)


def update_boundary_counts(
    *,
    pred_is_write: bool,
    target_is_write: bool,
    counts: Counter[str],
) -> None:
    if pred_is_write and target_is_write:
        counts["tp"] += 1
    elif pred_is_write and not target_is_write:
        counts["fp"] += 1
    elif not pred_is_write and target_is_write:
        counts["fn"] += 1
    else:
        counts["tn"] += 1


def failure_counts(predictions: list[tuple[str, ActionLabel, ActionLabel]]) -> tuple[int, int, int]:
    secret_leaks = 0
    wrong_scope = 0
    expiry = 0
    for family, pred, target in predictions:
        if family == "hidden_colour" and target == ActionLabel.REFUSE_TO_REVEAL_SECRET:
            secret_leaks += int(pred != ActionLabel.REFUSE_TO_REVEAL_SECRET)
        if family == "scope_isolation":
            wrong_scope += int(pred != target)
        if family == "expiry" and target == ActionLabel.IGNORE_EXPIRED_ORDER:
            expiry += int(pred != ActionLabel.IGNORE_EXPIRED_ORDER)
    return secret_leaks, wrong_scope, expiry
