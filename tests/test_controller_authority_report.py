from __future__ import annotations

from collections import Counter

import pytest

from hotbob.data.traces import generate_weighted_traces
from hotbob.training.authority_report import build_controller_authority_report
from hotbob.training.evaluate import EvalResult


def fake_result(authority_correct: int) -> EvalResult:
    return EvalResult(
        totals=Counter({"authority_conflict": 4, "tool_verified_override": 2}),
        correct=Counter({"authority_conflict": authority_correct, "tool_verified_override": 1}),
        memory_required_total=6,
        memory_required_correct=authority_correct + 1,
        failures=(0, 0, 0),
        write_accuracies={"op": 0.9, "authority": 0.8, "value_class": 0.7},
        boundary_tp=5,
        boundary_fp=1,
        boundary_fn=1,
        boundary_tn=3,
        read_target_mass_sum=3.0,
        read_target_mass_count=6,
        family_action_confusion={
            "authority_conflict": Counter({("HOLD_FIRE", "HOLD_FIRE"): 2})
        },
        family_read_mass_sum=Counter({"authority_conflict": 2.0}),
        family_read_mass_count=Counter({"authority_conflict": 4}),
        family_memory_required_total=Counter({"authority_conflict": 4}),
        family_memory_required_correct=Counter({"authority_conflict": authority_correct}),
    )


def test_controller_authority_report_summarizes_predicted_and_oracles(monkeypatch) -> None:
    traces = generate_weighted_traces(
        12,
        seed=9,
        family_weights={"authority_conflict": 8.0, "tool_verified_override": 2.0},
    )
    calls: list[str] = []

    def fake_evaluate(checkpoint, traces, device, *, memory_mode):
        calls.append(memory_mode)
        return fake_result(3 if memory_mode == "predicted" else 4)

    monkeypatch.setattr(
        "hotbob.training.authority_report.evaluate_sequential_neural",
        fake_evaluate,
    )
    report = build_controller_authority_report(
        checkpoint="runs/latest.pt",
        traces=traces,
        device="cpu",
    )

    assert "predicted" in report["modes"]
    assert "predicted_oracle_all" in report["modes"]
    assert report["modes"]["predicted"]["authority_accuracy"] == pytest.approx(0.75)
    assert report["modes"]["predicted"]["authority_probe_accuracy"] == pytest.approx(4 / 6)
    assert report["modes"]["teacher_forced"]["write_accuracies"]["authority"] == pytest.approx(0.8)
    assert report["metadata"]["authority_total"] > 0
    assert set(calls) == {
        "context_only",
        "teacher_forced",
        "predicted",
        "predicted_oracle_slot",
        "predicted_oracle_scope",
        "predicted_oracle_value",
        "predicted_oracle_slot_oracle_scope_oracle_value",
    }
