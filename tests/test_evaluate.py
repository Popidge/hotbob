import json
from collections import Counter

from hotbob.data.traces import generate_traces, write_jsonl
from hotbob.training.evaluate import (
    diagnostic_scenario,
    top_confusions,
    update_family_diagnostics,
)
from hotbob.training.evaluate import main as evaluate_main
from hotbob.training.train import main as train_main
from hotbob.types import ActionLabel


def test_evaluate_loads_checkpoint(tmp_path, monkeypatch, capsys) -> None:
    traces = tmp_path / "traces.jsonl"
    write_jsonl(generate_traces(20), traces)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--traces",
            str(traces),
            "--steps",
            "2",
            "--smoke",
            "--batch-size",
            "4",
        ],
    )
    train_main()
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--checkpoint",
            "runs/latest.pt",
            "--traces",
            str(traces),
            "--batch-size",
            "4",
        ],
    )
    evaluate_main()
    output = capsys.readouterr().out
    assert "teacher-forced prewrite" in output
    assert "predicted-write prewrite" in output
    assert "sequential teacher-forced" in output
    assert "sequential predicted" in output
    assert "Memory write metrics" in output
    assert "Boundary write decision metrics" in output
    assert "Memory retrieval metrics" in output
    assert "Sequential predicted oracle ablations" in output
    assert "Task-family diagnostics" in output


def test_evaluate_debug_dump_redacts_private_values(tmp_path, monkeypatch) -> None:
    traces = tmp_path / "traces.jsonl"
    write_jsonl(generate_traces(20, seed=7), traces)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--traces",
            str(traces),
            "--steps",
            "2",
            "--smoke",
            "--batch-size",
            "4",
        ],
    )
    train_main()
    redacted = tmp_path / "debug-redacted.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--checkpoint",
            "runs/latest.pt",
            "--traces",
            str(traces),
            "--batch-size",
            "4",
            "--debug-dumps-out",
            str(redacted),
            "--debug-families",
            "privacy_disclosure_conflict",
            "--debug-max-traces",
            "1",
        ],
    )
    evaluate_main()

    row = json.loads(redacted.read_text(encoding="utf-8").splitlines()[0])
    assert row["task_family"] == "privacy_disclosure_conflict"
    assert row["active_memory_slots"]
    assert row["active_memory_slots"][0]["value"] == "<redacted>"

    unredacted = tmp_path / "debug-unredacted.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--checkpoint",
            "runs/latest.pt",
            "--traces",
            str(traces),
            "--batch-size",
            "4",
            "--debug-dumps-out",
            str(unredacted),
            "--debug-families",
            "privacy_disclosure_conflict",
            "--debug-max-traces",
            "1",
            "--debug-include-private-values",
        ],
    )
    evaluate_main()

    visible_row = json.loads(unredacted.read_text(encoding="utf-8").splitlines()[0])
    assert visible_row["active_memory_slots"][0]["value"] != "<redacted>"


def test_family_diagnostic_helpers_track_scenarios() -> None:
    trace = next(
        trace
        for trace in generate_traces(30, seed=5)
        if trace.task_family == "privacy_disclosure_conflict"
        and trace.expected_final_action == ActionLabel.REFUSE_TO_REVEAL_SECRET
    )
    action_confusion = {"privacy_disclosure_conflict": Counter()}
    read_mass_sum: Counter[str] = Counter()
    read_mass_count: Counter[str] = Counter()
    memory_total: Counter[str] = Counter()
    memory_correct: Counter[str] = Counter()

    update_family_diagnostics(
        trace=trace,
        pred=ActionLabel.ANSWER_NO,
        target=trace.expected_final_action,
        target_read_mass=0.75,
        action_confusion=action_confusion,
        read_mass_sum=read_mass_sum,
        read_mass_count=read_mass_count,
        memory_required_total=memory_total,
        memory_required_correct=memory_correct,
    )

    scenario = diagnostic_scenario(trace)
    assert scenario == "privacy_disclosure_conflict"
    assert action_confusion[scenario][("REFUSE_TO_REVEAL_SECRET", "ANSWER_NO")] == 1
    assert read_mass_sum[scenario] == 0.75
    assert read_mass_count[scenario] == 1
    assert memory_total[scenario] == 1
    assert memory_correct[scenario] == 0
    assert top_confusions(action_confusion[scenario]) == "REFUSE_TO_REVEAL_SECRET->ANSWER_NO:1"
