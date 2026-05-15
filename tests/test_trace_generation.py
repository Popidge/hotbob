from hotbob.data.generate import main as generate_main
from hotbob.data.traces import generate_traces
from hotbob.types import MemoryAuthority, MemoryPrivacy, MemoryType


def test_generation_covers_all_families_and_ops_are_typed() -> None:
    traces = generate_traces(50, seed=7)
    assert {t.task_family for t in traces} == {
        "hidden_colour",
        "symbol_binding",
        "standing_order",
        "scope_isolation",
        "expiry",
    }
    for trace in traces:
        assert trace.expected_final_action
        assert trace.expected_memory_ops
        for op in trace.expected_memory_ops:
            assert isinstance(op.type, MemoryType)
            assert op.scope
            assert isinstance(op.authority, MemoryAuthority)
            assert isinstance(op.privacy, MemoryPrivacy)


def test_generate_train_eval_split(tmp_path, monkeypatch) -> None:
    train_out = tmp_path / "train.jsonl"
    eval_out = tmp_path / "eval.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate",
            "--train-out",
            str(train_out),
            "--eval-out",
            str(eval_out),
            "--train-n",
            "10",
            "--eval-n",
            "7",
            "--seed",
            "1",
        ],
    )
    generate_main()
    assert train_out.read_text(encoding="utf-8").count("\n") == 10
    assert eval_out.read_text(encoding="utf-8").count("\n") == 7
