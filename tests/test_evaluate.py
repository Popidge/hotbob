from hotbob.data.traces import generate_traces, write_jsonl
from hotbob.training.evaluate import main as evaluate_main
from hotbob.training.train import main as train_main


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
