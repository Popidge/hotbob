from hotbob.data.traces import generate_traces, write_jsonl
from hotbob.training.train import main


def test_write_jsonl_roundtrip_input(tmp_path) -> None:
    out = tmp_path / "traces.jsonl"
    write_jsonl(generate_traces(5), out)
    assert out.read_text(encoding="utf-8").count("\n") == 5


def test_smoke_train_cli(tmp_path, monkeypatch) -> None:
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
    main()
    assert (tmp_path / "runs" / "latest.pt").exists()
