from hotbob.data.traces import generate_traces, write_jsonl


def test_write_jsonl_roundtrip_input(tmp_path) -> None:
    out = tmp_path / "traces.jsonl"
    write_jsonl(generate_traces(5), out)
    assert out.read_text(encoding="utf-8").count("\n") == 5
