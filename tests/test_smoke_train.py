import torch

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
    checkpoint_path = tmp_path / "runs" / "latest.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["config"]["sequential_controller_loss"] is True
    assert checkpoint["config"]["sequential_predicted_warmup_steps"] == 0
    assert checkpoint["config"]["structured_loss_weight"] == 0.2
    assert checkpoint["config"]["num_payload_kinds"] > 1
    assert checkpoint["config"]["num_policy_actions"] > 1
    assert any(key.endswith("payload_kind_head.weight") for key in checkpoint["model_state"])
