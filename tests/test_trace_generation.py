from hotbob.data.generate import main as generate_main
from hotbob.data.traces import generate_traces, read_jsonl, write_jsonl
from hotbob.training.dataset import TraceDataset, collate_traces
from hotbob.types import MemoryAuthority, MemoryOpName, MemoryPrivacy, MemoryType


def test_generation_covers_all_families_and_ops_are_typed() -> None:
    traces = generate_traces(50, seed=7)
    assert {t.task_family for t in traces} == {
        "standing_order",
        "active_expiry",
        "authority_conflict",
        "tool_verified_override",
        "interrupted_task",
        "stale_state_replacement",
        "privacy_disclosure_conflict",
        "multi_step_tool_routing",
    }
    for trace in traces:
        assert trace.expected_final_action
        assert trace.expected_memory_ops
        for op in trace.expected_memory_ops:
            assert isinstance(op.type, MemoryType)
            assert op.scope
            assert op.payload is not None
            assert isinstance(op.authority, MemoryAuthority)
            assert isinstance(op.privacy, MemoryPrivacy)


def test_generated_traces_round_trip_with_payloads(tmp_path) -> None:
    traces = generate_traces(16, seed=8)
    dumped = [trace.model_dump(mode="json") for trace in traces]
    assert all(row["expected_memory_ops"][0]["payload"]["kind"] for row in dumped)
    path = tmp_path / "traces.jsonl"
    write_jsonl(traces, path)
    loaded = read_jsonl(path)
    assert [trace.model_dump(mode="json") for trace in loaded] == dumped


def test_structured_payload_targets_are_collated() -> None:
    traces = generate_traces(64, seed=9)
    dataset = TraceDataset(traces)
    batch = collate_traces([dataset[idx] for idx in range(16)])
    for key in [
        "event_payload_kind_ids",
        "event_default_action_ids",
        "event_trigger_ids",
        "event_expiry_policy_ids",
        "event_authority_level_ids",
        "event_tool_name_ids",
        "event_route_step_ids",
        "event_has_payload",
        "event_has_tool_name",
    ]:
        assert key in batch
    assert batch["event_has_payload"].any()
    assert batch["event_has_tool_name"].any()
    standing = next(
        dataset[idx] for idx, trace in enumerate(traces) if trace.task_family == "standing_order"
    )
    assert standing.default_action_id > 0
    assert standing.trigger_id > 0
    assert standing.expiry_policy_id > 0
    assert standing.authority_level_id > 0
    routing = next(
        dataset[idx]
        for idx, trace in enumerate(traces)
        if trace.task_family == "multi_step_tool_routing"
    )
    assert routing.tool_name_id > 0
    assert routing.route_step_id >= 0
    assert any(
        op.op == MemoryOpName.UPDATE
        for trace in traces
        if trace.task_family == "stale_state_replacement"
        for op in trace.expected_memory_ops
    )
    assert any(
        op.op == MemoryOpName.DELETE
        for trace in traces
        if trace.task_family == "active_expiry"
        for op in trace.expected_memory_ops
    )


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
