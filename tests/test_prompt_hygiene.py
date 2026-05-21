from hotbob.baselines import SymbolicMemoryBaseline
from hotbob.data.hygiene import final_event_memory_leaks, is_memory_required_trace
from hotbob.data.traces import generate_traces
from hotbob.llm.prompts import final_prompt_from_trace
from hotbob.types import ActionLabel


def test_memory_required_final_events_do_not_leak_memory_values() -> None:
    traces = generate_traces(100, seed=12)
    assert all(is_memory_required_trace(trace) for trace in traces)
    leaks = {
        idx: final_event_memory_leaks(trace)
        for idx, trace in enumerate(traces)
        if final_event_memory_leaks(trace)
    }
    assert leaks == {}


def test_memory_required_families_have_nontrivial_action_balance() -> None:
    traces = generate_traces(200, seed=13)
    for family in {trace.task_family for trace in traces}:
        actions = {trace.expected_final_action for trace in traces if trace.task_family == family}
        assert len(actions) >= 2


def test_hidden_private_payload_values_do_not_appear_in_final_prompts() -> None:
    traces = [
        trace
        for trace in generate_traces(100, seed=21)
        if any("HIDDEN_FROM_USER" in str(op.privacy) for op in trace.expected_memory_ops)
    ]
    assert traces
    for trace in traces:
        final = final_prompt_from_trace(trace)
        for op in trace.expected_memory_ops:
            if "HIDDEN_FROM_USER" in str(op.privacy):
                assert op.value not in final
        assert final_event_memory_leaks(trace) == []


def test_memory_required_final_prompts_have_action_variants() -> None:
    traces = generate_traces(300, seed=22)
    assert all("payload" not in final_prompt_from_trace(trace).lower() for trace in traces)
    assert any(
        trace.task_family == "privacy_disclosure_conflict"
        and trace.expected_final_action == ActionLabel.REFUSE_TO_REVEAL_SECRET
        for trace in traces
    )


def test_symbolic_baseline_solves_hardened_generated_traces() -> None:
    baseline = SymbolicMemoryBaseline()
    traces = generate_traces(100, seed=23)
    assert all(baseline.predict(trace) == trace.expected_final_action for trace in traces)
