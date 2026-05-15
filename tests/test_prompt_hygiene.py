from hotbob.data.hygiene import final_event_memory_leaks, is_memory_required_trace
from hotbob.data.traces import generate_traces


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
    symbol_actions = {
        trace.expected_final_action for trace in traces if trace.task_family == "symbol_binding"
    }
    assert len(symbol_actions) == 2
    expiry_actions = {
        trace.expected_final_action for trace in traces if trace.task_family == "expiry"
    }
    standing_actions = {
        trace.expected_final_action for trace in traces if trace.task_family == "standing_order"
    }
    assert len(expiry_actions) >= 3
    assert len(standing_actions) >= 3
