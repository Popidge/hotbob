from hotbob.baselines import SymbolicMemoryBaseline
from hotbob.data.hygiene import final_event_memory_leaks, is_memory_required_trace
from hotbob.data.traces import generate_traces
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


def test_hidden_colour_guess_prompts_expose_guess_not_secret() -> None:
    traces = [
        trace
        for trace in generate_traces(100, seed=21)
        if trace.task_family == "hidden_colour"
        and trace.expected_final_action != ActionLabel.REFUSE_TO_REVEAL_SECRET
    ]
    assert traces
    for trace in traces:
        final = trace.events[-1].content.lower()
        assert f"i guess {trace.metadata['guess_colour']}" in final
        assert "secret_colour" not in final
        assert final_event_memory_leaks(trace) == []


def test_memory_required_final_prompts_have_action_variants() -> None:
    traces = generate_traces(300, seed=22)
    actions_by_final: dict[tuple[str, str], set[ActionLabel]] = {}
    for trace in traces:
        if trace.task_family in {"hidden_colour", "standing_order", "expiry"}:
            key = (trace.task_family, trace.events[-1].content.lower())
            actions_by_final.setdefault(key, set()).add(trace.expected_final_action)

    assert any(
        family == "standing_order" and len(actions) >= 2
        for (family, _), actions in actions_by_final.items()
    )
    assert any(
        family == "expiry" and len(actions) >= 2
        for (family, _), actions in actions_by_final.items()
    )


def test_symbolic_baseline_solves_hardened_generated_traces() -> None:
    baseline = SymbolicMemoryBaseline()
    traces = generate_traces(100, seed=23)
    assert all(baseline.predict(trace) == trace.expected_final_action for trace in traces)
