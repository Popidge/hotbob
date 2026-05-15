from hotbob.baselines import SymbolicMemoryBaseline
from hotbob.data.tasks_scope import make_scope_isolation_trace
from hotbob.types import ActionLabel


def test_scope_isolation_repo_a_uses_dave() -> None:
    trace = make_scope_isolation_trace(__import__("random").Random(1), 2)
    trace.current_scope = trace.expected_memory_ops[0].scope
    trace.events[-1].scope = trace.current_scope
    trace.expected_memory_ops[0].value = "dave"
    assert SymbolicMemoryBaseline().predict(trace) == ActionLabel.INSPECT_FUNCTION_DAVE


def test_scope_isolation_repo_b_does_not_use_repo_a_binding() -> None:
    trace = make_scope_isolation_trace(__import__("random").Random(2), 2)
    assert SymbolicMemoryBaseline().predict(trace) == trace.expected_final_action
