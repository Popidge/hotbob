from hotbob.baselines import SymbolicMemoryBaseline
from hotbob.data.tasks_scope import make_scope_isolation_trace
from hotbob.types import ActionLabel


def test_scope_isolation_repo_a_uses_dave() -> None:
    trace = make_scope_isolation_trace(__import__("random").Random(2), 2)
    trace.current_scope = "repo_a"
    trace.events[-1].scope = "repo_a"
    assert SymbolicMemoryBaseline().predict(trace) == ActionLabel.INSPECT_FUNCTION_DAVE


def test_scope_isolation_repo_b_does_not_use_repo_a_binding() -> None:
    trace = make_scope_isolation_trace(__import__("random").Random(2), 2)
    trace.current_scope = "repo_b"
    trace.events[-1].scope = "repo_b"
    assert (
        SymbolicMemoryBaseline().predict(trace)
        == ActionLabel.INSPECT_FUNCTION_CALCULATE_FINAL_SCORE
    )
