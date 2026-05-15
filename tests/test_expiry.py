from hotbob.baselines import SymbolicMemoryBaseline
from hotbob.data.tasks_expiry import make_expiry_trace
from hotbob.types import ActionLabel


def test_expired_order_is_ignored() -> None:
    rng = __import__("random").Random(4)
    traces = [make_expiry_trace(rng, idx) for idx in range(20)]
    trace = next(trace for trace in traces if not trace.metadata["active_preference"])
    assert SymbolicMemoryBaseline().predict(trace) == ActionLabel.IGNORE_EXPIRED_ORDER
