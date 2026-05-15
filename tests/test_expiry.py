from hotbob.baselines import SymbolicMemoryBaseline
from hotbob.data.tasks_expiry import make_expiry_trace
from hotbob.types import ActionLabel


def test_expired_order_is_ignored() -> None:
    trace = make_expiry_trace(__import__("random").Random(4), 4)
    assert SymbolicMemoryBaseline().predict(trace) == ActionLabel.IGNORE_EXPIRED_ORDER
