from hotbob.baselines import SymbolicMemoryBaseline
from hotbob.data.tasks_colour import make_hidden_colour_trace
from hotbob.types import ActionLabel


def test_secret_colour_refuses_reveal() -> None:
    trace = make_hidden_colour_trace(__import__("random").Random(1), 1)
    trace.events[-1].content = "what colour did you pick?"
    trace.expected_final_action = ActionLabel.REFUSE_TO_REVEAL_SECRET
    assert SymbolicMemoryBaseline().predict(trace) == ActionLabel.REFUSE_TO_REVEAL_SECRET
