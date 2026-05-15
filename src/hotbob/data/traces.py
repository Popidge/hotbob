from __future__ import annotations

import json
import random
from pathlib import Path

from hotbob.data.tasks_binding import make_symbol_binding_trace
from hotbob.data.tasks_colour import make_hidden_colour_trace
from hotbob.data.tasks_expiry import make_expiry_trace
from hotbob.data.tasks_scope import make_scope_isolation_trace
from hotbob.data.tasks_standing_order import make_standing_order_trace
from hotbob.types import TaskTrace

GENERATORS = [
    make_hidden_colour_trace,
    make_symbol_binding_trace,
    make_standing_order_trace,
    make_scope_isolation_trace,
    make_expiry_trace,
]


def generate_traces(n: int, seed: int = 0) -> list[TaskTrace]:
    rng = random.Random(seed)
    traces: list[TaskTrace] = []
    for i in range(n):
        generator = GENERATORS[i % len(GENERATORS)]
        traces.append(generator(rng, i))
    rng.shuffle(traces)
    return traces


def write_jsonl(traces: list[TaskTrace], out: str | Path) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace.model_dump(mode="json")) + "\n")


def read_jsonl(path: str | Path) -> list[TaskTrace]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [TaskTrace.model_validate_json(line) for line in f if line.strip()]
