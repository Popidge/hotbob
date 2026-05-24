from __future__ import annotations

import json
import random
from pathlib import Path

from hotbob.data.tasks_active_expiry import make_active_expiry_trace
from hotbob.data.tasks_authority_conflict import make_authority_conflict_trace
from hotbob.data.tasks_interrupted_task import make_interrupted_task_trace
from hotbob.data.tasks_multi_step_tool_routing import make_multi_step_tool_routing_trace
from hotbob.data.tasks_privacy_disclosure_conflict import (
    make_privacy_disclosure_conflict_trace,
)
from hotbob.data.tasks_rich_standing_order import make_rich_standing_order_trace
from hotbob.data.tasks_stale_state_replacement import make_stale_state_replacement_trace
from hotbob.data.tasks_tool_verified_override import make_tool_verified_override_trace
from hotbob.types import TaskTrace

GENERATOR_BY_FAMILY = {
    "standing_order": make_rich_standing_order_trace,
    "active_expiry": make_active_expiry_trace,
    "authority_conflict": make_authority_conflict_trace,
    "tool_verified_override": make_tool_verified_override_trace,
    "interrupted_task": make_interrupted_task_trace,
    "stale_state_replacement": make_stale_state_replacement_trace,
    "privacy_disclosure_conflict": make_privacy_disclosure_conflict_trace,
    "multi_step_tool_routing": make_multi_step_tool_routing_trace,
}

GENERATORS = list(GENERATOR_BY_FAMILY.values())


def generate_traces(n: int, seed: int = 0) -> list[TaskTrace]:
    rng = random.Random(seed)
    traces: list[TaskTrace] = []
    for i in range(n):
        generator = GENERATORS[i % len(GENERATORS)]
        trace = generator(rng, i)
        trace.metadata.setdefault("memory_required", True)
        trace.metadata.setdefault("structured_payload_required", True)
        trace.metadata.setdefault("final_event_hides_memory_value", True)
        traces.append(trace)
    rng.shuffle(traces)
    return traces


def generate_weighted_traces(
    n: int,
    seed: int = 0,
    family_weights: dict[str, float] | None = None,
) -> list[TaskTrace]:
    if family_weights is None:
        return generate_traces(n, seed)
    invalid = sorted(set(family_weights) - set(GENERATOR_BY_FAMILY))
    if invalid:
        valid = ", ".join(sorted(GENERATOR_BY_FAMILY))
        raise ValueError(f"Unknown family weight(s): {', '.join(invalid)}. Valid families: {valid}")
    rng = random.Random(seed)
    families = list(GENERATOR_BY_FAMILY)
    weights = [family_weights.get(family, 1.0) for family in families]
    traces: list[TaskTrace] = []
    for i in range(n):
        family = rng.choices(families, weights=weights, k=1)[0]
        trace = GENERATOR_BY_FAMILY[family](rng, i)
        trace.metadata.setdefault("memory_required", True)
        trace.metadata.setdefault("structured_payload_required", True)
        trace.metadata.setdefault("final_event_hides_memory_value", True)
        traces.append(trace)
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
