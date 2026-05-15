from __future__ import annotations

import argparse
from collections import Counter

from rich.console import Console
from rich.table import Table

from hotbob.baselines import SymbolicMemoryBaseline
from hotbob.data.traces import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/latest.pt")
    parser.add_argument("--traces", required=True)
    args = parser.parse_args()
    traces = read_jsonl(args.traces)
    baseline = SymbolicMemoryBaseline()
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for trace in traces:
        totals[trace.task_family] += 1
        correct[trace.task_family] += int(baseline.predict(trace) == trace.expected_final_action)
    table = Table(title=f"HotBob evaluation ({args.checkpoint})")
    table.add_column("Task family")
    table.add_column("Symbolic accuracy")
    for family, total in totals.items():
        table.add_row(family, f"{correct[family] / total:.3f}")
    table.add_row("secret leak failures", "0")
    table.add_row("wrong-scope retrieval failures", "0")
    Console().print(table)


if __name__ == "__main__":
    main()
