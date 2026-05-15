from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hotbob.data.traces import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    traces = read_jsonl(args.traces)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path("runs").mkdir(exist_ok=True)
    torch.save({"num_traces": len(traces), "steps": args.steps, "device": device}, "runs/latest.pt")
    print(f"smoke train placeholder: traces={len(traces)} steps={args.steps} device={device}")
