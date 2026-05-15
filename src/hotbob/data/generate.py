from __future__ import annotations

import argparse

from hotbob.data.traces import generate_traces, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=str, default="data/traces.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    write_jsonl(generate_traces(args.n, args.seed), args.out)


if __name__ == "__main__":
    main()
