from __future__ import annotations

import argparse

from hotbob.data.traces import generate_traces, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=str, default="data/traces.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-out", type=str, default=None)
    parser.add_argument("--eval-out", type=str, default=None)
    parser.add_argument("--train-n", type=int, default=1000)
    parser.add_argument("--eval-n", type=int, default=250)
    parser.add_argument("--eval-seed", type=int, default=None)
    args = parser.parse_args()
    if args.train_out or args.eval_out:
        train_out = args.train_out or "data/train.jsonl"
        eval_out = args.eval_out or "data/eval.jsonl"
        eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 10_000
        write_jsonl(generate_traces(args.train_n, args.seed), train_out)
        write_jsonl(generate_traces(args.eval_n, eval_seed), eval_out)
        return
    write_jsonl(generate_traces(args.n, args.seed), args.out)


if __name__ == "__main__":
    main()
