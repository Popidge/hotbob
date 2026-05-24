from __future__ import annotations

import argparse

from hotbob.data.traces import generate_weighted_traces, write_jsonl


def parse_family_weights(values: list[str] | None) -> dict[str, float] | None:
    if not values:
        return None
    weights: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid family weight {value!r}; expected FAMILY=FLOAT")
        family, raw_weight = value.split("=", 1)
        weights[family] = float(raw_weight)
    return weights


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
    parser.add_argument("--family-weight", action="append", default=None)
    args = parser.parse_args()
    family_weights = parse_family_weights(args.family_weight)
    if args.train_out or args.eval_out:
        train_out = args.train_out or "data/train.jsonl"
        eval_out = args.eval_out or "data/eval.jsonl"
        eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 10_000
        write_jsonl(generate_weighted_traces(args.train_n, args.seed, family_weights), train_out)
        write_jsonl(generate_weighted_traces(args.eval_n, eval_seed, family_weights), eval_out)
        return
    write_jsonl(generate_weighted_traces(args.n, args.seed, family_weights), args.out)


if __name__ == "__main__":
    main()
