from __future__ import annotations

import argparse

from hotbob.llm.dataset import generate_weighted_llm_traces, write_llm_jsonl


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
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--train-n", type=int, default=10000)
    parser.add_argument("--eval-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--family-weight", action="append", default=None)
    args = parser.parse_args()
    family_weights = parse_family_weights(args.family_weight)
    write_llm_jsonl(
        generate_weighted_llm_traces(args.train_n, args.seed, family_weights),
        args.train_out,
    )
    write_llm_jsonl(
        generate_weighted_llm_traces(args.eval_n, args.seed + 1, family_weights),
        args.eval_out,
    )


if __name__ == "__main__":
    main()
