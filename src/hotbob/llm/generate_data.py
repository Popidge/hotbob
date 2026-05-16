from __future__ import annotations

import argparse

from hotbob.llm.dataset import generate_llm_traces, write_llm_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--train-n", type=int, default=10000)
    parser.add_argument("--eval-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    write_llm_jsonl(generate_llm_traces(args.train_n, args.seed), args.train_out)
    write_llm_jsonl(generate_llm_traces(args.eval_n, args.seed + 1), args.eval_out)


if __name__ == "__main__":
    main()
