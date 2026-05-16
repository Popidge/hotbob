from __future__ import annotations

import argparse

from hotbob.llm.qwen_memory_model import QwenMemoryConfig, QwenMemoryModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--scope", default="default")
    parser.add_argument("--no-memory", action="store_true")
    args = parser.parse_args()
    model = QwenMemoryModel(QwenMemoryConfig(model_name=args.model))
    print(
        model.generate_final(
            args.prompt,
            current_scope=args.scope,
            use_memory=not args.no_memory,
        )
    )


if __name__ == "__main__":
    main()
