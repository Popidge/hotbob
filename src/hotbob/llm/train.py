from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
from tqdm import tqdm

from hotbob.llm.dataset import build_llm_scope_vocab, read_llm_jsonl
from hotbob.llm.qwen_memory_model import QwenMemoryConfig, QwenMemoryModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--integration-mode",
        choices=["prefix", "layer_cross_attention"],
        default="prefix",
    )
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--write-loss-weight", type=float, default=0.2)
    parser.add_argument("--out", default="runs/qwen_memory/latest.pt")
    args = parser.parse_args()
    traces = read_llm_jsonl(args.traces)
    model = QwenMemoryModel(
        QwenMemoryConfig(
            model_name=args.model,
            freeze_base=args.freeze_base,
            integration_mode=args.integration_mode,
            scope_vocab=build_llm_scope_vocab(traces),
        )
    )
    optimizer = torch.optim.AdamW(model.memory_heads.parameters(), lr=1e-4)
    model.train()
    losses: list[float] = []
    for trace in tqdm(itertools.islice(itertools.cycle(traces), args.steps), total=args.steps):
        optimizer.zero_grad(set_to_none=True)
        lm_loss = model.teacher_forced_lm_loss(trace)
        write_loss = model.write_supervision_loss(trace)
        loss = lm_loss + args.write_loss_weight * write_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_heads_state": model.memory_heads.state_dict(),
            "config": model.config_obj.__dict__,
            "loss": losses[-1] if losses else None,
            "mean_loss": sum(losses) / len(losses) if losses else None,
        },
        args.out,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
