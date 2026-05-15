from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rich.progress import track
from torch import nn
from torch.utils.data import DataLoader

from hotbob.data.traces import read_jsonl
from hotbob.model import MemoryBank
from hotbob.model.stateful_transformer import StatefulTransformer
from hotbob.training.dataset import (
    AUTHORITY_TO_ID,
    PRIVACY_TO_ID,
    TYPE_TO_ID,
    TraceDataset,
    collate_traces,
)
from hotbob.types import ActionLabel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    traces = read_jsonl(args.traces)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = TraceDataset(traces)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_traces,
    )
    d_model = 64 if args.smoke else 128
    num_memory_slots = 32
    model = StatefulTransformer(
        vocab_size=len(dataset.vocab),
        action_vocab_size=len(ActionLabel),
        d_model=d_model,
        num_memory_slots=num_memory_slots,
        num_types=len(TYPE_TO_ID),
        num_scopes=max(dataset.scope_vocab.values()) + 1,
        num_privacy=len(PRIVACY_TO_ID),
        num_authority=len(AUTHORITY_TO_ID),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    losses: list[float] = []
    data_iter = iter(loader)

    for step in track(range(args.steps), description="training"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = {key: value.to(device) for key, value in batch.items()}
        memory = MemoryBank(num_slots=num_memory_slots, d_model=d_model, device=device)
        memory.reset(batch["tokens"].shape[0])

        # Teacher-forced prefill: make a non-text memory vector available in the correct scope.
        for row in range(batch["tokens"].shape[0]):
            vector = model.transformer.embed(batch["tokens"][row, -1]).detach()
            memory.apply_write(
                row,
                int(batch["slot_ids"][row].item()),
                vector,
                type_id=int(batch["type_ids"][row].item()),
                scope_id=int(batch["scope_ids"][row].item()),
                privacy_id=int(batch["privacy_ids"][row].item()),
                authority_id=int(batch["authority_ids"][row].item()),
            )

        outputs = model(batch["tokens"], memory, batch["current_scope_ids"])
        loss = ce(outputs["action_logits"], batch["action_ids"])
        loss = loss + ce(outputs["op_logits"], batch["op_ids"])
        loss = loss + ce(outputs["slot_logits"], batch["slot_ids"])
        loss = loss + ce(outputs["type_logits"], batch["type_ids"])
        loss = loss + ce(outputs["scope_logits"], batch["scope_ids"])
        loss = loss + ce(outputs["privacy_logits"], batch["privacy_ids"])
        loss = loss + ce(outputs["authority_logits"], batch["authority_ids"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(f"step={step + 1} loss={losses[-1]:.4f}")

    Path("runs").mkdir(exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "vocab": dataset.vocab.token_to_id,
        "scope_vocab": dataset.scope_vocab,
        "num_traces": len(traces),
        "steps": args.steps,
        "device": device,
        "losses": losses,
        "config": {
            "d_model": d_model,
            "num_memory_slots": num_memory_slots,
            "action_vocab_size": len(ActionLabel),
        },
    }
    torch.save(checkpoint, "runs/latest.pt")
    print(f"saved checkpoint runs/latest.pt final_loss={losses[-1]:.4f} device={device}")


if __name__ == "__main__":
    main()
