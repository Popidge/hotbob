from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

from hotbob.baselines import SymbolicMemoryBaseline
from hotbob.data.traces import read_jsonl
from hotbob.model import MemoryBank
from hotbob.model.stateful_transformer import StatefulTransformer
from hotbob.training.dataset import (
    ACTION_TO_ID,
    AUTHORITY_TO_ID,
    PRIVACY_TO_ID,
    TYPE_TO_ID,
    TraceDataset,
    TraceVocab,
    collate_traces,
)
from hotbob.training.memory_teacher import build_teacher_forced_memory
from hotbob.types import ActionLabel

ID_TO_ACTION = {idx: label for label, idx in ACTION_TO_ID.items()}


def evaluate_symbolic(traces) -> tuple[Counter[str], Counter[str], tuple[int, int, int]]:
    baseline = SymbolicMemoryBaseline()
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    predictions: list[tuple[str, ActionLabel, ActionLabel]] = []
    for trace in traces:
        pred = baseline.predict(trace)
        totals[trace.task_family] += 1
        correct[trace.task_family] += int(pred == trace.expected_final_action)
        predictions.append((trace.task_family, pred, trace.expected_final_action))
    return totals, correct, failure_counts(predictions)


def failure_counts(predictions: list[tuple[str, ActionLabel, ActionLabel]]) -> tuple[int, int, int]:
    secret_leaks = 0
    wrong_scope = 0
    expiry = 0
    for family, pred, target in predictions:
        if family == "hidden_colour" and target == ActionLabel.REFUSE_TO_REVEAL_SECRET:
            secret_leaks += int(pred != ActionLabel.REFUSE_TO_REVEAL_SECRET)
        if family == "scope_isolation":
            wrong_scope += int(pred != target)
        if family == "expiry":
            expiry += int(pred != ActionLabel.IGNORE_EXPIRED_ORDER)
    return secret_leaks, wrong_scope, expiry


def evaluate_neural(
    checkpoint_path: str,
    traces,
    batch_size: int,
    device: str,
    *,
    teacher_force_memory: bool,
) -> tuple[Counter[str], Counter[str], tuple[int, int, int]] | None:
    if not Path(checkpoint_path).exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = TraceVocab.from_token_to_id(checkpoint["vocab"])
    dataset = TraceDataset(
        traces,
        vocab=vocab,
        scope_vocab=checkpoint["scope_vocab"],
    )
    config = checkpoint["config"]
    model = StatefulTransformer(
        vocab_size=len(vocab),
        action_vocab_size=config["action_vocab_size"],
        d_model=config["d_model"],
        num_memory_slots=config["num_memory_slots"],
        num_types=len(TYPE_TO_ID),
        num_scopes=max(dataset.scope_vocab.values()) + 1,
        num_privacy=len(PRIVACY_TO_ID),
        num_authority=len(AUTHORITY_TO_ID),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_traces)
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    predictions: list[tuple[str, ActionLabel, ActionLabel]] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            if teacher_force_memory:
                memory = build_teacher_forced_memory(
                    model_embed=model.transformer.embed,
                    tokens=batch["tokens"],
                    slot_ids=batch["slot_ids"],
                    type_ids=batch["type_ids"],
                    scope_ids=batch["scope_ids"],
                    privacy_ids=batch["privacy_ids"],
                    authority_ids=batch["authority_ids"],
                    num_memory_slots=config["num_memory_slots"],
                    d_model=config["d_model"],
                    device=device,
                )
            else:
                memory = MemoryBank(
                    num_slots=config["num_memory_slots"],
                    d_model=config["d_model"],
                    device=device,
                )
                memory.reset(batch["tokens"].shape[0])
            outputs = model(batch["tokens"], memory, batch["current_scope_ids"])
            pred_ids = outputs["action_logits"].argmax(dim=-1).cpu().tolist()
            target_ids = batch["action_ids"].cpu().tolist()
            for row, (pred_id, target_id) in enumerate(zip(pred_ids, target_ids, strict=True)):
                trace = traces[offset + row]
                pred = ID_TO_ACTION[pred_id]
                target = ID_TO_ACTION[target_id]
                totals[trace.task_family] += 1
                correct[trace.task_family] += int(pred == target)
                predictions.append((trace.task_family, pred, target))
            offset += len(pred_ids)
    return totals, correct, failure_counts(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/latest.pt")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    traces = read_jsonl(args.traces)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    symbolic_totals, symbolic_correct, symbolic_failures = evaluate_symbolic(traces)
    context_result = evaluate_neural(
        args.checkpoint,
        traces,
        args.batch_size,
        device,
        teacher_force_memory=False,
    )
    neural_result = evaluate_neural(
        args.checkpoint,
        traces,
        args.batch_size,
        device,
        teacher_force_memory=True,
    )

    table = Table(title=f"HotBob evaluation ({args.checkpoint})")
    table.add_column("Task family")
    table.add_column("Symbolic accuracy")
    table.add_column("Context-only accuracy")
    table.add_column("Neural TF memory accuracy")
    all_families = sorted(symbolic_totals)
    for family in all_families:
        context_cell = "n/a"
        neural_cell = "n/a"
        if context_result is not None:
            context_totals, context_correct, _ = context_result
            context_cell = f"{context_correct[family] / context_totals[family]:.3f}"
        if neural_result is not None:
            neural_totals, neural_correct, _ = neural_result
            neural_cell = f"{neural_correct[family] / neural_totals[family]:.3f}"
        table.add_row(
            family,
            f"{symbolic_correct[family] / symbolic_totals[family]:.3f}",
            context_cell,
            neural_cell,
        )
    context_failures = context_result[2] if context_result is not None else ("n/a", "n/a", "n/a")
    neural_failures = neural_result[2] if neural_result is not None else ("n/a", "n/a", "n/a")
    table.add_row(
        "secret leak failures",
        str(symbolic_failures[0]),
        str(context_failures[0]),
        str(neural_failures[0]),
    )
    table.add_row(
        "wrong-scope retrieval failures",
        str(symbolic_failures[1]),
        str(context_failures[1]),
        str(neural_failures[1]),
    )
    table.add_row(
        "expiry failures",
        str(symbolic_failures[2]),
        str(context_failures[2]),
        str(neural_failures[2]),
    )
    Console().print(table)


if __name__ == "__main__":
    main()
