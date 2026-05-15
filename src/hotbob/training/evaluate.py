from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
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
from hotbob.training.memory_teacher import build_predicted_memory, build_teacher_forced_memory
from hotbob.types import ActionLabel

ID_TO_ACTION = {idx: label for label, idx in ACTION_TO_ID.items()}


@dataclass(frozen=True)
class EvalResult:
    totals: Counter[str]
    correct: Counter[str]
    failures: tuple[int, int, int]
    write_accuracies: dict[str, float]


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
    memory_mode: str,
) -> EvalResult | None:
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
    write_totals: Counter[str] = Counter()
    write_correct: Counter[str] = Counter()
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            empty_memory = MemoryBank(
                num_slots=config["num_memory_slots"],
                d_model=config["d_model"],
                device=device,
            )
            empty_memory.reset(batch["tokens"].shape[0])
            prewrite_outputs = model(
                batch["tokens"], empty_memory, batch["current_scope_ids"], batch["lengths"]
            )
            write_targets = {
                "op": ("op_logits", "op_ids"),
                "slot": ("slot_logits", "slot_ids"),
                "type": ("type_logits", "type_ids"),
                "scope": ("scope_logits", "scope_ids"),
                "privacy": ("privacy_logits", "privacy_ids"),
                "authority": ("authority_logits", "authority_ids"),
            }
            for name, (logit_key, target_key) in write_targets.items():
                write_totals[name] += batch[target_key].numel()
                write_correct[name] += int(
                    (prewrite_outputs[logit_key].argmax(dim=-1) == batch[target_key]).sum().item()
                )

            if memory_mode == "teacher_forced":
                memory = build_teacher_forced_memory(
                    model_embed=model.transformer.embed,
                    memory_value_tokens=batch["memory_value_tokens"],
                    memory_value_mask=batch["memory_value_mask"],
                    slot_ids=batch["slot_ids"],
                    type_ids=batch["type_ids"],
                    scope_ids=batch["scope_ids"],
                    privacy_ids=batch["privacy_ids"],
                    authority_ids=batch["authority_ids"],
                    num_memory_slots=config["num_memory_slots"],
                    d_model=config["d_model"],
                    device=device,
                )
            elif memory_mode == "predicted":
                memory = build_predicted_memory(
                    outputs=prewrite_outputs,
                    batch_size=batch["tokens"].shape[0],
                    num_memory_slots=config["num_memory_slots"],
                    d_model=config["d_model"],
                    device=device,
                )
            else:
                memory = empty_memory
            outputs = model(batch["tokens"], memory, batch["current_scope_ids"], batch["lengths"])
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
    return EvalResult(
        totals=totals,
        correct=correct,
        failures=failure_counts(predictions),
        write_accuracies={
            name: write_correct[name] / write_totals[name]
            for name in sorted(write_totals)
            if write_totals[name]
        },
    )


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
        memory_mode="context_only",
    )
    neural_result = evaluate_neural(
        args.checkpoint,
        traces,
        args.batch_size,
        device,
        memory_mode="teacher_forced",
    )
    predicted_result = evaluate_neural(
        args.checkpoint,
        traces,
        args.batch_size,
        device,
        memory_mode="predicted",
    )

    table = Table(title=f"HotBob evaluation ({args.checkpoint})")
    table.add_column("Task family")
    table.add_column("Symbolic accuracy")
    table.add_column("Context-only accuracy")
    table.add_column("Neural TF memory accuracy")
    table.add_column("Predicted-write accuracy")
    all_families = sorted(symbolic_totals)
    for family in all_families:
        context_cell = "n/a"
        neural_cell = "n/a"
        if context_result is not None:
            context_cell = f"{context_result.correct[family] / context_result.totals[family]:.3f}"
        if neural_result is not None:
            neural_cell = f"{neural_result.correct[family] / neural_result.totals[family]:.3f}"
        predicted_cell = "n/a"
        if predicted_result is not None:
            predicted_cell = (
                f"{predicted_result.correct[family] / predicted_result.totals[family]:.3f}"
            )
        table.add_row(
            family,
            f"{symbolic_correct[family] / symbolic_totals[family]:.3f}",
            context_cell,
            neural_cell,
            predicted_cell,
        )
    context_failures = (
        context_result.failures if context_result is not None else ("n/a", "n/a", "n/a")
    )
    neural_failures = neural_result.failures if neural_result is not None else ("n/a", "n/a", "n/a")
    predicted_failures = (
        predicted_result.failures if predicted_result is not None else ("n/a", "n/a", "n/a")
    )
    table.add_row(
        "secret leak failures",
        str(symbolic_failures[0]),
        str(context_failures[0]),
        str(neural_failures[0]),
        str(predicted_failures[0]),
    )
    table.add_row(
        "wrong-scope retrieval failures",
        str(symbolic_failures[1]),
        str(context_failures[1]),
        str(neural_failures[1]),
        str(predicted_failures[1]),
    )
    table.add_row(
        "expiry failures",
        str(symbolic_failures[2]),
        str(context_failures[2]),
        str(neural_failures[2]),
        str(predicted_failures[2]),
    )
    memory_table = Table(title="Memory write metrics")
    memory_table.add_column("Mode")
    memory_table.add_column("Head")
    memory_table.add_column("Accuracy")
    for name, result in [
        ("context-only prewrite", context_result),
        ("teacher-forced prewrite", neural_result),
        ("predicted-write prewrite", predicted_result),
    ]:
        if result is None:
            memory_table.add_row(name, "all", "n/a")
        else:
            for head, value in result.write_accuracies.items():
                memory_table.add_row(name, head, f"{value:.3f}")
    Console().print(table)
    Console().print(memory_table)


if __name__ == "__main__":
    main()
