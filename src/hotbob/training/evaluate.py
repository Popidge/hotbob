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
from hotbob.data.hygiene import is_memory_required_trace
from hotbob.data.traces import read_jsonl
from hotbob.model import MemoryBank
from hotbob.model.stateful_transformer import StatefulTransformer
from hotbob.training.dataset import (
    ACTION_TO_ID,
    AUTHORITY_TO_ID,
    OP_TO_ID,
    PRIVACY_TO_ID,
    TYPE_TO_ID,
    TraceDataset,
    TraceVocab,
    collate_traces,
    tokenize_text,
)
from hotbob.training.memory_teacher import (
    build_predicted_memory,
    build_teacher_forced_memory,
    mean_value_embedding,
)
from hotbob.types import ActionLabel, MemoryOpName, TaskTrace, TraceEvent

ID_TO_ACTION = {idx: label for label, idx in ACTION_TO_ID.items()}
ID_TO_OP = {idx: label for label, idx in OP_TO_ID.items()}


@dataclass(frozen=True)
class EvalResult:
    totals: Counter[str]
    correct: Counter[str]
    memory_required_total: int
    memory_required_correct: int
    failures: tuple[int, int, int]
    write_accuracies: dict[str, float]
    boundary_tp: int = 0
    boundary_fp: int = 0
    boundary_fn: int = 0
    boundary_tn: int = 0
    read_target_mass_sum: float = 0.0
    read_target_mass_count: int = 0


def boundary_scores(result: EvalResult | None) -> tuple[str, str, str]:
    if result is None:
        return "n/a", "n/a", "n/a"
    precision_den = result.boundary_tp + result.boundary_fp
    recall_den = result.boundary_tp + result.boundary_fn
    precision = result.boundary_tp / precision_den if precision_den else 0.0
    recall = result.boundary_tp / recall_den if recall_den else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"


def update_boundary_counts(
    *,
    pred_is_write: bool,
    target_is_write: bool,
    counts: Counter[str],
) -> None:
    if pred_is_write and target_is_write:
        counts["tp"] += 1
    elif pred_is_write and not target_is_write:
        counts["fp"] += 1
    elif not pred_is_write and target_is_write:
        counts["fn"] += 1
    else:
        counts["tn"] += 1


def read_target_mass(result: EvalResult | None) -> str:
    if result is None or result.read_target_mass_count == 0:
        return "n/a"
    return f"{result.read_target_mass_sum / result.read_target_mass_count:.3f}"


def relevant_slot_index(trace: TaskTrace) -> int:
    for idx, op in enumerate(trace.expected_memory_ops):
        if op.scope == trace.current_scope:
            return idx
    return 0


def evaluate_symbolic(traces) -> tuple[Counter[str], Counter[str], tuple[int, int, int]]:
    baseline = SymbolicMemoryBaseline()
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    memory_required_total = 0
    memory_required_correct = 0
    predictions: list[tuple[str, ActionLabel, ActionLabel]] = []
    for trace in traces:
        pred = baseline.predict(trace)
        totals[trace.task_family] += 1
        correct[trace.task_family] += int(pred == trace.expected_final_action)
        if is_memory_required_trace(trace):
            memory_required_total += 1
            memory_required_correct += int(pred == trace.expected_final_action)
        predictions.append((trace.task_family, pred, trace.expected_final_action))
    return (
        totals,
        correct,
        memory_required_total,
        memory_required_correct,
        failure_counts(predictions),
    )


def failure_counts(predictions: list[tuple[str, ActionLabel, ActionLabel]]) -> tuple[int, int, int]:
    secret_leaks = 0
    wrong_scope = 0
    expiry = 0
    for family, pred, target in predictions:
        if family == "hidden_colour" and target == ActionLabel.REFUSE_TO_REVEAL_SECRET:
            secret_leaks += int(pred != ActionLabel.REFUSE_TO_REVEAL_SECRET)
        if family == "scope_isolation":
            wrong_scope += int(pred != target)
        if family == "expiry" and target == ActionLabel.IGNORE_EXPIRED_ORDER:
            expiry += int(pred != ActionLabel.IGNORE_EXPIRED_ORDER)
    return secret_leaks, wrong_scope, expiry


def memory_required_accuracy(result: EvalResult | None) -> str:
    if result is None or result.memory_required_total == 0:
        return "n/a"
    return f"{result.memory_required_correct / result.memory_required_total:.3f}"


def family_accuracy(result: EvalResult | None, family: str) -> str:
    if result is None or result.totals[family] == 0:
        return "n/a"
    return f"{result.correct[family] / result.totals[family]:.3f}"


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
        max_seq_len=config.get("max_seq_len", 256),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_traces)
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    memory_required_total = 0
    memory_required_correct = 0
    predictions: list[tuple[str, ActionLabel, ActionLabel]] = []
    write_totals: Counter[str] = Counter()
    write_correct: Counter[str] = Counter()
    boundary_counts: Counter[str] = Counter()
    read_target_mass_sum = 0.0
    read_target_mass_count = 0
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
                batch["write_tokens"], empty_memory, batch["scope_ids"], batch["write_lengths"]
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
            predicted_write = (
                prewrite_outputs["op_logits"].argmax(dim=-1) != OP_TO_ID[MemoryOpName.NOOP]
            )
            target_write = batch["op_ids"] != OP_TO_ID[MemoryOpName.NOOP]
            for pred, target in zip(predicted_write.tolist(), target_write.tolist(), strict=True):
                update_boundary_counts(
                    pred_is_write=bool(pred),
                    target_is_write=bool(target),
                    counts=boundary_counts,
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
            target_read_mass = outputs["read_attention"][
                torch.arange(batch["tokens"].shape[0], device=device),
                outputs["boundary_indices"],
                batch["slot_ids"],
            ]
            read_target_mass_sum += float(target_read_mass.sum().cpu().item())
            read_target_mass_count += int(target_read_mass.numel())
            pred_ids = outputs["action_logits"].argmax(dim=-1).cpu().tolist()
            target_ids = batch["action_ids"].cpu().tolist()
            for row, (pred_id, target_id) in enumerate(zip(pred_ids, target_ids, strict=True)):
                trace = traces[offset + row]
                pred = ID_TO_ACTION[pred_id]
                target = ID_TO_ACTION[target_id]
                totals[trace.task_family] += 1
                correct[trace.task_family] += int(pred == target)
                if is_memory_required_trace(trace):
                    memory_required_total += 1
                    memory_required_correct += int(pred == target)
                predictions.append((trace.task_family, pred, target))
            offset += len(pred_ids)
    return EvalResult(
        totals=totals,
        correct=correct,
        memory_required_total=memory_required_total,
        memory_required_correct=memory_required_correct,
        failures=failure_counts(predictions),
        write_accuracies={
            name: write_correct[name] / write_totals[name]
            for name in sorted(write_totals)
            if write_totals[name]
        },
        boundary_tp=boundary_counts["tp"],
        boundary_fp=boundary_counts["fp"],
        boundary_fn=boundary_counts["fn"],
        boundary_tn=boundary_counts["tn"],
        read_target_mass_sum=read_target_mass_sum,
        read_target_mass_count=read_target_mass_count,
    )


def load_model_and_dataset(
    checkpoint_path: str,
    traces: list[TaskTrace],
    device: str,
) -> tuple[StatefulTransformer, TraceDataset, dict] | None:
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
        max_seq_len=config.get("max_seq_len", 256),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, dataset, config


def encode_event_tensor(
    dataset: TraceDataset,
    event: TraceEvent,
    current_scope: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = dataset._encode_event(event, current_scope)[-dataset.max_seq_len :]
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=device)
    return tokens, lengths


def value_vector_for_op(
    model: StatefulTransformer,
    dataset: TraceDataset,
    op_value: str,
    device: str,
) -> torch.Tensor:
    value_ids = dataset.vocab.encode(tokenize_text(op_value)) or [0]
    value_tokens = torch.tensor([value_ids], dtype=torch.long, device=device)
    value_mask = torch.ones_like(value_tokens, dtype=torch.bool)
    return mean_value_embedding(model.transformer.embed, value_tokens, value_mask)[0].detach()


def apply_teacher_op(
    memory: MemoryBank,
    model: StatefulTransformer,
    dataset: TraceDataset,
    trace: TaskTrace,
    op_index: int,
    device: str,
) -> None:
    op = trace.expected_memory_ops[op_index]
    slot_idx = min(op_index, memory.num_slots - 1)
    if op.op == MemoryOpName.DELETE:
        for slot in range(memory.num_slots):
            if (
                bool(memory.occupied[0, slot].item())
                and int(memory.scope_ids[0, slot].item()) == dataset.scope_vocab.get(op.scope, -1)
                and int(memory.type_ids[0, slot].item()) == TYPE_TO_ID[op.type]
            ):
                memory.apply_delete(0, slot)
                return
        return
    vector = value_vector_for_op(model, dataset, op.value, device)
    if op.op == MemoryOpName.UPDATE and bool(memory.occupied[0, slot_idx].item()):
        memory.apply_update(0, slot_idx, vector)
        return
    memory.apply_write(
        0,
        slot_idx,
        vector,
        type_id=TYPE_TO_ID[op.type],
        scope_id=dataset.scope_vocab.get(op.scope, 0),
        privacy_id=PRIVACY_TO_ID[op.privacy],
        authority_id=AUTHORITY_TO_ID[op.authority],
    )


def apply_predicted_op(memory: MemoryBank, outputs: dict[str, torch.Tensor]) -> None:
    op = ID_TO_OP[int(outputs["op_logits"].argmax(dim=-1)[0].item())]
    slot_idx = int(outputs["slot_logits"].argmax(dim=-1)[0].item())
    if op == MemoryOpName.NOOP:
        return
    if op == MemoryOpName.DELETE:
        memory.apply_delete(0, slot_idx)
        return
    vector = outputs["value_vector"][0].detach()
    if op == MemoryOpName.UPDATE and bool(memory.occupied[0, slot_idx].item()):
        memory.apply_update(0, slot_idx, vector, gate=float(outputs["write_gate"][0, 0].item()))
        return
    memory.apply_write(
        0,
        slot_idx,
        vector,
        type_id=int(outputs["type_logits"].argmax(dim=-1)[0].item()),
        scope_id=int(outputs["scope_logits"].argmax(dim=-1)[0].item()),
        privacy_id=int(outputs["privacy_logits"].argmax(dim=-1)[0].item()),
        authority_id=int(outputs["authority_logits"].argmax(dim=-1)[0].item()),
        strength=float(outputs["write_gate"][0, 0].item()),
    )


def evaluate_sequential_neural(
    checkpoint_path: str,
    traces: list[TaskTrace],
    device: str,
    *,
    memory_mode: str,
) -> EvalResult | None:
    loaded = load_model_and_dataset(checkpoint_path, traces, device)
    if loaded is None:
        return None
    model, dataset, config = loaded
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    memory_required_total = 0
    memory_required_correct = 0
    predictions: list[tuple[str, ActionLabel, ActionLabel]] = []
    write_totals: Counter[str] = Counter()
    write_correct: Counter[str] = Counter()
    boundary_counts: Counter[str] = Counter()
    read_target_mass_sum = 0.0
    read_target_mass_count = 0
    with torch.no_grad():
        for trace in traces:
            memory = MemoryBank(
                num_slots=config["num_memory_slots"],
                d_model=config["d_model"],
                device=device,
            )
            memory.reset(1)
            op_index = 0
            for event in trace.events[:-1]:
                scope_id = torch.tensor(
                    [dataset.scope_vocab.get(event.scope or trace.current_scope, 0)],
                    dtype=torch.long,
                    device=device,
                )
                event_tokens, event_lengths = encode_event_tensor(
                    dataset, event, trace.current_scope, device
                )
                outputs = model(event_tokens, memory, scope_id, event_lengths)
                target_op = (
                    trace.expected_memory_ops[op_index]
                    if op_index < len(trace.expected_memory_ops)
                    else None
                )
                target_is_write = (
                    target_op is not None
                    and (event.scope or trace.current_scope) == target_op.scope
                )
                pred_op_id = int(outputs["op_logits"].argmax(dim=-1)[0].item())
                update_boundary_counts(
                    pred_is_write=ID_TO_OP[pred_op_id] != MemoryOpName.NOOP,
                    target_is_write=target_is_write,
                    counts=boundary_counts,
                )
                write_totals["op"] += 1
                write_correct["op"] += int(
                    pred_op_id
                    == (
                        OP_TO_ID[target_op.op]
                        if target_is_write and target_op is not None
                        else OP_TO_ID[MemoryOpName.NOOP]
                    )
                )
                if target_is_write and target_op is not None:
                    write_targets = {
                        "slot": ("slot_logits", min(op_index, memory.num_slots - 1)),
                        "type": ("type_logits", TYPE_TO_ID[target_op.type]),
                        "scope": ("scope_logits", dataset.scope_vocab.get(target_op.scope, 0)),
                        "privacy": ("privacy_logits", PRIVACY_TO_ID[target_op.privacy]),
                        "authority": ("authority_logits", AUTHORITY_TO_ID[target_op.authority]),
                    }
                    for name, (logit_key, target_id) in write_targets.items():
                        write_totals[name] += 1
                        pred_id = int(outputs[logit_key].argmax(dim=-1)[0].item())
                        write_correct[name] += int(pred_id == target_id)
                    if memory_mode == "teacher_forced":
                        apply_teacher_op(memory, model, dataset, trace, op_index, device)
                    elif memory_mode == "predicted":
                        apply_predicted_op(memory, outputs)
                    op_index += 1
                elif memory_mode == "predicted":
                    apply_predicted_op(memory, outputs)

            final_tokens, final_lengths = encode_event_tensor(
                dataset, trace.events[-1], trace.current_scope, device
            )
            current_scope = torch.tensor(
                [dataset.scope_vocab.get(trace.current_scope, 0)], dtype=torch.long, device=device
            )
            outputs = model(final_tokens, memory, current_scope, final_lengths)
            slot_idx = min(relevant_slot_index(trace), memory.num_slots - 1)
            read_target_mass_sum += float(
                outputs["read_attention"][0, int(outputs["boundary_indices"][0].item()), slot_idx]
                .cpu()
                .item()
            )
            read_target_mass_count += 1
            pred = ID_TO_ACTION[int(outputs["action_logits"].argmax(dim=-1)[0].item())]
            target = trace.expected_final_action
            totals[trace.task_family] += 1
            correct[trace.task_family] += int(pred == target)
            if is_memory_required_trace(trace):
                memory_required_total += 1
                memory_required_correct += int(pred == target)
            predictions.append((trace.task_family, pred, target))
    return EvalResult(
        totals=totals,
        correct=correct,
        memory_required_total=memory_required_total,
        memory_required_correct=memory_required_correct,
        failures=failure_counts(predictions),
        write_accuracies={
            name: write_correct[name] / write_totals[name]
            for name in sorted(write_totals)
            if write_totals[name]
        },
        boundary_tp=boundary_counts["tp"],
        boundary_fp=boundary_counts["fp"],
        boundary_fn=boundary_counts["fn"],
        boundary_tn=boundary_counts["tn"],
        read_target_mass_sum=read_target_mass_sum,
        read_target_mass_count=read_target_mass_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/latest.pt")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    traces = read_jsonl(args.traces)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (
        symbolic_totals,
        symbolic_correct,
        symbolic_memory_total,
        symbolic_memory_correct,
        symbolic_failures,
    ) = evaluate_symbolic(traces)
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
    sequential_tf_result = evaluate_sequential_neural(
        args.checkpoint,
        traces,
        device,
        memory_mode="teacher_forced",
    )
    sequential_predicted_result = evaluate_sequential_neural(
        args.checkpoint,
        traces,
        device,
        memory_mode="predicted",
    )

    table = Table(title=f"HotBob evaluation ({args.checkpoint})")
    table.add_column("Task family")
    table.add_column("Symbolic accuracy")
    table.add_column("Context-only accuracy")
    table.add_column("Neural TF memory accuracy")
    table.add_column("Predicted-write accuracy")
    table.add_column("Sequential TF accuracy")
    table.add_column("Sequential predicted accuracy")
    all_families = sorted(symbolic_totals)
    for family in all_families:
        table.add_row(
            family,
            f"{symbolic_correct[family] / symbolic_totals[family]:.3f}",
            family_accuracy(context_result, family),
            family_accuracy(neural_result, family),
            family_accuracy(predicted_result, family),
            family_accuracy(sequential_tf_result, family),
            family_accuracy(sequential_predicted_result, family),
        )
    context_failures = (
        context_result.failures if context_result is not None else ("n/a", "n/a", "n/a")
    )
    neural_failures = neural_result.failures if neural_result is not None else ("n/a", "n/a", "n/a")
    predicted_failures = (
        predicted_result.failures if predicted_result is not None else ("n/a", "n/a", "n/a")
    )
    sequential_tf_failures = (
        sequential_tf_result.failures if sequential_tf_result is not None else ("n/a", "n/a", "n/a")
    )
    sequential_predicted_failures = (
        sequential_predicted_result.failures
        if sequential_predicted_result is not None
        else ("n/a", "n/a", "n/a")
    )
    table.add_row(
        "secret leak failures",
        str(symbolic_failures[0]),
        str(context_failures[0]),
        str(neural_failures[0]),
        str(predicted_failures[0]),
        str(sequential_tf_failures[0]),
        str(sequential_predicted_failures[0]),
    )
    table.add_row(
        "wrong-scope retrieval failures",
        str(symbolic_failures[1]),
        str(context_failures[1]),
        str(neural_failures[1]),
        str(predicted_failures[1]),
        str(sequential_tf_failures[1]),
        str(sequential_predicted_failures[1]),
    )
    table.add_row(
        "expiry failures",
        str(symbolic_failures[2]),
        str(context_failures[2]),
        str(neural_failures[2]),
        str(predicted_failures[2]),
        str(sequential_tf_failures[2]),
        str(sequential_predicted_failures[2]),
    )
    table.add_row(
        "memory-required aggregate",
        f"{symbolic_memory_correct / symbolic_memory_total:.3f}",
        memory_required_accuracy(context_result),
        memory_required_accuracy(neural_result),
        memory_required_accuracy(predicted_result),
        memory_required_accuracy(sequential_tf_result),
        memory_required_accuracy(sequential_predicted_result),
    )
    memory_table = Table(title="Memory write metrics")
    memory_table.add_column("Mode")
    memory_table.add_column("Head")
    memory_table.add_column("Accuracy")
    for name, result in [
        ("context-only prewrite", context_result),
        ("teacher-forced prewrite", neural_result),
        ("predicted-write prewrite", predicted_result),
        ("sequential teacher-forced", sequential_tf_result),
        ("sequential predicted", sequential_predicted_result),
    ]:
        if result is None:
            memory_table.add_row(name, "all", "n/a")
        else:
            for head, value in result.write_accuracies.items():
                memory_table.add_row(name, head, f"{value:.3f}")
    boundary_table = Table(title="Boundary write decision metrics")
    boundary_table.add_column("Mode")
    boundary_table.add_column("Precision")
    boundary_table.add_column("Recall")
    boundary_table.add_column("F1")
    for name, result in [
        ("context-only prewrite", context_result),
        ("teacher-forced prewrite", neural_result),
        ("predicted-write prewrite", predicted_result),
        ("sequential teacher-forced", sequential_tf_result),
        ("sequential predicted", sequential_predicted_result),
    ]:
        precision, recall, f1 = boundary_scores(result)
        boundary_table.add_row(name, precision, recall, f1)
    retrieval_table = Table(title="Memory retrieval metrics")
    retrieval_table.add_column("Mode")
    retrieval_table.add_column("Target-slot read mass")
    for name, result in [
        ("context-only prewrite", context_result),
        ("teacher-forced prewrite", neural_result),
        ("predicted-write prewrite", predicted_result),
        ("sequential teacher-forced", sequential_tf_result),
        ("sequential predicted", sequential_predicted_result),
    ]:
        retrieval_table.add_row(name, read_target_mass(result))
    Console().print(table)
    Console().print(memory_table)
    Console().print(boundary_table)
    Console().print(retrieval_table)


if __name__ == "__main__":
    main()
