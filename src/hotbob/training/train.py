from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from rich.progress import track
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hotbob.data.traces import read_jsonl
from hotbob.model import MemoryBank
from hotbob.model.stateful_transformer import StatefulTransformer
from hotbob.training.dataset import (
    AUTHORITY_LEVEL_TO_ID,
    AUTHORITY_TO_ID,
    EXPIRY_POLICY_TO_ID,
    OP_TO_ID,
    PAYLOAD_KIND_TO_ID,
    POLICY_ACTION_TO_ID,
    POLICY_TRIGGER_TO_ID,
    PRIVACY_TO_ID,
    TYPE_TO_ID,
    TraceDataset,
    collate_traces,
)
from hotbob.training.losses import contrastive_retrieval_loss
from hotbob.training.memory_teacher import build_teacher_forced_memory, mean_value_embedding
from hotbob.types import ActionLabel, MemoryOpName

STRUCTURED_EVENT_TARGETS = [
    ("payload_kind_logits", "event_payload_kind_ids", "event_has_payload"),
    ("payload_default_action_logits", "event_default_action_ids", "event_has_default_action"),
    ("payload_trigger_logits", "event_trigger_ids", "event_has_trigger"),
    ("payload_exception_logits", "event_exception_ids", "event_exception_ids"),
    ("payload_expiry_policy_logits", "event_expiry_policy_ids", "event_has_expiry_policy"),
    ("payload_authority_level_logits", "event_authority_level_ids", "event_has_authority_level"),
    (
        "payload_winning_authority_level_logits",
        "event_winning_authority_level_ids",
        "event_has_winning_authority_level",
    ),
    (
        "payload_losing_authority_level_logits",
        "event_losing_authority_level_ids",
        "event_has_losing_authority_level",
    ),
    ("payload_tool_name_logits", "event_tool_name_ids", "event_has_tool_name"),
    ("payload_route_step_logits", "event_route_step_ids", "event_has_route_step"),
]


def structured_event_loss(
    *,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    event_idx: int | None,
    mask: torch.Tensor,
    ce: nn.CrossEntropyLoss,
) -> torch.Tensor | None:
    losses: list[torch.Tensor] = []
    for logit_key, target_key, mask_key in STRUCTURED_EVENT_TARGETS:
        if event_idx is None:
            targets = batch[target_key].flatten(0, 1)
            if batch[mask_key].dtype == torch.bool:
                field_mask = batch[mask_key].flatten(0, 1)
            else:
                field_mask = targets > 0
        else:
            targets = batch[target_key][:, event_idx]
            if batch[mask_key].dtype == torch.bool:
                field_mask = batch[mask_key][:, event_idx]
            else:
                field_mask = targets > 0
        target_mask = mask & field_mask
        if not bool(target_mask.any().item()):
            continue
        logits = outputs[logit_key][target_mask]
        clamped_targets = targets[target_mask].clamp_max(logits.shape[-1] - 1)
        losses.append(ce(logits, clamped_targets))
    if not losses:
        return None
    return torch.stack(losses).mean()


def clone_memory_for_forward(memory: MemoryBank) -> MemoryBank:
    clone = MemoryBank(num_slots=memory.num_slots, d_model=memory.d_model, device=memory.device)
    clone.reset(memory.occupied.shape[0])
    clone.vectors = memory.vectors.clone()
    clone.occupied = memory.occupied.clone()
    clone.strength = memory.strength.clone()
    clone.type_ids = memory.type_ids.clone()
    clone.scope_ids = memory.scope_ids.clone()
    clone.privacy_ids = memory.privacy_ids.clone()
    clone.authority_ids = memory.authority_ids.clone()
    clone.payload_kind_ids = memory.payload_kind_ids.clone()
    clone.payload_default_action_ids = memory.payload_default_action_ids.clone()
    clone.payload_winning_authority_level_ids = (
        memory.payload_winning_authority_level_ids.clone()
    )
    clone.payload_losing_authority_level_ids = memory.payload_losing_authority_level_ids.clone()
    return clone


def apply_teacher_event_ops(
    *,
    memory: MemoryBank,
    model: StatefulTransformer,
    event_value_tokens: torch.Tensor,
    event_value_mask: torch.Tensor,
    op_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    type_ids: torch.Tensor,
    scope_ids: torch.Tensor,
    privacy_ids: torch.Tensor,
    authority_ids: torch.Tensor,
    payload_kind_ids: torch.Tensor,
    payload_default_action_ids: torch.Tensor,
    payload_winning_authority_level_ids: torch.Tensor,
    payload_losing_authority_level_ids: torch.Tensor,
    write_mask: torch.Tensor,
) -> None:
    if not bool(write_mask.any().item()):
        return
    value_vectors = mean_value_embedding(
        model.transformer.embed,
        event_value_tokens[write_mask],
        event_value_mask[write_mask],
    ).detach()
    write_rows = torch.nonzero(write_mask, as_tuple=False).flatten()
    for vector_idx, row_tensor in enumerate(write_rows):
        row = int(row_tensor.item())
        slot = int(slot_ids[row].item())
        op_id = int(op_ids[row].item())
        if op_id == OP_TO_ID[MemoryOpName.DELETE]:
            memory.apply_delete(row, slot)
        elif op_id == OP_TO_ID[MemoryOpName.UPDATE] and bool(memory.occupied[row, slot].item()):
            memory.apply_update(row, slot, value_vectors[vector_idx])
            memory.payload_kind_ids[row, slot] = int(payload_kind_ids[row].item())
            memory.payload_default_action_ids[row, slot] = int(
                payload_default_action_ids[row].item()
            )
            memory.payload_winning_authority_level_ids[row, slot] = int(
                payload_winning_authority_level_ids[row].item()
            )
            memory.payload_losing_authority_level_ids[row, slot] = int(
                payload_losing_authority_level_ids[row].item()
            )
        else:
            memory.apply_write(
                row,
                slot,
                value_vectors[vector_idx],
                type_id=int(type_ids[row].item()),
                scope_id=int(scope_ids[row].item()),
                privacy_id=int(privacy_ids[row].item()),
                authority_id=int(authority_ids[row].item()),
                payload_kind_id=int(payload_kind_ids[row].item()),
                payload_default_action_id=int(payload_default_action_ids[row].item()),
                payload_winning_authority_level_id=int(
                    payload_winning_authority_level_ids[row].item()
                ),
                payload_losing_authority_level_id=int(
                    payload_losing_authority_level_ids[row].item()
                ),
            )


def sequential_controller_loss(
    *,
    model: StatefulTransformer,
    batch: dict[str, torch.Tensor],
    num_memory_slots: int,
    d_model: int,
    device: str,
    ce: nn.CrossEntropyLoss,
    structured_loss_weight: float,
) -> torch.Tensor:
    memory = MemoryBank(num_slots=num_memory_slots, d_model=d_model, device=device)
    batch_size, max_events = batch["event_tokens"].shape[:2]
    memory.reset(batch_size)
    losses: list[torch.Tensor] = []
    for event_idx in range(max_events):
        event_mask = batch["event_mask"][:, event_idx]
        if not bool(event_mask.any().item()):
            continue
        outputs = model(
            batch["event_tokens"][:, event_idx],
            clone_memory_for_forward(memory),
            batch["event_scope_ids"][:, event_idx],
            batch["event_lengths"][:, event_idx].clamp_min(1),
        )
        losses.append(
            ce(
                outputs["op_logits"][event_mask],
                batch["event_op_ids"][:, event_idx][event_mask],
            )
        )
        write_mask = batch["event_has_write"][:, event_idx] & event_mask
        if bool(write_mask.any().item()):
            for logit_key, target_key in [
                ("slot_logits", "event_slot_ids"),
                ("type_logits", "event_type_ids"),
                ("scope_logits", "event_scope_ids"),
                ("privacy_logits", "event_privacy_ids"),
                ("authority_logits", "event_authority_ids"),
                ("value_class_logits", "event_value_class_ids"),
            ]:
                losses.append(
                    ce(
                        outputs[logit_key][write_mask],
                        batch[target_key][:, event_idx][write_mask],
                    )
                )
            structured_loss = structured_event_loss(
                outputs=outputs,
                batch=batch,
                event_idx=event_idx,
                mask=write_mask,
                ce=ce,
            )
            if structured_loss is not None:
                losses.append(structured_loss_weight * structured_loss)
        apply_teacher_event_ops(
            memory=memory,
            model=model,
            event_value_tokens=batch["event_value_tokens"][:, event_idx],
            event_value_mask=batch["event_value_mask"][:, event_idx],
            op_ids=batch["event_op_ids"][:, event_idx],
            slot_ids=batch["event_slot_ids"][:, event_idx],
            type_ids=batch["event_type_ids"][:, event_idx],
            scope_ids=batch["event_scope_ids"][:, event_idx],
            privacy_ids=batch["event_privacy_ids"][:, event_idx],
            authority_ids=batch["event_authority_ids"][:, event_idx],
            payload_kind_ids=batch["event_payload_kind_ids"][:, event_idx],
            payload_default_action_ids=batch["event_default_action_ids"][:, event_idx],
            payload_winning_authority_level_ids=batch[
                "event_winning_authority_level_ids"
            ][:, event_idx],
            payload_losing_authority_level_ids=batch[
                "event_losing_authority_level_ids"
            ][:, event_idx],
            write_mask=write_mask,
        )
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--retrieval-contrastive-weight", type=float, default=0.25)
    parser.add_argument("--structured-loss-weight", type=float, default=0.2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. Use cuda in cloud notebooks to fail fast if GPU Torch is unavailable.",
    )
    parser.add_argument(
        "--sequential-controller-loss",
        dest="sequential_controller_loss",
        action="store_true",
    )
    parser.add_argument(
        "--no-sequential-controller-loss",
        dest="sequential_controller_loss",
        action="store_false",
    )
    parser.add_argument("--sequential-predicted-warmup-steps", type=int, default=0)
    parser.set_defaults(sequential_controller_loss=True)
    args = parser.parse_args()
    traces = read_jsonl(args.traces)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested with --device cuda, but torch.cuda.is_available() is false. "
                "Check that the notebook runtime has a GPU and that uv installed a CUDA-enabled "
                "Torch wheel compatible with the runtime driver."
            )
        device = "cuda"
    else:
        device = "cpu"
    if device == "cuda":
        print(
            "training device=cuda "
            f"torch_cuda={torch.version.cuda} "
            f"gpu={torch.cuda.get_device_name(0)}"
        )
    else:
        print("training device=cpu")
    dataset = TraceDataset(traces)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_traces,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        persistent_workers=args.num_workers > 0,
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
        num_value_classes=max(dataset.value_vocab.values(), default=0) + 1,
        num_payload_kinds=max(PAYLOAD_KIND_TO_ID.values(), default=0) + 1,
        num_policy_actions=max(POLICY_ACTION_TO_ID.values(), default=0) + 1,
        num_policy_triggers=max(POLICY_TRIGGER_TO_ID.values(), default=0) + 1,
        num_expiry_policies=max(EXPIRY_POLICY_TO_ID.values(), default=0) + 1,
        num_authority_levels=max(AUTHORITY_LEVEL_TO_ID.values(), default=0) + 1,
        num_tool_names=max(dataset.tool_name_vocab.values(), default=0) + 1,
        num_route_steps=8,
        max_seq_len=dataset.max_seq_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    use_amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses: list[float] = []
    data_iter = iter(loader)
    train_started_at = time.perf_counter()

    for step in track(range(args.steps), description="training"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            memory = build_teacher_forced_memory(
                model_embed=model.transformer.embed,
                memory_value_tokens=batch["memory_value_tokens"],
                memory_value_mask=batch["memory_value_mask"],
                slot_ids=batch["slot_ids"],
                type_ids=batch["type_ids"],
                scope_ids=batch["scope_ids"],
                privacy_ids=batch["privacy_ids"],
                authority_ids=batch["authority_ids"],
                payload_kind_ids=batch["payload_kind_ids"],
                payload_default_action_ids=batch["default_action_ids"],
                payload_winning_authority_level_ids=batch["winning_authority_level_ids"],
                payload_losing_authority_level_ids=batch["losing_authority_level_ids"],
                num_memory_slots=num_memory_slots,
                d_model=d_model,
                device=device,
            )
            flat_event_tokens = batch["event_tokens"].flatten(0, 1)
            flat_event_lengths = batch["event_lengths"].flatten(0, 1)
            flat_event_mask = batch["event_mask"].flatten(0, 1)
            flat_event_scope_ids = batch["event_scope_ids"].flatten(0, 1)
            event_memory = MemoryBank(num_slots=num_memory_slots, d_model=d_model, device=device)
            event_memory.reset(flat_event_tokens.shape[0])
            event_outputs = model(
                flat_event_tokens,
                event_memory,
                flat_event_scope_ids,
                flat_event_lengths.clamp_min(1),
            )
            outputs = model(batch["tokens"], memory, batch["current_scope_ids"], batch["lengths"])
            loss = ce(outputs["action_logits"], batch["action_ids"])
            loss = loss + ce(
                event_outputs["op_logits"][flat_event_mask],
                batch["event_op_ids"].flatten(0, 1)[flat_event_mask],
            )
            flat_write_mask = batch["event_has_write"].flatten(0, 1) & flat_event_mask
            if bool(flat_write_mask.any().item()):
                loss = loss + ce(
                    event_outputs["slot_logits"][flat_write_mask],
                    batch["event_slot_ids"].flatten(0, 1)[flat_write_mask],
                )
                loss = loss + ce(
                    event_outputs["type_logits"][flat_write_mask],
                    batch["event_type_ids"].flatten(0, 1)[flat_write_mask],
                )
                loss = loss + ce(
                    event_outputs["scope_logits"][flat_write_mask],
                    batch["event_scope_ids"].flatten(0, 1)[flat_write_mask],
                )
                loss = loss + ce(
                    event_outputs["privacy_logits"][flat_write_mask],
                    batch["event_privacy_ids"].flatten(0, 1)[flat_write_mask],
                )
                loss = loss + ce(
                    event_outputs["authority_logits"][flat_write_mask],
                    batch["event_authority_ids"].flatten(0, 1)[flat_write_mask],
                )
                loss = loss + ce(
                    event_outputs["value_class_logits"][flat_write_mask],
                    batch["event_value_class_ids"].flatten(0, 1)[flat_write_mask],
                )
                structured_loss = structured_event_loss(
                    outputs=event_outputs,
                    batch=batch,
                    event_idx=None,
                    mask=flat_write_mask,
                    ce=ce,
                )
                if structured_loss is not None:
                    loss = loss + args.structured_loss_weight * structured_loss
            if bool(flat_write_mask.any().item()):
                flat_event_value_tokens = batch["event_value_tokens"].flatten(0, 1)
                flat_event_value_mask = batch["event_value_mask"].flatten(0, 1)
                target_value = mean_value_embedding(
                    model.transformer.embed,
                    flat_event_value_tokens[flat_write_mask],
                    flat_event_value_mask[flat_write_mask],
                ).detach()
                predicted_value = event_outputs["value_vector"][flat_write_mask]
                loss = loss + F.mse_loss(predicted_value, target_value)
                if predicted_value.shape[0] > 1:
                    pred_norm = F.normalize(predicted_value, dim=-1)
                    target_norm = F.normalize(target_value, dim=-1)
                    logits = pred_norm @ target_norm.T / 0.1
                    labels = torch.arange(predicted_value.shape[0], device=device)
                    loss = loss + F.cross_entropy(logits, labels)
            if args.sequential_controller_loss:
                loss = loss + sequential_controller_loss(
                    model=model,
                    batch=batch,
                    num_memory_slots=num_memory_slots,
                    d_model=d_model,
                    device=device,
                    ce=ce,
                    structured_loss_weight=args.structured_loss_weight,
                )
            read_attn = outputs["read_attention"][
                torch.arange(batch["tokens"].shape[0], device=device),
                outputs["boundary_indices"],
                batch["slot_ids"],
            ]
            loss = loss - 0.1 * torch.log(read_attn.clamp_min(1e-6)).mean()
            if batch["tokens"].shape[0] > 1 and args.retrieval_contrastive_weight > 0:
                target_value = mean_value_embedding(
                    model.transformer.embed,
                    batch["memory_value_tokens"],
                    batch["memory_value_mask"],
                ).detach()
                loss = loss + args.retrieval_contrastive_weight * contrastive_retrieval_loss(
                    outputs["memory_context"],
                    target_value,
                    target_value,
                )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu().item()))
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            elapsed = max(time.perf_counter() - train_started_at, 1e-9)
            steps_per_sec = (step + 1) / elapsed
            samples_per_sec = ((step + 1) * args.batch_size) / elapsed
            print(
                f"step={step + 1} loss={losses[-1]:.4f} "
                f"steps_per_sec={steps_per_sec:.2f} samples_per_sec={samples_per_sec:.1f}"
            )

    Path("runs").mkdir(exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "vocab": dataset.vocab.token_to_id,
        "scope_vocab": dataset.scope_vocab,
        "value_vocab": dataset.value_vocab,
        "tool_name_vocab": dataset.tool_name_vocab,
        "num_traces": len(traces),
        "steps": args.steps,
        "device": device,
        "losses": losses,
        "config": {
            "d_model": d_model,
            "num_memory_slots": num_memory_slots,
            "action_vocab_size": len(ActionLabel),
            "num_value_classes": max(dataset.value_vocab.values(), default=0) + 1,
            "num_payload_kinds": max(PAYLOAD_KIND_TO_ID.values(), default=0) + 1,
            "num_policy_actions": max(POLICY_ACTION_TO_ID.values(), default=0) + 1,
            "num_policy_triggers": max(POLICY_TRIGGER_TO_ID.values(), default=0) + 1,
            "num_expiry_policies": max(EXPIRY_POLICY_TO_ID.values(), default=0) + 1,
            "num_authority_levels": max(AUTHORITY_LEVEL_TO_ID.values(), default=0) + 1,
            "num_tool_names": max(dataset.tool_name_vocab.values(), default=0) + 1,
            "num_route_steps": 8,
            "structured_loss_weight": args.structured_loss_weight,
            "max_seq_len": dataset.max_seq_len,
            "action_readout_type": "fusion_mlp",
            "retrieval_contrastive_weight": args.retrieval_contrastive_weight,
            "sequential_controller_loss": args.sequential_controller_loss,
            "sequential_predicted_warmup_steps": args.sequential_predicted_warmup_steps,
        },
    }
    torch.save(checkpoint, "runs/latest.pt")
    print(f"saved checkpoint runs/latest.pt final_loss={losses[-1]:.4f} device={device}")


if __name__ == "__main__":
    main()
