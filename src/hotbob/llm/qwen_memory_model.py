from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from hotbob.llm.dataset import LLMTrace, build_llm_tool_name_vocab
from hotbob.llm.env import load_dotenv
from hotbob.llm.memory_adapter import (
    LQwenMemoryHeads,
    MemoryStateMode,
    apply_teacher_memory_op,
    scope_id,
)
from hotbob.model.memory_bank import MemoryBank
from hotbob.training.dataset import (
    AUTHORITY_TO_ID,
    OP_TO_ID,
    PRIVACY_TO_ID,
    TYPE_TO_ID,
    build_scope_vocab,
    structured_targets_from_payload,
)
from hotbob.types import MemoryOpName, MemoryPrivacy, TaskTrace


@dataclass
class QwenMemoryConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    memory_prefix_len: int = 4
    num_memory_slots: int = 32
    freeze_base: bool = True
    integration_mode: str = "prefix"
    memory_state_mode: str = MemoryStateMode.SHARED.value
    correction_rank: int = 16
    attention_patch_layers: str = "all"
    num_value_classes: int = 64
    structured_loss_weight: float = 0.2
    tool_name_vocab: dict[str, int] = field(default_factory=dict)
    num_route_steps: int = 8
    device: str = "auto"
    torch_dtype: str = "auto"
    scope_vocab: dict[str, int] = field(default_factory=lambda: {"default": 1})


class QwenMemoryModel(nn.Module):
    def __init__(
        self,
        config: QwenMemoryConfig | None = None,
        *,
        base_model: nn.Module | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        self.config_obj = config or QwenMemoryConfig()
        self.tokenizer = tokenizer
        self.base_model = base_model if base_model is not None else self._load_qwen_model()
        if self.tokenizer is None:
            self.tokenizer = self._load_qwen_tokenizer()
        self.hidden_size = self._hidden_size()
        self.memory = MemoryBank(
            self.config_obj.num_memory_slots,
            self.hidden_size,
            device=self._device(),
        )
        self.memory_heads = LQwenMemoryHeads(
            self.hidden_size,
            self.config_obj.num_memory_slots,
            max(self.config_obj.scope_vocab.values(), default=0) + 1,
            self.config_obj.num_value_classes,
            max(self.config_obj.tool_name_vocab.values(), default=0) + 1,
            self.config_obj.num_route_steps,
            self.config_obj.memory_prefix_len,
            self.config_obj.correction_rank,
            self.config_obj.memory_state_mode,
        )
        self.memory_heads.to(self._device())
        self._q_patch_modules, self._o_patch_modules = self._discover_attention_patch_targets()
        self._has_q_patch_modules = bool(self._q_patch_modules)
        self._has_o_patch_modules = bool(self._o_patch_modules)
        if self.config_obj.freeze_base:
            self.freeze_base()

    @classmethod
    def from_pretrained(cls, config: QwenMemoryConfig | None = None) -> QwenMemoryModel:
        return cls(config)

    def _load_qwen_model(self) -> nn.Module:
        load_dotenv()
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("Install transformers to load Qwen models.") from exc
        kwargs: dict[str, Any] = {}
        if self.config_obj.torch_dtype != "auto":
            kwargs["dtype"] = getattr(torch, self.config_obj.torch_dtype)
        elif torch.cuda.is_available():
            kwargs["dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(self.config_obj.model_name, **kwargs)
        if self.config_obj.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config_obj.device
        return model.to(device)

    def _load_qwen_tokenizer(self) -> Any:
        load_dotenv()
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install transformers to load Qwen tokenizers.") from exc
        return AutoTokenizer.from_pretrained(self.config_obj.model_name)

    def _hidden_size(self) -> int:
        cfg = getattr(self.base_model, "config", None)
        if cfg is not None:
            for attr in ("hidden_size", "n_embd", "d_model"):
                value = getattr(cfg, attr, None)
                if value is not None:
                    return int(value)
        return int(self.base_model.get_input_embeddings().embedding_dim)

    def _device(self) -> torch.device:
        try:
            return next(self.base_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def freeze_base(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False

    def reset_memory(self, batch_size: int = 1) -> None:
        self.memory = MemoryBank(
            self.config_obj.num_memory_slots,
            self.hidden_size,
            device=self._device(),
        )
        self.memory.reset(batch_size)

    def _tokenize(self, text: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(text, return_tensors="pt")
        return {key: value.to(self._device()) for key, value in encoded.items()}

    def _embed_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self._tokenize(text)
        embeds = self.base_model.get_input_embeddings()(encoded["input_ids"])
        mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))
        return embeds, mask

    def _mode_flags(self) -> tuple[bool, bool, bool]:
        mode = self.config_obj.integration_mode
        if mode == "prefix":
            return True, False, False
        if mode == "attention_q":
            return False, True, False
        if mode == "attention_o":
            return False, False, True
        if mode == "attention_qo":
            return False, True, True
        raise ValueError(f"Unknown integration_mode: {mode}")

    def _scope_tensor(
        self,
        current_scope: str,
        device: torch.device,
        *,
        batch_size: int = 1,
    ) -> torch.Tensor:
        return torch.tensor(
            [scope_id(current_scope, self.config_obj.scope_vocab)] * batch_size,
            dtype=torch.long,
            device=device,
        )

    @contextmanager
    def _repeated_memory_batch(self, batch_size: int) -> Iterator[None]:
        if batch_size == self.memory.vectors.shape[0]:
            yield
            return
        old_memory = self.memory
        self.memory = self.memory.repeat_batch(batch_size)
        try:
            yield
        finally:
            self.memory = old_memory

    def _output_embeddings(self) -> nn.Module | None:
        if hasattr(self.base_model, "get_output_embeddings"):
            output_embeddings = self.base_model.get_output_embeddings()
            if output_embeddings is not None:
                return output_embeddings
        return getattr(self.base_model, "lm_head", None)

    def _logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        output_embeddings = self._output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("Base model does not expose output embeddings for o-correction.")
        return output_embeddings(hidden)

    def _discover_attention_patch_targets(self) -> tuple[list[nn.Module], list[nn.Module]]:
        q_modules: list[nn.Module] = []
        o_modules: list[nn.Module] = []
        for name, module in self.base_model.named_modules():
            if name.endswith("self_attn.q_proj"):
                q_modules.append(module)
            elif name.endswith("self_attn.o_proj"):
                o_modules.append(module)
        q_modules = self._select_attention_layers(q_modules)
        o_modules = self._select_attention_layers(o_modules)
        return q_modules, o_modules

    def _attention_patch_targets(self) -> tuple[list[nn.Module], list[nn.Module]]:
        return self._q_patch_modules, self._o_patch_modules

    def _select_attention_layers(self, modules: list[nn.Module]) -> list[nn.Module]:
        spec = self.config_obj.attention_patch_layers
        if spec == "all" or not modules:
            return modules
        if spec.startswith("last"):
            count = int(spec.removeprefix("last"))
            return modules[-count:]
        indices = {int(part) for part in spec.split(",") if part.strip()}
        return [module for idx, module in enumerate(modules) if idx in indices]

    def _has_attention_patch_targets(self, *, apply_q: bool, apply_o: bool) -> bool:
        return (not apply_q or self._has_q_patch_modules) and (
            not apply_o or self._has_o_patch_modules
        )

    @contextmanager
    def _memory_attention_patch(
        self,
        current_scope: str,
        *,
        apply_q: bool,
        apply_o: bool,
        use_memory: bool,
    ) -> Iterator[None]:
        if not use_memory or not (apply_q or apply_o):
            yield
            return
        q_modules, o_modules = self._attention_patch_targets()
        handles = []
        scopes_by_device: dict[torch.device, torch.Tensor] = {}

        def scopes_for(device: torch.device) -> torch.Tensor:
            if device not in scopes_by_device:
                scopes_by_device[device] = self._scope_tensor(
                    current_scope,
                    device,
                    batch_size=self.memory.vectors.shape[0],
                )
            return scopes_by_device[device]

        def correction_hook(
            correction_side: str,
        ) -> Callable[[nn.Module, tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
            def hook(
                _module: nn.Module,
                args: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, ...]:
                if not args:
                    return args
                hidden = args[0]
                if not torch.is_tensor(hidden):
                    return args
                corrected, _, _ = self.memory_heads.correction(
                    hidden.to(dtype=next(self.memory_heads.parameters()).dtype),
                    self.memory,
                    scopes_for(hidden.device),
                    apply_q=correction_side == "q",
                    apply_o=correction_side == "o",
                )
                return (corrected.to(dtype=hidden.dtype), *args[1:])

            return hook

        if apply_q:
            handles.extend(
                module.register_forward_pre_hook(correction_hook("q")) for module in q_modules
            )
        if apply_o:
            handles.extend(
                module.register_forward_pre_hook(correction_hook("o")) for module in o_modules
            )
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _base_logits_from_embeds(
        self,
        embeds: torch.Tensor,
        mask: torch.Tensor,
        *,
        current_scope: str,
        use_memory: bool,
        prefix_prompt_embeds: torch.Tensor | None = None,
        prefix_prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_prefix, apply_q, apply_o = self._mode_flags()
        if use_memory and use_prefix:
            prefix, _ = self.memory_prefix_for_prompt_embeds(
                prefix_prompt_embeds if prefix_prompt_embeds is not None else embeds,
                prefix_prompt_mask if prefix_prompt_mask is not None else mask,
                current_scope,
            )
            prefix = prefix.to(dtype=embeds.dtype)
            embeds = torch.cat([prefix, embeds], dim=1)
            prefix_mask = torch.ones(
                (mask.shape[0], prefix.shape[1]), dtype=mask.dtype, device=mask.device
            )
            mask = torch.cat([prefix_mask, mask], dim=1)
        elif use_memory and apply_q and not self._has_attention_patch_targets(
            apply_q=apply_q, apply_o=apply_o
        ):
            scopes = self._scope_tensor(
                current_scope, embeds.device, batch_size=embeds.shape[0]
            )
            corrected, _, _ = self.memory_heads.correction(
                embeds.to(dtype=next(self.memory_heads.parameters()).dtype),
                self.memory,
                scopes,
                apply_q=True,
                apply_o=False,
            )
            embeds = corrected.to(dtype=embeds.dtype)
        use_internal_patch = use_memory and self._has_attention_patch_targets(
            apply_q=apply_q,
            apply_o=apply_o,
        )
        with self._memory_attention_patch(
            current_scope,
            apply_q=apply_q,
            apply_o=apply_o,
            use_memory=use_internal_patch,
        ):
            outputs = self.base_model(
                inputs_embeds=embeds,
                attention_mask=mask,
                output_hidden_states=bool(use_memory and apply_o and not use_internal_patch),
            )
        if use_internal_patch or not (use_memory and apply_o):
            return outputs.logits
        hidden = outputs.hidden_states[-1]
        scopes = self._scope_tensor(current_scope, hidden.device, batch_size=hidden.shape[0])
        corrected, _, _ = self.memory_heads.correction(
            hidden.to(dtype=next(self.memory_heads.parameters()).dtype),
            self.memory,
            scopes,
            apply_q=False,
            apply_o=True,
        )
        return self._logits_from_hidden(corrected.to(dtype=hidden.dtype))

    def encode_event(self, role: str, content: str, scope: str | None = None) -> torch.Tensor:
        text = f"{role}: {content}"
        if scope:
            text = f"[scope={scope}] {text}"
        embeds, mask = self._embed_text(text)
        weights = mask.unsqueeze(-1).to(embeds.dtype)
        return (embeds * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def forward_event(
        self,
        role: str,
        content: str,
        *,
        scope: str | None = None,
        apply_predicted: bool = False,
    ) -> dict[str, torch.Tensor]:
        head_param = next(self.memory_heads.parameters())
        boundary_hidden = self.encode_event(role, content, scope).to(dtype=head_param.dtype)
        outputs = self.memory_heads.writer(boundary_hidden)
        if apply_predicted:
            self.apply_memory_op(outputs)
        return outputs

    def apply_memory_op(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        batch_idx: int = 0,
    ) -> None:
        slot = int(outputs["slot_logits"].argmax(dim=-1)[batch_idx].item())
        vector = self.memory_heads.value_projector(outputs["value_vector"])[batch_idx].detach()
        type_id = int(outputs["type_logits"].argmax(dim=-1)[batch_idx].item())
        scope = int(outputs["scope_logits"].argmax(dim=-1)[batch_idx].item())
        privacy = int(outputs["privacy_logits"].argmax(dim=-1)[batch_idx].item())
        authority = int(outputs["authority_logits"].argmax(dim=-1)[batch_idx].item())
        self.memory.apply_write(
            batch_idx,
            slot,
            vector,
            type_id=type_id,
            scope_id=scope,
            privacy_id=privacy,
            authority_id=authority,
        )

    def apply_teacher_trace_memory(self, trace: LLMTrace) -> None:
        self.reset_memory(1)
        for idx, op in enumerate(trace.expected_memory_ops):
            value_vector = self.encode_event("MEMORY_VALUE", op.value, op.scope)[0].float().detach()
            apply_teacher_memory_op(
                self.memory,
                op,
                value_vector,
                slot_idx=idx,
                scope_vocab=self.config_obj.scope_vocab,
            )

    def memory_prefix_for_prompt(
        self,
        final_prompt: str,
        current_scope: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeds, mask = self._embed_text(final_prompt)
        return self.memory_prefix_for_prompt_embeds(embeds, mask, current_scope)

    def memory_prefix_for_prompt_embeds(
        self,
        embeds: torch.Tensor,
        mask: torch.Tensor,
        current_scope: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_param = next(self.memory_heads.parameters())
        weights = mask.unsqueeze(-1).to(embeds.dtype)
        query = (embeds * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        query = query.to(dtype=head_param.dtype)
        scopes = self._scope_tensor(current_scope, query.device, batch_size=query.shape[0])
        return self.memory_heads.prefix(query, self.memory, scopes)

    def _tokenize_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:
        try:
            encoded = self.tokenizer(texts, return_tensors="pt", padding=True)
            return {key: value.to(self._device()) for key, value in encoded.items()}
        except TypeError:
            rows = [self._tokenize(text) for text in texts]
            max_len = max(row["input_ids"].shape[1] for row in rows)
            input_ids = torch.zeros((len(rows), max_len), dtype=torch.long, device=self._device())
            attention_mask = torch.zeros_like(input_ids)
            for idx, row in enumerate(rows):
                length = row["input_ids"].shape[1]
                input_ids[idx, :length] = row["input_ids"][0]
                attention_mask[idx, :length] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _candidate_texts(self, final_prompt: str, candidates: list[str]) -> list[str]:
        return [f"{final_prompt} {candidate}" for candidate in candidates]

    def _candidate_token_spans(
        self,
        final_prompt: str,
        candidates: list[str],
        encoded: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        prompt_len = self._tokenize(final_prompt)["input_ids"].shape[1]
        input_ids = encoded["input_ids"]
        mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        token_positions = torch.arange(input_ids.shape[1] - 1, device=input_ids.device)
        return mask[:, 1:].bool() & (token_positions.unsqueeze(0) >= max(prompt_len - 1, 0))

    def score_final_candidates(
        self,
        final_prompt: str,
        candidates: list[str],
        *,
        current_scope: str,
        use_memory: bool = True,
    ) -> torch.Tensor:
        encoded = self._tokenize_texts(self._candidate_texts(final_prompt, candidates))
        input_ids = encoded["input_ids"]
        mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        embeds = self.base_model.get_input_embeddings()(input_ids)
        prefix_prompt_embeds = None
        prefix_prompt_mask = None
        if use_memory and self.config_obj.integration_mode == "prefix":
            prefix_prompt_embeds, prefix_prompt_mask = self._embed_text(final_prompt)
            prefix_prompt_embeds = prefix_prompt_embeds.repeat(input_ids.shape[0], 1, 1)
            prefix_prompt_mask = prefix_prompt_mask.repeat(input_ids.shape[0], 1)
        with self._repeated_memory_batch(input_ids.shape[0]):
            logits = self._base_logits_from_embeds(
                embeds,
                mask,
                current_scope=current_scope,
                use_memory=use_memory,
                prefix_prompt_embeds=prefix_prompt_embeds,
                prefix_prompt_mask=prefix_prompt_mask,
            )
        candidate_mask = self._candidate_token_spans(final_prompt, candidates, encoded)
        if logits.shape[1] != input_ids.shape[1]:
            prefix_len = logits.shape[1] - input_ids.shape[1]
            input_ids = torch.cat(
                [
                    torch.zeros(
                        (input_ids.shape[0], prefix_len),
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    ),
                    input_ids,
                ],
                dim=1,
            )
            candidate_mask = torch.cat(
                [
                    torch.zeros(
                        (candidate_mask.shape[0], prefix_len),
                        dtype=torch.bool,
                        device=candidate_mask.device,
                    ),
                    candidate_mask,
                ],
                dim=1,
            )
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        losses = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="none",
        ).view_as(shift_labels)
        scores = -(losses * candidate_mask).sum(dim=1) / candidate_mask.sum(dim=1).clamp_min(1)
        return scores

    def choose_final_candidate(
        self,
        final_prompt: str,
        candidates: list[str],
        *,
        current_scope: str,
        use_memory: bool = True,
    ) -> str:
        scores = self.score_final_candidates(
            final_prompt,
            candidates,
            current_scope=current_scope,
            use_memory=use_memory,
        )
        return candidates[int(scores.argmax(dim=0).item())]

    def final_prompt_logits(
        self,
        final_prompt: str,
        *,
        current_scope: str,
        use_memory: bool,
    ) -> torch.Tensor:
        embeds, mask = self._embed_text(final_prompt)
        return self._base_logits_from_embeds(
            embeds,
            mask,
            current_scope=current_scope,
            use_memory=use_memory,
        )

    def teacher_forced_lm_loss(self, trace: LLMTrace) -> torch.Tensor:
        self.apply_teacher_trace_memory(trace)
        prompt_encoded = self._tokenize(trace.final_prompt)
        full_encoded = self._tokenize(f"{trace.final_prompt} {trace.target_text}")
        input_ids = full_encoded["input_ids"]
        mask = full_encoded.get("attention_mask", torch.ones_like(input_ids))
        embeds = self.base_model.get_input_embeddings()(input_ids)
        labels = input_ids.clone()
        prompt_len = prompt_encoded["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        if self.config_obj.integration_mode == "prefix":
            prefix, _ = self.memory_prefix_for_prompt(trace.final_prompt, trace.current_scope)
            prefix = prefix.to(dtype=embeds.dtype)
            embeds = torch.cat([prefix, embeds], dim=1)
            prefix_mask = torch.ones(
                (mask.shape[0], prefix.shape[1]), dtype=mask.dtype, device=mask.device
            )
            mask = torch.cat([prefix_mask, mask], dim=1)
            labels = torch.cat(
                [
                    torch.full(
                        (labels.shape[0], prefix.shape[1]),
                        -100,
                        dtype=labels.dtype,
                        device=labels.device,
                    ),
                    labels,
                ],
                dim=1,
            )
            logits = self.base_model(inputs_embeds=embeds, attention_mask=mask).logits
        else:
            logits = self._base_logits_from_embeds(
                embeds,
                mask,
                current_scope=trace.current_scope,
                use_memory=True,
            )
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def write_supervision_loss(self, trace: LLMTrace) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        op_index = 0
        for event in trace.events:
            outputs = self.forward_event(event.role, event.content, scope=event.scope)
            next_op = (
                trace.expected_memory_ops[op_index]
                if op_index < len(trace.expected_memory_ops)
                else None
            )
            target_op = (
                next_op
                if next_op is not None and (event.scope or trace.current_scope) == next_op.scope
                else None
            )
            target_op_id = (
                OP_TO_ID[target_op.op] if target_op is not None else OP_TO_ID[MemoryOpName.NOOP]
            )
            target = torch.tensor([target_op_id], dtype=torch.long, device=self._device())
            losses.append(nn.functional.cross_entropy(outputs["op_logits"], target))
            if target_op is None:
                continue
            target_type = torch.tensor(
                [TYPE_TO_ID[target_op.type]], dtype=torch.long, device=self._device()
            )
            target_scope = torch.tensor(
                [scope_id(target_op.scope, self.config_obj.scope_vocab)],
                dtype=torch.long,
                device=self._device(),
            )
            target_privacy = torch.tensor(
                [PRIVACY_TO_ID[target_op.privacy]], dtype=torch.long, device=self._device()
            )
            target_authority = torch.tensor(
                [AUTHORITY_TO_ID[target_op.authority]], dtype=torch.long, device=self._device()
            )
            target_slot = torch.tensor(
                [min(op_index, self.config_obj.num_memory_slots - 1)],
                dtype=torch.long,
                device=self._device(),
            )
            losses.extend(
                [
                    nn.functional.cross_entropy(outputs["type_logits"], target_type),
                    nn.functional.cross_entropy(outputs["scope_logits"], target_scope),
                    nn.functional.cross_entropy(outputs["privacy_logits"], target_privacy),
                    nn.functional.cross_entropy(outputs["authority_logits"], target_authority),
                    nn.functional.cross_entropy(outputs["slot_logits"], target_slot),
                ]
            )
            predicted_value = self.memory_heads.value_projector(outputs["value_vector"])[0]
            target_value = self.encode_event(
                "MEMORY_VALUE", target_op.value, target_op.scope
            )[0].float()
            losses.append(nn.functional.mse_loss(predicted_value, target_value))
            structured = structured_targets_from_payload(
                target_op.payload, self.config_obj.tool_name_vocab
            )
            structured_losses: list[torch.Tensor] = []
            structured_specs = [
                ("payload_kind_logits", "payload_kind_id", "has_payload"),
                (
                    "payload_default_action_logits",
                    "default_action_id",
                    "has_default_action",
                ),
                ("payload_trigger_logits", "trigger_id", "has_trigger"),
                ("payload_exception_logits", "exception_id", "exception_id"),
                (
                    "payload_expiry_policy_logits",
                    "expiry_policy_id",
                    "has_expiry_policy",
                ),
                (
                    "payload_authority_level_logits",
                    "authority_level_id",
                    "has_authority_level",
                ),
                ("payload_tool_name_logits", "tool_name_id", "has_tool_name"),
                ("payload_route_step_logits", "route_step_id", "has_route_step"),
            ]
            for logits_key, target_key, mask_key in structured_specs:
                target_id = int(structured[target_key])
                has_target = (
                    bool(structured[mask_key])
                    if isinstance(structured.get(mask_key), bool)
                    else target_id > 0
                )
                if not has_target:
                    continue
                logits = outputs[logits_key]
                target_id = min(target_id, logits.shape[-1] - 1)
                target_tensor = torch.tensor(
                    [target_id], dtype=torch.long, device=self._device()
                )
                structured_losses.append(nn.functional.cross_entropy(logits, target_tensor))
            if structured_losses:
                losses.append(
                    self.config_obj.structured_loss_weight
                    * torch.stack(structured_losses).mean()
                )
            op_index += 1
        if not losses:
            return torch.zeros((), device=self._device(), requires_grad=True)
        return torch.stack(losses).mean()

    def generate_final(
        self,
        final_prompt: str,
        *,
        current_scope: str,
        use_memory: bool = True,
        max_new_tokens: int = 16,
    ) -> str:
        use_prefix, apply_q, apply_o = self._mode_flags()
        embeds, mask = self._embed_text(final_prompt)
        if use_memory and use_prefix:
            prefix, _ = self.memory_prefix_for_prompt(final_prompt, current_scope)
            prefix = prefix.to(dtype=embeds.dtype)
            embeds = torch.cat([prefix, embeds], dim=1)
            mask = torch.cat(
                [
                    torch.ones(
                        (mask.shape[0], prefix.shape[1]),
                        dtype=mask.dtype,
                        device=mask.device,
                    ),
                    mask,
                ],
                dim=1,
            )
        use_internal_patch = use_memory and self._has_attention_patch_targets(
            apply_q=apply_q,
            apply_o=apply_o,
        )
        if hasattr(self.base_model, "generate") and (
            not (use_memory and (apply_q or apply_o)) or use_internal_patch
        ):
            with self._memory_attention_patch(
                current_scope,
                apply_q=apply_q,
                apply_o=apply_o,
                use_memory=use_internal_patch,
            ):
                ids = self.base_model.generate(
                    inputs_embeds=embeds,
                    attention_mask=mask,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                )
            return self.tokenizer.decode(ids[0], skip_special_tokens=True)
        encoded = self._tokenize(final_prompt)
        input_ids = encoded["input_ids"]
        generated: list[int] = []
        for _ in range(max_new_tokens):
            mask = torch.ones_like(input_ids)
            embeds = self.base_model.get_input_embeddings()(input_ids)
            logits = self._base_logits_from_embeds(
                embeds,
                mask,
                current_scope=current_scope,
                use_memory=use_memory,
            )
            next_id = int(logits[:, -1].argmax(dim=-1)[0].item())
            generated.append(next_id)
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.tensor([[next_id]], dtype=input_ids.dtype, device=input_ids.device),
                ],
                dim=1,
            )
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def debug_dump(self, *, include_private_values: bool = False) -> list[dict[str, Any]]:
        rows = self.memory.debug_dump()
        for row in rows:
            if (
                not include_private_values
                and int(row.get("privacy_id", -1))
                == list(MemoryPrivacy).index(MemoryPrivacy.HIDDEN_FROM_USER)
            ):
                row["value"] = "<redacted>"
        return rows

    @staticmethod
    def config_from_task_traces(traces: list[TaskTrace], **kwargs: Any) -> QwenMemoryConfig:
        llm_like = [
            LLMTrace(
                events=[],
                final_prompt="",
                target_text="",
                target_action="",
                current_scope=trace.current_scope,
                expected_memory_ops=trace.expected_memory_ops,
            )
            for trace in traces
        ]
        return QwenMemoryConfig(
            scope_vocab=build_scope_vocab(traces),
            tool_name_vocab=build_llm_tool_name_vocab(llm_like),
            **kwargs,
        )

    def run_trace_teacher_forced(self, trace: LLMTrace) -> str:
        self.apply_teacher_trace_memory(trace)
        return self.generate_final(
            trace.final_prompt,
            current_scope=trace.current_scope,
            use_memory=True,
        )
