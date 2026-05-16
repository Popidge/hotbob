from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from hotbob.llm.dataset import LLMTrace
from hotbob.llm.memory_adapter import LQwenMemoryHeads, apply_teacher_memory_op, scope_id
from hotbob.model.memory_bank import MemoryBank
from hotbob.training.dataset import build_scope_vocab
from hotbob.types import MemoryPrivacy, TaskTrace


@dataclass
class QwenMemoryConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    memory_prefix_len: int = 4
    num_memory_slots: int = 32
    freeze_base: bool = True
    integration_mode: str = "prefix"
    num_value_classes: int = 64
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
            self.config_obj.memory_prefix_len,
        )
        if self.config_obj.freeze_base:
            self.freeze_base()

    @classmethod
    def from_pretrained(cls, config: QwenMemoryConfig | None = None) -> QwenMemoryModel:
        return cls(config)

    def _load_qwen_model(self) -> nn.Module:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("Install transformers to load Qwen models.") from exc
        return AutoModelForCausalLM.from_pretrained(self.config_obj.model_name)

    def _load_qwen_tokenizer(self) -> Any:
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
        boundary_hidden = self.encode_event(role, content, scope)
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
            value_vector = self.encode_event("MEMORY_VALUE", op.value, op.scope)[0].detach()
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
        query = self.encode_event("USER", final_prompt)
        scopes = torch.tensor(
            [scope_id(current_scope, self.config_obj.scope_vocab)],
            dtype=torch.long,
            device=query.device,
        )
        return self.memory_heads.prefix(query, self.memory, scopes)

    def final_prompt_logits(
        self,
        final_prompt: str,
        *,
        current_scope: str,
        use_memory: bool,
    ) -> torch.Tensor:
        embeds, mask = self._embed_text(final_prompt)
        if use_memory:
            prefix, _ = self.memory_prefix_for_prompt(final_prompt, current_scope)
            embeds = torch.cat([prefix, embeds], dim=1)
            prefix_mask = torch.ones(
                (mask.shape[0], prefix.shape[1]), dtype=mask.dtype, device=mask.device
            )
            mask = torch.cat([prefix_mask, mask], dim=1)
        outputs = self.base_model(inputs_embeds=embeds, attention_mask=mask)
        return outputs.logits

    def teacher_forced_lm_loss(self, trace: LLMTrace) -> torch.Tensor:
        self.apply_teacher_trace_memory(trace)
        prompt_encoded = self._tokenize(trace.final_prompt)
        full_encoded = self._tokenize(f"{trace.final_prompt} {trace.target_text}")
        input_ids = full_encoded["input_ids"]
        mask = full_encoded.get("attention_mask", torch.ones_like(input_ids))
        embeds = self.base_model.get_input_embeddings()(input_ids)
        prefix, _ = self.memory_prefix_for_prompt(trace.final_prompt, trace.current_scope)
        embeds = torch.cat([prefix, embeds], dim=1)
        prefix_mask = torch.ones(
            (mask.shape[0], prefix.shape[1]), dtype=mask.dtype, device=mask.device
        )
        mask = torch.cat([prefix_mask, mask], dim=1)
        labels = input_ids.clone()
        prompt_len = prompt_encoded["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
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
        outputs = self.base_model(inputs_embeds=embeds, attention_mask=mask)
        shift_logits = outputs.logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def generate_final(
        self,
        final_prompt: str,
        *,
        current_scope: str,
        use_memory: bool = True,
        max_new_tokens: int = 16,
    ) -> str:
        embeds, mask = self._embed_text(final_prompt)
        if use_memory:
            prefix, _ = self.memory_prefix_for_prompt(final_prompt, current_scope)
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
        if hasattr(self.base_model, "generate"):
            ids = self.base_model.generate(
                inputs_embeds=embeds,
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
            return self.tokenizer.decode(ids[0], skip_special_tokens=True)
        logits = self.base_model(inputs_embeds=embeds, attention_mask=mask).logits
        return str(int(logits[:, -1].argmax(dim=-1)[0].item()))

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
        return QwenMemoryConfig(scope_vocab=build_scope_vocab(traces), **kwargs)

    def run_trace_teacher_forced(self, trace: LLMTrace) -> str:
        self.apply_teacher_trace_memory(trace)
        return self.generate_final(
            trace.final_prompt,
            current_scope=trace.current_scope,
            use_memory=True,
        )
