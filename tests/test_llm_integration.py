from __future__ import annotations

from types import MethodType

import pytest
import torch
from torch import nn

from hotbob.data.traces import generate_traces
from hotbob.experiment import checkpoint_payload
from hotbob.llm.architecture_compare import (
    bucket_losses,
    parse_variant,
    run_architecture_comparison,
)
from hotbob.llm.architecture_compare import (
    build_arg_parser as build_architecture_compare_arg_parser,
)
from hotbob.llm.compare import comparison_rows, parse_checkpoints
from hotbob.llm.dataset import (
    build_llm_scope_vocab,
    build_llm_tool_name_vocab,
    generate_llm_traces,
    privacy_report,
    task_trace_to_llm_trace,
    write_llm_jsonl,
)
from hotbob.llm.evaluate import (
    build_arg_parser as build_eval_arg_parser,
)
from hotbob.llm.evaluate import (
    evaluate_traces,
    exact_match,
    extract_allowed_answer,
    normalize_generated_text,
)
from hotbob.llm.evaluate import main as llm_eval_main
from hotbob.llm.generate_data import main as generate_llm_main
from hotbob.llm.memory_adapter import LowRankMemoryCorrectionAdapter, MemoryPrefixAdapter
from hotbob.llm.qwen_memory_model import QwenMemoryConfig, QwenMemoryModel
from hotbob.llm.train import build_arg_parser as build_train_arg_parser
from hotbob.llm.train import main as llm_train_main
from hotbob.model.memory_bank import MemoryBank
from hotbob.model.memory_write import MemoryWrite
from hotbob.training.dataset import build_scope_vocab, structured_targets_from_payload


class TinyTokenizer:
    def __call__(self, text: str | list[str], return_tensors: str = "pt", padding: bool = False):
        if isinstance(text, list):
            rows = [self(row, return_tensors=return_tensors)["input_ids"][0] for row in text]
            max_len = max(len(row) for row in rows)
            input_ids = torch.zeros((len(rows), max_len), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for idx, row in enumerate(rows):
                input_ids[idx, : len(row)] = row
                attention_mask[idx, : len(row)] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        ids = [min((ord(ch) % 31) + 1, 31) for ch in text][:16] or [1]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "decoded"


class TinyCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 8, vocab_size: int = 32) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        output = {"logits": self.lm_head(inputs_embeds)}
        if kwargs.get("output_hidden_states"):
            output["hidden_states"] = (inputs_embeds,)
        return type("Output", (), output)()


class TinyQwenAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.o_proj(torch.tanh(self.q_proj(hidden)))


class TinyQwenLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = TinyQwenAttention(hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.self_attn(hidden)


class TinyQwenBody(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyQwenLayer(hidden_size), TinyQwenLayer(hidden_size)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class TinyQwenLikeCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 8, vocab_size: int = 32) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.model = TinyQwenBody(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.generate_called = False

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        hidden = self.model(inputs_embeds)
        output = {"logits": self.lm_head(hidden)}
        if kwargs.get("output_hidden_states"):
            output["hidden_states"] = (hidden,)
        return type("Output", (), output)()

    def generate(self, inputs_embeds=None, attention_mask=None, **kwargs):
        self.generate_called = True
        _ = self.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return torch.tensor([[1, 2]], dtype=torch.long, device=inputs_embeds.device)


def tiny_memory_model(traces=None, integration_mode: str = "prefix") -> QwenMemoryModel:
    scope_vocab = build_scope_vocab(traces) if traces else {"default": 1}
    return QwenMemoryModel(
        QwenMemoryConfig(
            num_memory_slots=4,
            memory_prefix_len=3,
            scope_vocab=scope_vocab,
            freeze_base=True,
            integration_mode=integration_mode,
            correction_rank=4,
        ),
        base_model=TinyCausalLM(),
        tokenizer=TinyTokenizer(),
    )


def tiny_qwen_like_memory_model(traces=None, integration_mode: str = "attention_qo"):
    scope_vocab = build_scope_vocab(traces) if traces else {"default": 1}
    base = TinyQwenLikeCausalLM()
    return QwenMemoryModel(
        QwenMemoryConfig(
            num_memory_slots=4,
            memory_prefix_len=3,
            scope_vocab=scope_vocab,
            freeze_base=True,
            integration_mode=integration_mode,
            correction_rank=4,
            attention_patch_layers="last1",
        ),
        base_model=base,
        tokenizer=TinyTokenizer(),
    )


def tiny_train_model(config: QwenMemoryConfig) -> QwenMemoryModel:
    return QwenMemoryModel(config, base_model=TinyCausalLM(), tokenizer=TinyTokenizer())


class TinyLoRAMemoryModel(QwenMemoryModel):
    def _maybe_apply_lora(self) -> None:
        self.base_model.lora_probe = nn.Parameter(torch.ones(1))

    def teacher_forced_lm_loss(self, trace) -> torch.Tensor:
        return self.base_model.lora_probe.sum()

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {"lora_probe": self.base_model.lora_probe.detach().clone()}

    def load_lora_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.base_model.lora_probe.copy_(state["lora_probe"])


def tiny_lora_model(config: QwenMemoryConfig) -> TinyLoRAMemoryModel:
    return TinyLoRAMemoryModel(config, base_model=TinyCausalLM(), tokenizer=TinyTokenizer())


def test_llm_trace_conversion_preserves_privacy_and_scope_metadata() -> None:
    task_trace = next(
        t for t in generate_traces(20, seed=4) if t.task_family == "privacy_disclosure_conflict"
    )
    llm_trace = task_trace_to_llm_trace(task_trace)
    report = privacy_report(llm_trace)
    assert llm_trace.current_scope == task_trace.current_scope
    assert llm_trace.expected_memory_ops[0].scope == task_trace.current_scope
    assert llm_trace.metadata["task_family"] == "privacy_disclosure_conflict"
    assert not report.final_prompt_contains_hidden_value


def test_memory_prefix_shape_matches_model_hidden_size() -> None:
    adapter = MemoryPrefixAdapter(hidden_size=8, memory_prefix_len=4)
    model = tiny_memory_model()
    query = torch.randn(1, 8)
    prefix, attention = adapter(query, model.memory, torch.tensor([1]))
    assert prefix.shape == (1, 4, 8)
    assert attention.shape == (1, 4)


def test_qwen_memory_config_prefix_metadata_defaults_to_none() -> None:
    assert QwenMemoryConfig().memory_prefix_metadata_mode == "none"


def test_memory_prefix_metadata_mode_preserves_shape_and_uses_ids() -> None:
    bank = MemoryBank(num_slots=2, d_model=8)
    bank.reset(batch_size=1)
    vector = torch.ones(8)
    bank.apply_write(0, 0, vector, type_id=1, scope_id=1, privacy_id=1, authority_id=1)
    bank.apply_write(0, 1, vector, type_id=2, scope_id=1, privacy_id=1, authority_id=1)
    query = torch.randn(1, 8)
    none_adapter = MemoryPrefixAdapter(hidden_size=8, memory_prefix_len=4)
    metadata_adapter = MemoryPrefixAdapter(
        hidden_size=8,
        memory_prefix_len=4,
        num_types=4,
        num_scopes=3,
        num_privacy=3,
        num_authority=4,
        num_payload_kinds=4,
        num_policy_actions=8,
        num_authority_levels=8,
        metadata_mode="metadata",
    )
    none_prefix, _ = none_adapter(query, bank, torch.tensor([1]))
    metadata_prefix, attention = metadata_adapter(query, bank, torch.tensor([1]))
    bank.type_ids[0, 1] = bank.type_ids[0, 0]
    same_metadata_prefix, _ = metadata_adapter(query, bank, torch.tensor([1]))
    assert metadata_prefix.shape == none_prefix.shape == (1, 4, 8)
    assert attention.shape == (1, 2)
    assert not torch.allclose(metadata_prefix, same_metadata_prefix)


def test_memory_prefix_metadata_mode_uses_structured_payload_ids() -> None:
    bank = MemoryBank(num_slots=2, d_model=8)
    bank.reset(batch_size=1)
    vector = torch.ones(8)
    bank.apply_write(
        0,
        0,
        vector,
        type_id=1,
        scope_id=1,
        privacy_id=1,
        authority_id=1,
        payload_kind_id=1,
        payload_default_action_id=1,
        payload_winning_authority_level_id=1,
        payload_losing_authority_level_id=1,
    )
    bank.apply_write(
        0,
        1,
        vector,
        type_id=1,
        scope_id=1,
        privacy_id=1,
        authority_id=1,
        payload_kind_id=1,
        payload_default_action_id=2,
        payload_winning_authority_level_id=3,
        payload_losing_authority_level_id=4,
    )
    adapter = MemoryPrefixAdapter(
        hidden_size=8,
        memory_prefix_len=4,
        num_types=4,
        num_scopes=3,
        num_privacy=3,
        num_authority=4,
        num_payload_kinds=4,
        num_policy_actions=8,
        num_authority_levels=8,
        metadata_mode="metadata",
    )
    query = torch.randn(1, 8)
    prefix, _ = adapter(query, bank, torch.tensor([1]))
    bank.payload_default_action_ids[0, 1] = bank.payload_default_action_ids[0, 0]
    bank.payload_winning_authority_level_ids[0, 1] = (
        bank.payload_winning_authority_level_ids[0, 0]
    )
    bank.payload_losing_authority_level_ids[0, 1] = (
        bank.payload_losing_authority_level_ids[0, 0]
    )
    same_payload_prefix, _ = adapter(query, bank, torch.tensor([1]))
    assert not torch.allclose(prefix, same_payload_prefix)


def test_memory_write_outputs_winning_and_losing_authority_heads() -> None:
    writer = MemoryWrite(
        d_model=8,
        num_slots=4,
        num_types=3,
        num_scopes=3,
        num_privacy=2,
        num_authority=3,
        num_authority_levels=7,
    )
    outputs = writer(torch.randn(2, 8))
    assert outputs["payload_winning_authority_level_logits"].shape == (2, 7)
    assert outputs["payload_losing_authority_level_logits"].shape == (2, 7)


def test_attention_correction_adapter_cpu_smoke_forward() -> None:
    bank = MemoryBank(num_slots=3, d_model=8)
    bank.reset(batch_size=1)
    bank.apply_write(
        0,
        0,
        torch.ones(8),
        type_id=1,
        scope_id=1,
        privacy_id=1,
        authority_id=1,
    )
    adapter = LowRankMemoryCorrectionAdapter(
        hidden_size=8,
        rank=4,
        num_types=6,
        num_scopes=3,
        num_privacy=3,
        num_authority=4,
    )
    hidden = torch.randn(1, 5, 8)
    corrected, q_attention, o_attention = adapter(
        hidden,
        bank,
        torch.tensor([1]),
        apply_q=True,
        apply_o=True,
    )
    assert corrected.shape == hidden.shape
    assert q_attention is not None and q_attention.shape == (1, 5, 3)
    assert o_attention is not None and o_attention.shape == (1, 5, 3)


def test_frozen_base_parameters_have_requires_grad_false() -> None:
    model = tiny_memory_model()
    assert all(not param.requires_grad for param in model.base_model.parameters())
    assert any(param.requires_grad for param in model.memory_heads.parameters())


def test_qwen_memory_config_lora_defaults_to_none() -> None:
    config = QwenMemoryConfig()
    assert config.lora_backend == "none"
    assert config.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")


def test_peft_missing_raises_only_when_lora_requested(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "peft":
            raise ImportError("missing peft")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    QwenMemoryModel(
        QwenMemoryConfig(lora_backend="none"),
        base_model=TinyCausalLM(),
        tokenizer=TinyTokenizer(),
    )
    with pytest.raises(RuntimeError, match="Install peft"):
        QwenMemoryModel(
            QwenMemoryConfig(lora_backend="peft"),
            base_model=TinyCausalLM(),
            tokenizer=TinyTokenizer(),
        )


def test_checkpoint_payload_can_include_lora_state() -> None:
    payload = checkpoint_payload(
        config={"integration_mode": "prefix"},
        memory_heads_state={"head": torch.tensor([1.0])},
        lora_state={"lora_probe": torch.tensor([2.0])},
        losses=[1.0],
        training_args={},
    )
    assert "lora_state" in payload
    assert torch.equal(payload["lora_state"]["lora_probe"], torch.tensor([2.0]))


def test_teacher_forced_memory_changes_generated_logits_versus_context_only() -> None:
    task_trace = generate_traces(1, seed=9)[0]
    llm_trace = task_trace_to_llm_trace(task_trace)
    model = tiny_memory_model([task_trace])
    context_logits = model.final_prompt_logits(
        llm_trace.final_prompt,
        current_scope=llm_trace.current_scope,
        use_memory=False,
    )
    model.apply_teacher_trace_memory(llm_trace)
    memory_logits = model.final_prompt_logits(
        llm_trace.final_prompt,
        current_scope=llm_trace.current_scope,
        use_memory=True,
    )
    assert memory_logits.shape[1] == context_logits.shape[1] + model.config_obj.memory_prefix_len
    assert not torch.allclose(memory_logits[:, : context_logits.shape[1]], context_logits)


def test_llm_memory_value_encoding_uses_structured_payload_text(monkeypatch) -> None:
    task_trace = next(
        trace for trace in generate_traces(32, seed=7) if trace.task_family == "authority_conflict"
    )
    llm_trace = task_trace_to_llm_trace(task_trace)
    model = tiny_memory_model([task_trace])
    captured: list[str] = []

    def fake_encode_event(role: str, content: str, scope: str | None = None) -> torch.Tensor:
        if role == "MEMORY_VALUE":
            captured.append(content)
        return torch.ones((1, model.hidden_size))

    monkeypatch.setattr(model, "encode_event", fake_encode_event)
    model.apply_teacher_trace_memory(llm_trace)
    model.write_supervision_loss(llm_trace)
    assert any("authority_rule" in item for item in captured)
    assert any("tool_unverified" in item or "model_inferred" in item for item in captured)


def test_teacher_forced_memory_stores_structured_payload_metadata() -> None:
    task_trace = next(
        trace for trace in generate_traces(32, seed=7) if trace.task_family == "authority_conflict"
    )
    llm_trace = task_trace_to_llm_trace(task_trace)
    model = tiny_memory_model([task_trace])
    model.apply_teacher_trace_memory(llm_trace)
    structured = structured_targets_from_payload(llm_trace.expected_memory_ops[0].payload)
    assert int(model.memory.payload_kind_ids[0, 0].item()) == structured["payload_kind_id"]
    assert (
        int(model.memory.payload_default_action_ids[0, 0].item())
        == structured["default_action_id"]
    )
    assert (
        int(model.memory.payload_winning_authority_level_ids[0, 0].item())
        == structured["winning_authority_level_id"]
    )
    assert (
        int(model.memory.payload_losing_authority_level_ids[0, 0].item())
        == structured["losing_authority_level_id"]
    )


def test_attention_qo_memory_changes_logits_without_prefix_tokens() -> None:
    task_trace = generate_traces(1, seed=9)[0]
    llm_trace = task_trace_to_llm_trace(task_trace)
    model = tiny_memory_model([task_trace], integration_mode="attention_qo")
    context_logits = model.final_prompt_logits(
        llm_trace.final_prompt,
        current_scope=llm_trace.current_scope,
        use_memory=False,
    )
    model.apply_teacher_trace_memory(llm_trace)
    memory_logits = model.final_prompt_logits(
        llm_trace.final_prompt,
        current_scope=llm_trace.current_scope,
        use_memory=True,
    )
    assert memory_logits.shape == context_logits.shape
    assert not torch.allclose(memory_logits, context_logits)


def test_attention_correction_ignores_wrong_scope_memory() -> None:
    task_trace = generate_traces(1, seed=9)[0]
    model = tiny_memory_model([task_trace], integration_mode="attention_qo")
    model.memory.apply_write(
        0,
        0,
        torch.ones(model.hidden_size),
        type_id=1,
        scope_id=999,
        privacy_id=1,
        authority_id=1,
    )
    context_logits = model.final_prompt_logits(
        "Answer now.",
        current_scope=task_trace.current_scope,
        use_memory=False,
    )
    memory_logits = model.final_prompt_logits(
        "Answer now.",
        current_scope=task_trace.current_scope,
        use_memory=True,
    )
    assert torch.allclose(memory_logits, context_logits)


def test_attention_qo_instantiates_with_tiny_fake_model() -> None:
    model = tiny_memory_model(integration_mode="attention_qo")
    assert model.config_obj.integration_mode == "attention_qo"
    assert all(not param.requires_grad for param in model.base_model.parameters())


def test_attention_qo_uses_internal_patch_and_generate_with_qwen_like_model() -> None:
    task_trace = generate_traces(1, seed=9)[0]
    llm_trace = task_trace_to_llm_trace(task_trace)
    model = tiny_qwen_like_memory_model([task_trace])
    model.apply_teacher_trace_memory(llm_trace)
    text = model.generate_final(
        llm_trace.final_prompt,
        current_scope=llm_trace.current_scope,
        use_memory=True,
        max_new_tokens=2,
    )
    assert text == "decoded"
    assert model.base_model.generate_called
    assert model._has_attention_patch_targets(apply_q=True, apply_o=True)


def test_attention_patch_targets_are_cached_and_last_layer_selected() -> None:
    model = tiny_qwen_like_memory_model()
    first_q, first_o = model._attention_patch_targets()
    model.base_model.model.layers.append(TinyQwenLayer(model.hidden_size))
    second_q, second_o = model._attention_patch_targets()
    assert first_q is second_q
    assert first_o is second_o
    assert len(first_q) == 1
    assert first_q[0] is model.base_model.model.layers[1].self_attn.q_proj
    assert len(first_o) == 1
    assert first_o[0] is model.base_model.model.layers[1].self_attn.o_proj


def test_attention_fallback_residual_path_still_works_for_non_qwen_fake_model() -> None:
    model = tiny_memory_model(integration_mode="attention_q")
    assert not model._has_attention_patch_targets(apply_q=True, apply_o=False)
    logits = model.final_prompt_logits("Answer now.", current_scope="default", use_memory=True)
    assert logits.shape[1] == model._tokenize("Answer now.")["input_ids"].shape[1]


def test_debug_dump_redacts_hidden_secrets_by_default() -> None:
    task_trace = next(
        t for t in generate_traces(20, seed=4) if t.task_family == "privacy_disclosure_conflict"
    )
    llm_trace = task_trace_to_llm_trace(task_trace)
    model = tiny_memory_model([task_trace])
    model.apply_teacher_trace_memory(llm_trace)
    dump = model.debug_dump()
    assert dump
    assert dump[0]["value"] == "<redacted>"


def test_evaluation_normalizes_generated_text_correctly() -> None:
    assert normalize_generated_text("  Hold   fire!  ") == "hold fire."
    assert exact_match("Inspect dave", "Inspect dave.")
    assert extract_allowed_answer(" Inspect dave. | Correct. | Hold fire.") == "Inspect dave."


def test_llm_train_and_evaluate_argument_parsing_accept_attention_modes() -> None:
    train_args = build_train_arg_parser().parse_args(
        [
            "--traces",
            "data/llm_train.jsonl",
            "--integration-mode",
            "attention_qo",
            "--memory-state-mode",
            "by_type",
            "--memory-prefix-metadata-mode",
            "metadata",
            "--attention-patch-layers",
            "last4",
            "--freeze-base",
            "--family-loss-weight",
            "authority_conflict=2",
        ]
    )
    assert train_args.integration_mode == "attention_qo"
    assert train_args.memory_state_mode == "by_type"
    assert train_args.memory_prefix_metadata_mode == "metadata"
    assert train_args.attention_patch_layers == "last4"
    assert train_args.family_loss_weight == ["authority_conflict=2"]
    assert train_args.freeze_base
    assert train_args.structured_loss_weight == 0.2
    assert train_args.authority_payload_loss_weight == 1.0
    assert train_args.memory_heads_checkpoint is None
    assert not train_args.freeze_memory_heads
    assert train_args.lora_backend == "none"
    lora_args = build_train_arg_parser().parse_args(
        [
            "--traces",
            "data/llm_train.jsonl",
            "--memory-heads-checkpoint",
            "runs/qwen_memory/controller.pt",
            "--freeze-memory-heads",
            "--lora-backend",
            "peft",
            "--lora-r",
            "8",
            "--lora-target-modules",
            "q_proj",
            "o_proj",
            "--authority-payload-loss-weight",
            "4.0",
        ]
    )
    assert lora_args.lora_backend == "peft"
    assert lora_args.lora_r == 8
    assert lora_args.authority_payload_loss_weight == 4.0
    assert lora_args.lora_target_modules == ["q_proj", "o_proj"]
    assert lora_args.memory_heads_checkpoint == "runs/qwen_memory/controller.pt"
    assert lora_args.freeze_memory_heads

    eval_args = build_eval_arg_parser().parse_args(
        [
            "--checkpoint",
            "runs/qwen_memory/latest.pt",
            "--traces",
            "data/llm_eval.jsonl",
            "--integration-mode",
            "attention_q",
            "--mode",
            "all",
        ]
    )
    assert eval_args.integration_mode == "attention_q"
    assert eval_args.mode == "all"
    eval_args = build_eval_arg_parser().parse_args(
        ["--traces", "data/llm_eval.jsonl", "--batch-size", "2"]
    )
    assert eval_args.batch_size == 2


def test_authority_payload_loss_weight_scales_failing_write_heads() -> None:
    task_trace = next(
        trace for trace in generate_traces(40, seed=3) if trace.task_family == "authority_conflict"
    )
    llm_trace = task_trace_to_llm_trace(task_trace)
    model = tiny_memory_model([task_trace])

    def fixed_forward_event(self, role, content, scope=None):
        hidden = torch.zeros((1, self.hidden_size), requires_grad=True)
        return {
            "op_logits": torch.zeros((1, 32), requires_grad=True),
            "type_logits": torch.zeros((1, 32), requires_grad=True),
            "scope_logits": torch.zeros((1, 32), requires_grad=True),
            "privacy_logits": torch.zeros((1, 32), requires_grad=True),
            "authority_logits": torch.zeros((1, 32), requires_grad=True),
            "slot_logits": torch.zeros((1, self.config_obj.num_memory_slots), requires_grad=True),
            "value_vector": hidden,
            "payload_kind_logits": torch.zeros((1, 16), requires_grad=True),
            "payload_default_action_logits": torch.zeros((1, 16), requires_grad=True),
            "payload_trigger_logits": torch.zeros((1, 16), requires_grad=True),
            "payload_exception_logits": torch.zeros((1, 16), requires_grad=True),
            "payload_expiry_policy_logits": torch.zeros((1, 16), requires_grad=True),
            "payload_authority_level_logits": torch.zeros((1, 32), requires_grad=True),
            "payload_winning_authority_level_logits": torch.zeros((1, 32), requires_grad=True),
            "payload_losing_authority_level_logits": torch.zeros((1, 32), requires_grad=True),
            "payload_tool_name_logits": torch.zeros((1, 16), requires_grad=True),
            "payload_route_step_logits": torch.zeros((1, 8), requires_grad=True),
        }

    model.forward_event = MethodType(fixed_forward_event, model)
    model.config_obj.authority_payload_loss_weight = 1.0
    base_loss = model.write_supervision_loss(llm_trace)
    model.config_obj.authority_payload_loss_weight = 4.0
    weighted_loss = model.write_supervision_loss(llm_trace)

    assert weighted_loss > base_loss


def test_llm_evaluate_reports_mode_and_integration_metrics() -> None:
    task_trace = generate_traces(1, seed=9)[0]
    llm_trace = task_trace_to_llm_trace(task_trace)
    model = tiny_memory_model([task_trace], integration_mode="attention_qo")
    result = evaluate_traces(model, [llm_trace], mode="context_only")
    assert result["mode"] == "context_only"
    assert result["integration_mode"] == "attention_qo"
    assert result["decode_strategy"] == "score_answers"
    assert "secret_leak_failures" in result
    assert "wrong_scope_failures" in result
    assert "expiry_failures" in result
    assert "authority_conflict_failures" in result
    assert "tool_override_failures" in result
    assert "structured_policy_failures" in result


def test_candidate_scoring_returns_allowed_answer() -> None:
    task_trace = generate_traces(1, seed=9)[0]
    llm_trace = task_trace_to_llm_trace(task_trace)
    model = tiny_qwen_like_memory_model([task_trace])
    model.apply_teacher_trace_memory(llm_trace)
    answer = model.choose_final_candidate(
        llm_trace.final_prompt,
        ["Correct.", "Hold fire."],
        current_scope=llm_trace.current_scope,
        use_memory=True,
    )
    assert answer in {"Correct.", "Hold fire."}


def test_prefix_candidate_scoring_masks_prefix_and_prompt_positions() -> None:
    model = tiny_memory_model(integration_mode="prefix")
    encoded = model._tokenize_texts(model._candidate_texts("Prompt", ["Correct."]))
    base_mask = model._candidate_token_spans("Prompt", ["Correct."], encoded)
    scores = model.score_final_candidates(
        "Prompt",
        ["Correct."],
        current_scope="default",
        use_memory=True,
    )
    prefix_len = model.config_obj.memory_prefix_len
    padded_mask = torch.cat(
        [
            torch.zeros((1, prefix_len), dtype=torch.bool),
            base_mask,
        ],
        dim=1,
    )
    assert scores.shape == (1,)
    assert not padded_mask[:, :prefix_len].any()
    assert padded_mask.any()


def test_llm_evaluate_batch_size_results_match() -> None:
    traces = [task_trace_to_llm_trace(trace) for trace in generate_traces(2, seed=9)]
    result_one = evaluate_traces(
        tiny_memory_model(integration_mode="attention_qo"),
        traces,
        mode="context_only",
        batch_size=1,
    )
    result_two = evaluate_traces(
        tiny_memory_model(integration_mode="attention_qo"),
        traces,
        mode="context_only",
        batch_size=2,
    )
    assert result_one == result_two


def test_llm_train_batch_size_two_writes_checkpoint(tmp_path, monkeypatch) -> None:
    traces = [task_trace_to_llm_trace(trace) for trace in generate_traces(2, seed=11)]
    trace_path = tmp_path / "llm_train.jsonl"
    out_path = tmp_path / "checkpoint.pt"
    write_llm_jsonl(traces, trace_path)
    monkeypatch.setattr("hotbob.llm.train.QwenMemoryModel", tiny_train_model)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--traces",
            str(trace_path),
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--integration-mode",
            "attention_qo",
            "--memory-state-mode",
            "by_type",
            "--correction-rank",
            "4",
            "--attention-patch-layers",
            "last1",
            "--freeze-base",
            "--out",
            str(out_path),
        ],
    )
    llm_train_main()
    state = torch.load(out_path, map_location="cpu", weights_only=False)
    assert state["config"]["integration_mode"] == "attention_qo"
    assert state["config"]["memory_state_mode"] == "by_type"
    assert state["config"]["correction_rank"] == 4
    assert state["config"]["attention_patch_layers"] == "last1"
    assert state["config"]["structured_loss_weight"] == 0.2
    assert state["config"]["authority_payload_loss_weight"] == 1.0
    assert len(state["losses"]) == 1


def test_llm_train_can_load_and_freeze_memory_heads_for_lora_stage(tmp_path, monkeypatch) -> None:
    traces = [task_trace_to_llm_trace(trace) for trace in generate_traces(2, seed=11)]
    trace_path = tmp_path / "llm_train.jsonl"
    memory_checkpoint = tmp_path / "memory.pt"
    out_path = tmp_path / "lora_stage.pt"
    write_llm_jsonl(traces, trace_path)
    model = tiny_lora_model(
        QwenMemoryConfig(
            scope_vocab=build_llm_scope_vocab(traces),
            tool_name_vocab=build_llm_tool_name_vocab(traces),
            lora_backend="peft",
        )
    )
    torch.save(
        checkpoint_payload(
            config={**model.config_obj.__dict__},
            memory_heads_state=model.memory_heads.state_dict(),
            losses=[0.5],
            training_args={},
        ),
        memory_checkpoint,
    )
    monkeypatch.setattr("hotbob.llm.train.QwenMemoryModel", tiny_lora_model)
    args = build_train_arg_parser().parse_args(
        [
            "--model",
            "tiny",
            "--traces",
            str(trace_path),
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--integration-mode",
            "prefix",
            "--freeze-base",
            "--memory-heads-checkpoint",
            str(memory_checkpoint),
            "--freeze-memory-heads",
            "--lora-backend",
            "peft",
            "--write-loss-weight",
            "0",
            "--out",
            str(out_path),
        ]
    )
    from hotbob.llm.train import train_checkpoint

    train_checkpoint(args)
    state = torch.load(out_path, map_location="cpu", weights_only=False)
    assert state["training_args"]["freeze_memory_heads"]
    assert "lora_state" in state


def test_llm_compare_context_only_json_rows(monkeypatch) -> None:
    traces = [task_trace_to_llm_trace(trace) for trace in generate_traces(2, seed=9)]
    monkeypatch.setattr(
        "hotbob.llm.compare.QwenMemoryModel",
        lambda config: tiny_train_model(config),
    )
    rows = comparison_rows(
        traces=traces,
        model_name="tiny",
        checkpoints=parse_checkpoints([]),
        modes=["context_only"],
        decode_strategy="score_answers",
        batch_size=2,
    )
    assert rows[0]["run_name"] == "context_only"
    assert rows[0]["eval_mode"] == "context_only"
    assert "aggregate_accuracy" in rows[0]
    assert "tool_routing_failures" in rows[0]
    assert rows[0]["lora_backend"] == "none"
    assert rows[0]["lora_r"] == 16
    assert rows[0]["lora_target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_compare_loads_memory_and_lora_checkpoint(tmp_path, monkeypatch) -> None:
    traces = [task_trace_to_llm_trace(trace) for trace in generate_traces(1, seed=9)]
    checkpoint_path = tmp_path / "lora.pt"
    model = tiny_lora_model(
        QwenMemoryConfig(
            num_memory_slots=4,
            memory_prefix_len=3,
            correction_rank=4,
            lora_backend="peft",
            lora_r=4,
            lora_target_modules=("q_proj",),
        )
    )
    torch.save(
        checkpoint_payload(
            config={**model.config_obj.__dict__},
            memory_heads_state=model.memory_heads.state_dict(),
            lora_state={"lora_probe": torch.tensor([5.0])},
            losses=[0.5],
            training_args={},
        ),
        checkpoint_path,
    )
    monkeypatch.setattr("hotbob.llm.compare.QwenMemoryModel", tiny_lora_model)
    rows = comparison_rows(
        traces=traces,
        model_name="tiny",
        checkpoints={"prefix_lora": str(checkpoint_path)},
        modes=["context_only"],
        decode_strategy="score_answers",
        batch_size=1,
    )
    assert rows[0]["lora_backend"] == "peft"
    assert rows[0]["lora_r"] == 4
    assert rows[0]["lora_target_modules"] == ["q_proj"]


def test_evaluate_loads_memory_and_lora_checkpoint(tmp_path, monkeypatch) -> None:
    traces = [task_trace_to_llm_trace(trace) for trace in generate_traces(1, seed=9)]
    trace_path = tmp_path / "llm_eval.jsonl"
    checkpoint_path = tmp_path / "lora.pt"
    write_llm_jsonl(traces, trace_path)
    model = tiny_lora_model(
        QwenMemoryConfig(
            num_memory_slots=4,
            memory_prefix_len=3,
            correction_rank=4,
            lora_backend="peft",
            lora_target_modules=("q_proj",),
        )
    )
    torch.save(
        checkpoint_payload(
            config={**model.config_obj.__dict__},
            memory_heads_state=model.memory_heads.state_dict(),
            lora_state={"lora_probe": torch.tensor([7.0])},
            losses=[0.5],
            training_args={},
        ),
        checkpoint_path,
    )
    monkeypatch.setattr("hotbob.llm.evaluate.QwenMemoryModel", tiny_lora_model)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint_path),
            "--traces",
            str(trace_path),
            "--mode",
            "context_only",
        ],
    )
    llm_eval_main()


def test_architecture_compare_parses_variants_and_buckets_losses() -> None:
    variant = parse_variant("attention_qo:by_type:last4")
    assert variant.name == "attention_qo_by_type_last4"
    assert variant.integration_mode == "attention_qo"
    assert variant.memory_state_mode == "by_type"
    assert variant.attention_patch_layers == "last4"
    layer_only = parse_variant("attention_q:last1")
    assert layer_only.name == "attention_q_last1"
    assert layer_only.memory_state_mode == "shared"
    assert layer_only.attention_patch_layers == "last1"
    assert bucket_losses([1.0, 3.0, 5.0], 2) == [
        {"start_step": 1, "end_step": 2, "mean_loss": 2.0},
        {"start_step": 3, "end_step": 3, "mean_loss": 5.0},
    ]


def test_architecture_compare_runs_tiny_matrix(tmp_path, monkeypatch) -> None:
    traces = [task_trace_to_llm_trace(trace) for trace in generate_traces(2, seed=11)]
    train_path = tmp_path / "llm_train.jsonl"
    eval_path = tmp_path / "llm_eval.jsonl"
    run_dir = tmp_path / "runs"
    write_llm_jsonl(traces, train_path)
    write_llm_jsonl(traces, eval_path)
    monkeypatch.setattr("hotbob.llm.train.QwenMemoryModel", tiny_train_model)
    monkeypatch.setattr("hotbob.llm.compare.QwenMemoryModel", tiny_train_model)
    args = build_architecture_compare_arg_parser().parse_args(
        [
            "--model",
            "tiny",
            "--train-traces",
            str(train_path),
            "--eval-traces",
            str(eval_path),
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--correction-rank",
            "4",
            "--run-dir",
            str(run_dir),
            "--variant",
            "prefix",
            "--variant",
            "attention_qo:by_type:last1",
        ]
    )
    result = run_architecture_comparison(args)
    assert (run_dir / "comparison.json").exists()
    assert result["config"]["structured_loss_weight"] == 0.2
    assert result["config"]["authority_payload_loss_weight"] == 1.0
    assert len(result["training"]) == 2
    assert len(result["evaluation"]) == 2
    assert result["training"][1]["variant"]["memory_state_mode"] == "by_type"
    assert (run_dir / "checkpoints" / "prefix.pt").exists()


def test_llm_generate_data_cli_argument_parsing(tmp_path, monkeypatch) -> None:
    train_out = tmp_path / "llm_train.jsonl"
    eval_out = tmp_path / "llm_eval.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_data",
            "--train-out",
            str(train_out),
            "--eval-out",
            str(eval_out),
            "--train-n",
            "3",
            "--eval-n",
            "2",
            "--seed",
            "3",
        ],
    )
    generate_llm_main()
    assert len(generate_llm_traces(1, 1)) == 1
    assert train_out.read_text(encoding="utf-8").count("\n") == 3
    assert eval_out.read_text(encoding="utf-8").count("\n") == 2


@pytest.mark.skip(reason="slow/network Qwen smoke test is skipped by default")
def test_qwen_load_smoke_network() -> None:
    model = QwenMemoryModel.from_pretrained(
        QwenMemoryConfig(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    )
    text = model.generate_final(
        "Say OK.",
        current_scope="default",
        use_memory=False,
        max_new_tokens=2,
    )
    assert isinstance(text, str)
