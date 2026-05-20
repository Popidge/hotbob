from __future__ import annotations

import pytest
import torch
from torch import nn

from hotbob.data.traces import generate_traces
from hotbob.llm.architecture_compare import (
    bucket_losses,
    build_arg_parser as build_architecture_compare_arg_parser,
    parse_variant,
    run_architecture_comparison,
)
from hotbob.llm.compare import comparison_rows, parse_checkpoints
from hotbob.llm.dataset import (
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
from hotbob.llm.generate_data import main as generate_llm_main
from hotbob.llm.memory_adapter import LowRankMemoryCorrectionAdapter, MemoryPrefixAdapter
from hotbob.llm.qwen_memory_model import QwenMemoryConfig, QwenMemoryModel
from hotbob.llm.train import build_arg_parser as build_train_arg_parser
from hotbob.llm.train import main as llm_train_main
from hotbob.model.memory_bank import MemoryBank
from hotbob.training.dataset import build_scope_vocab


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


def test_llm_trace_conversion_preserves_privacy_and_scope_metadata() -> None:
    task_trace = next(t for t in generate_traces(20, seed=4) if t.task_family == "hidden_colour")
    llm_trace = task_trace_to_llm_trace(task_trace)
    report = privacy_report(llm_trace)
    assert llm_trace.current_scope == task_trace.current_scope
    assert llm_trace.expected_memory_ops[0].scope == task_trace.current_scope
    assert llm_trace.metadata["task_family"] == "hidden_colour"
    assert not report.final_prompt_contains_hidden_value


def test_memory_prefix_shape_matches_model_hidden_size() -> None:
    adapter = MemoryPrefixAdapter(hidden_size=8, memory_prefix_len=4)
    model = tiny_memory_model()
    query = torch.randn(1, 8)
    prefix, attention = adapter(query, model.memory, torch.tensor([1]))
    assert prefix.shape == (1, 4, 8)
    assert attention.shape == (1, 4)


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
    task_trace = next(t for t in generate_traces(20, seed=4) if t.task_family == "hidden_colour")
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
            "--attention-patch-layers",
            "last4",
            "--freeze-base",
        ]
    )
    assert train_args.integration_mode == "attention_qo"
    assert train_args.memory_state_mode == "by_type"
    assert train_args.attention_patch_layers == "last4"
    assert train_args.freeze_base

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
    assert len(state["losses"]) == 1


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
