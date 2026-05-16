from __future__ import annotations

import pytest
import torch
from torch import nn

from hotbob.data.traces import generate_traces
from hotbob.llm.dataset import generate_llm_traces, privacy_report, task_trace_to_llm_trace
from hotbob.llm.evaluate import exact_match, normalize_generated_text
from hotbob.llm.generate_data import main as generate_llm_main
from hotbob.llm.memory_adapter import MemoryPrefixAdapter
from hotbob.llm.qwen_memory_model import QwenMemoryConfig, QwenMemoryModel
from hotbob.training.dataset import build_scope_vocab


class TinyTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt"):
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

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        return type("Output", (), {"logits": self.lm_head(inputs_embeds)})()


def tiny_memory_model(traces=None) -> QwenMemoryModel:
    scope_vocab = build_scope_vocab(traces) if traces else {"default": 1}
    return QwenMemoryModel(
        QwenMemoryConfig(
            num_memory_slots=4,
            memory_prefix_len=3,
            scope_vocab=scope_vocab,
            freeze_base=True,
        ),
        base_model=TinyCausalLM(),
        tokenizer=TinyTokenizer(),
    )


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
