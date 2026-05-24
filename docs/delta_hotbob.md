# Delta-HotBob

HotBob is not trying to become a generic long-context replacement. Titans and
delta-mem frame memory as a compact way to preserve broad history for language
models. HotBob's narrower target is operational working memory: small typed
objects that affect policy, action, and tool choice in the hot inference path.

The memory object remains explicit:

- type
- scope
- privacy
- authority
- expiry/lifecycle
- active/inactive state
- strength

The delta-mem idea worth adapting is not "store everything." It is that compact
online memory can steer a frozen backbone directly, and that low-rank query-side
and output-side corrections are a practical efficiency tradeoff. For HotBob,
the question is whether scoped typed memory can condition a decoder more deeply
than a soft prefix while preserving the audit/debug surface.

## First Experiment

The first delta-HotBob mode keeps Qwen frozen and trains only HotBob memory
heads plus a low-rank correction adapter.

Modes:

- `prefix`: existing learned soft-prefix adapter.
- `attention_q`: memory-conditioned q-style residual correction before the
  frozen decoder.
- `attention_o`: memory-conditioned o-style residual correction after the
  frozen decoder hidden state and before the LM head.
- `attention_qo`: both q-style and o-style corrections.

The Qwen path patches selected `self_attn.q_proj` and `self_attn.o_proj` modules
with scoped forward pre-hooks. That keeps the original Hugging Face module
signatures and `generate()` cache behavior while injecting memory-conditioned
low-rank corrections inside the attention path. If a non-Qwen fake or unsupported
model has no Qwen-style attention targets, HotBob falls back to the earlier
residual approximation for CPU tests and development.

The correction readout consumes `MemoryBank`, not prompt text. It uses only
active slots in the current normalized scope. Memory vectors are enriched with
type, scope, privacy, and authority embeddings before readout, so the correction
path still sees HotBob metadata instead of raw text.

## What This Proves

This can show whether low-rank memory-conditioned residual corrections are a
useful next integration path compared with the existing prefix adapter.

It does not prove:

- true attention-internal delta-mem behavior
- robust long-context replacement
- production safety for private memory
- that predicted writes are good enough for real agents

## Planned Comparison

Run the same traces with:

- context-only
- current prefix memory adapter
- q-only correction
- o-only correction
- q+o correction

Later ablations should add corrupted memory, wrong-scope memory, inactive memory,
and type-grouped memory states to test interference and scope isolation.

## Typed Memory State Scaffolding

The current default is `shared`: all active same-scope slots are read together.

The experimental `by_type` state mode groups active same-scope slots by memory
type before readout. This mirrors the multi-state motivation from delta-mem in
HotBob terms: reduce interference by separating operational memory categories.
It is scaffolding for later experiments, not a tuned result.

## Typed Payload Roadmap

1. Structured trace targets: rich `MemoryOp.payload` schemas and supervised
   payload heads for policy/action/trigger/expiry/authority/tool-route fields.
2. q/o middle/all-layer correction: broaden the current last-layer comparison
   into deeper layer placement and type-grouped correction experiments.
3. Native memory read/write gates inside transformer blocks: move beyond prefix
   and residual correction into architecture-native operational memory.
4. Train-from-zero deeply integrated memory architecture: test whether memory
   mechanisms become more reliable when learned as part of the backbone rather
   than attached to a frozen decoder.

## Prefix+LoRA Memory Reader Experiment

The structured memory controller is now strong enough to isolate the LLM
integration question: can Qwen consume HotBob memory-prefix embeddings reliably
when the decoder is allowed a small adapter rather than being fully frozen?

The frozen-prefix baseline validates the memory route but still struggles on
authority semantics. Prefix+LoRA keeps the same structured memory writes and
prefix adapter, freezes the base Qwen weights, and trains only HotBob memory
heads plus PEFT LoRA weights on Qwen attention projections. The memory-ablated
comparison is the same LoRA checkpoint evaluated in `context_only`, so memory is
disabled while prompt priors and LoRA weights remain active.

If `context_only` rises with `predicted` and `teacher_forced`, treat that as
prompt-pattern contamination rather than memory use. The stricter follow-up is a
two-stage split: train memory heads on one trace split, load and freeze them,
then train only LoRA on a separate trace split before evaluating on a third
split.

This experiment is deliberately before q/o and native memory gates. q/o
correction and native transformer memory remain later integration experiments
after the prefix+LoRA baseline establishes whether a small decoder adapter can
learn to read the existing memory-prefix channel.

## Authority Memory Reader Follow-Up

Authority is security-relevant for prompt-injection-resistant agent harnesses:
the model has to prefer stored system, captain, or verified-tool authority over
lower-authority incoming text without copying hidden/private values into the
prompt. The current failure appears in both teacher-forced and predicted modes,
so the first fix has to improve memory readout and encoding rather than only the
write controller.

Prefix metadata enrichment is a conservative next step before q/o correction or
native memory gates. It leaves the baseline prefix mode unchanged by default,
but allows the authority experiment to add learned type, scope, privacy, and
authority embeddings to memory vectors before prefix readout. The latest
authority follow-up also stores structured payload metadata on memory slots and
adds learned prefix embeddings for payload kind, payload action, and winning/
losing authority levels. The same follow-up encodes full structured authority
payload text for LLM memory values and adds explicit winning and losing
authority supervision.

Authority traces now include a small bounded prompt-injection/channel-authority
slice: system-over-user, captain-over-unverified-tool, and verified-tool-over-
model-inference conflicts include realistic lower-authority pressure while the
target remains the stored authority rule.

## Attribution

This experiment adapts ideas from:

```bibtex
@misc{lei2026deltamemefficientonlinememory,
  title={$\delta$-mem: Efficient Online Memory for Large Language Models},
  author={Jingdi Lei and Di Zhang and Junxian Li and Weida Wang and Kaixuan Fan and Xiang Liu and Qihan Liu and Xiaoteng Ma and Baian Chen and Soujanya Poria},
  year={2026},
  eprint={2605.12357},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2605.12357},
}
```

HotBob uses delta-mem as an architectural and methodological reference while
remaining focused on scoped, typed, auditable operational working memory.
