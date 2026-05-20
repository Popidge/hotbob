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
