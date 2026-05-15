# HotBob

HotBob is a small research prototype for a **Scoped Typed Working Memory Transformer**.

Central claim:

> Prompt context is a poor substrate for task-local operational state. HotBob tests whether a scoped, typed neural working-memory layer can preserve and retrieve operational facts across task boundaries without representing that memory only as prompt text.

This is not a production LLM. It is a compact lab for testing whether a transformer-like model can store task-local facts in neural memory slots, retrieve them later as activations, and make action decisions when the answer is no longer present in the prompt.

## Why The Name?

When you ask an LLM-powered agent to do a task, it is stateless. It is like asking it to work with Bob today. However, every time it talks to Bob, it has mysteriously forgotten everything about Bob and what it has worked on. It must reread the "Book of Bob" before every interaction to remind itself who Bob is, what he does, and what has happened so far.

HotBob aims to give the model a genuine **hot working memory** of its Bob: memory deeply integrated into the neural architecture and readily available as part of inference, so it no longer has to recover operational state only by rereading prompt text.

It is a working title. No further questions.

## What Exists

HotBob currently includes:

- synthetic task traces for secrets, symbol bindings, standing orders, scope isolation, and expiry
- typed memory operations: `WRITE`, `UPDATE`, `DELETE`, `NOOP`
- memory metadata: type, scope, privacy, authority, slot, strength
- a symbolic oracle baseline
- a tensor `MemoryBank`
- learned `MemoryRead` cross-attention over memory slots
- learned `MemoryWrite` heads for write decisions and value vectors
- a tiny transformer action classifier
- sequential event evaluation
- prompt hygiene tests
- held-out train/eval generation
- memory-controller and retrieval metrics

The model predicts categorical actions, not prose. This keeps failures visible.

## Architecture

The core loop is:

1. Events arrive one at a time.
2. Boundary events may trigger a memory write/update/delete.
3. Memory is stored as typed tensor slots, not appended prose.
4. Later action events are evaluated with only the final event text plus neural memory.
5. The model reads memory through cross-attention and predicts an action.

The memory bank stores:

```text
vectors:       [batch, slots, d_model]
occupied:      [batch, slots]
strength:      [batch, slots]
type_ids:      [batch, slots]
scope_ids:     [batch, slots]
privacy_ids:   [batch, slots]
authority_ids: [batch, slots]
```

## Evaluation

Evaluation compares:

- symbolic oracle memory
- context-only neural model
- teacher-forced memory
- predicted-write memory
- sequential teacher-forced memory
- sequential predicted memory

Metrics include:

- action accuracy by task family
- memory-required aggregate accuracy
- secret leak failures
- wrong-scope retrieval failures
- expiry failures
- write-head accuracies
- boundary write precision/recall/F1
- target-slot read attention mass

Recent cleaner held-out runs show the intended gap: teacher-forced memory beats context-only when prompts are hygienic, while predicted memory still needs better write/value/retrieval training.

## Commands

Generate one trace file:

```bash
uv run python -m hotbob.data.generate --n 1000 --out data/traces.jsonl
```

Generate held-out train/eval splits:

```bash
uv run python -m hotbob.data.generate \
  --train-out data/train.jsonl \
  --eval-out data/eval.jsonl \
  --train-n 1000 \
  --eval-n 250 \
  --seed 3
```

Run tests:

```bash
uv run pytest
```

Smoke train:

```bash
uv run python -m hotbob.training.train --traces data/train.jsonl --steps 50 --smoke
```

Evaluate:

```bash
uv run python -m hotbob.training.evaluate --checkpoint runs/latest.pt --traces data/eval.jsonl
```

## Current Challenges

The harness now exposes the real bottleneck:

- correct memory is useful
- context-only prompts are no longer allowed to leak the answer
- teacher-forced memory retrieves strongly
- predicted memory is still weak

Next work:

- improve value-vector supervision
- add contrastive retrieval objectives
- improve scope representation beyond flat one-off IDs
- add oracle ablations for scope/value/slot
- train the sequential memory controller more directly
- keep hardening synthetic data so final prompts genuinely require memory
