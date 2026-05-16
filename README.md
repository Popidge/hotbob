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

- hygienic synthetic task traces for secrets, symbol bindings, standing orders, scope isolation, and expiry
- typed memory operations: `WRITE`, `UPDATE`, `DELETE`, `NOOP`
- memory metadata: type, scope, privacy, authority, slot, strength
- a symbolic oracle baseline
- a tensor `MemoryBank`
- learned `MemoryRead` cross-attention over memory slots
- learned `MemoryWrite` heads for write decisions, metadata, value vectors, and value-class probes
- an explicit action readout that fuses prompt state, retrieved memory context, interaction features, and distance features
- sequential event evaluation
- sequential memory-controller training
- prompt hygiene tests
- held-out train/eval generation
- contrastive retrieval training objectives
- active-memory debug dumps with private-value redaction
- memory-controller, retrieval, diagnostics, and audit metrics

The model predicts categorical actions, not prose. This keeps failures visible.

## Architecture

The core loop is:

1. Events arrive one at a time.
2. Boundary events may trigger a memory write/update/delete.
3. Memory is stored as typed tensor slots, not appended prose.
4. Later action events are evaluated with only the final event text plus neural memory.
5. The model reads memory through cross-attention.
6. The action readout sees the final hidden state, retrieved memory context, and explicit fusion features.

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

The memory write controller predicts:

- operation: `NOOP`, `WRITE`, `UPDATE`, `DELETE`
- slot
- type
- normalized scope
- privacy
- authority
- value vector
- value class probe
- write gate

The value class probe is an auditing surface: it helps inspect what a memory vector appears to encode without turning memory back into prompt text.

## Evaluation

Evaluation compares:

- symbolic oracle memory
- context-only neural model
- teacher-forced memory
- predicted-write memory
- sequential teacher-forced memory
- sequential predicted memory
- sequential predicted memory with oracle slot/scope/value ablations

Metrics include:

- action accuracy by task family
- memory-required aggregate accuracy
- task-family diagnostics for standing orders, hidden colours, and expiry
- secret leak failures
- wrong-scope retrieval failures
- expiry failures
- write-head accuracies
- value-class probe accuracy
- boundary write precision/recall/F1
- target-slot read attention mass
- oracle ablation scores for slot/scope/value failures
- active-memory JSONL debug dumps

Current held-out runs show the intended memory signal: context-only drops on memory-required prompts, teacher-forced memory improves performance, and sequential predicted memory retrieves strongly after scope normalization, contrastive retrieval training, and direct sequential-controller supervision.

The current bottleneck is no longer basic memory storage, scoping, or retrieval. The remaining failures are mostly in converting retrieved memory into the right policy/action semantics for harder families such as standing orders and active expiry.

### Latest Sit-Rep

A 1,000-step smoke-sized run on 10,000 train traces and 1,000 held-out eval traces produced:

```text
memory-required aggregate:
  symbolic oracle:        1.000
  context-only:           0.290
  teacher-forced memory:  0.803
  sequential predicted:   0.803

sequential predicted memory:
  boundary F1:            1.000
  target-slot read mass:  0.897
  op/scope/slot/type/privacy/authority heads: ~1.000
  value-class probe:      0.957
```

Task-family results:

```text
hidden_colour:     1.000
scope_isolation:   1.000
symbol_binding:    1.000
expiry:            0.730
standing_order:    0.285
```

Interpretation:

- the model can learn when to write memory
- it can assign typed metadata, scope, and slots reliably
- it can retrieve the right active slot later without prompt leakage
- predicted sequential memory now matches teacher-forced memory on aggregate
- oracle slot/scope/value ablations no longer improve the predicted-memory score on the latest run
- the remaining weakness is using the retrieved vector as an executable policy representation

For the larger goal, this is a useful inflection point. HotBob is now less a test of whether a neural working-memory layer can store and retrieve scoped facts, and more a test of how that memory should be represented, decoded, and consumed by an agent policy. That is the relevant bridge toward an agentic LLM: memory should not merely be recallable; it must become operational state that reliably conditions tool choice, refusal, prioritisation, and task execution.

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

The harness now exposes a narrower bottleneck:

- correct memory is useful
- context-only prompts are no longer allowed to leak the answer
- predicted and teacher-forced memory retrieve strongly on the current synthetic setup
- the memory controller can classify write boundaries, scopes, slots, metadata, and value classes reliably
- the remaining failures concentrate where a retrieved memory value must be interpreted as policy, not just recalled as a fact

Next work:

- strengthen value-to-policy/action decoding for standing orders and active expiry
- test richer memory value representations than mean token embeddings
- add typed action heads or type-conditioned policy decoders
- evaluate whether memory vectors should carry structured latent fields, not only dense values
- introduce longer multi-turn agent traces with tool calls, interruptions, stale state, and authority conflicts
- preserve the key constraint: final prompts must require memory without smuggling the answer back through text

## Direction Toward Agentic LLMs

The long-term target is a neural working-memory layer that can sit beside or inside an agentic LLM loop.

HotBob deliberately tests the hard part in miniature:

- operational state is written at event boundaries
- state has type, scope, authority, privacy, and expiry metadata
- later decisions are made without replaying the full prompt history
- memory can be inspected and audited without converting it back into user-visible prose

To become relevant to real LLM agents, the next research step is to move from categorical synthetic actions toward policy conditioning:

- memory-conditioned tool routing
- memory-conditioned refusal and disclosure behavior
- scoped project/session state
- user/tool authority conflict handling
- stale-memory expiry and replacement
- compact memory reads that condition generation or planning without becoming prompt stuffing

The current result supports the core premise that scoped typed neural memory can be learned and retrieved. It does not yet prove that the retrieved memory is a sufficient substrate for robust agent policy. That is now the main research frontier.
