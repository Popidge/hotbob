# HotBob

HotBob is a small research prototype for a Scoped Typed Working Memory Transformer.

Central claim:

> Prompt context is a poor substrate for task-local operational state. HotBob tests whether a scoped, typed neural working-memory layer can preserve and retrieve operational facts across task boundaries without representing that memory only as prompt text.

The first milestone is intentionally small: synthetic traces, explicit memory-operation labels, a symbolic memory baseline, and tensor memory-bank scaffolding. It uses action classification instead of language generation so failures remain visible.

## Current Prototype

HotBob now has a minimal neural loop:

- a write-boundary pass predicts typed memory write metadata and a value vector
- a final-action pass sees only the final event tokens plus tensor memory
- teacher-forced memory uses labelled memory ops as tensor slots, not prompt text
- predicted-write evaluation writes the model's own value vector into memory before action classification
- evaluation compares symbolic, context-only, teacher-forced memory, and predicted-write modes

## Generate Traces

```bash
uv run python -m hotbob.data.generate --n 1000 --out data/traces.jsonl
```

## Run Tests

```bash
uv run pytest
```

## Smoke Training

```bash
uv run python -m hotbob.training.train --traces data/traces.jsonl --steps 50 --smoke
```

## Current Limitations

This is still a research prototype. Scope prediction is difficult with many one-off synthetic scope IDs, the model is tiny, and predicted-write evaluation is one write boundary rather than a full multi-event recurrent loop. The next architectural step is sequential event processing where predicted writes are applied after each boundary and later actions read the accumulated working memory.
