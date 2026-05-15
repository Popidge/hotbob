# HotBob

HotBob is a small research prototype for a Scoped Typed Working Memory Transformer.

Central claim:

> Prompt context is a poor substrate for task-local operational state. HotBob tests whether a scoped, typed neural working-memory layer can preserve and retrieve operational facts across task boundaries without representing that memory only as prompt text.

The first milestone is intentionally small: synthetic traces, explicit memory-operation labels, a symbolic memory baseline, and tensor memory-bank scaffolding. It uses action classification instead of language generation so failures remain visible.

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

The transformer training path is only scaffolded. The current working code proves the trace schema, five synthetic task families, symbolic memory behavior, and tensor memory masking/update primitives.
