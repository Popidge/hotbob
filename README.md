# HotBob

HotBob is a research prototype for **scoped, typed neural working memory** in
agentic model loops.

The core claim is narrow:

> Prompt context is a poor substrate for task-local operational state. HotBob
> tests whether small scoped, typed memory objects can preserve operational facts
> across task boundaries and directly condition policy, action, tool choice, and
> generation without representing that memory only as prompt text.

HotBob is not a production LLM and it is not a generic long-context replacement.
It is a compact lab for asking how memory should be written, scoped, audited,
retrieved, and consumed in the hot inference path.

## Why The Name?

When an LLM-powered agent is stateless, it has to reread the "Book of Bob" before
every interaction. HotBob is the working title for giving the model a hot working
memory of its Bob: operational state that is immediately available during
inference instead of being reconstructed only from prompt history.

## Research Lane

HotBob memory objects are deliberately small and governed. A memory is not just a
blob of old text. It has:

- type
- scope
- privacy
- authority
- lifecycle/expiry
- active/inactive state
- strength
- operational effect on action or generation

This makes HotBob different from broad long-context memory projects. The target
is working memory for agent behavior: scoped project state, standing orders,
private facts, expiry, authority conflicts, tool routing, and refusal/disclosure
decisions.

## What Exists

The repo currently includes two experimental tracks.

The compact synthetic-controller path includes:

- hygienic synthetic traces for secrets, symbol bindings, standing orders, scope
  isolation, and expiry
- typed memory operations: `WRITE`, `UPDATE`, `DELETE`, `NOOP`
- tensor `MemoryBank` storage with vectors, occupied flags, strength, type,
  scope, privacy, and authority
- learned `MemoryRead` cross-attention over memory slots
- learned `MemoryWrite` heads for operation, slot, metadata, value vector, and
  value-class probes
- explicit action readout over prompt state, retrieved memory context,
  interaction features, and distance features
- sequential teacher-forced and predicted-memory evaluation
- contrastive retrieval objectives
- active-memory JSONL debug dumps with private-value redaction
- symbolic oracle, context-only, and neural memory baselines

The real-LLM path in `src/hotbob/llm` wraps `Qwen/Qwen2.5-0.5B-Instruct` and
keeps the base decoder frozen by default. It supports:

- `prefix`: learned soft-prefix conditioning from active HotBob memory
- `attention_q`: low-rank query-side memory correction
- `attention_o`: low-rank output-side memory correction
- `attention_qo`: combined query/output correction
- `shared` memory readout state
- `by_type` scaffolding for typed multi-state memory readout
- context-only, teacher-forced memory, and predicted-memory evaluation modes
- score-based closed-set evaluation by default, with autoregressive generation
  still available

Memory values are not inserted into the prompt. The LLM adapter consumes
`MemoryBank` tensors and metadata.

## Typed Memory Families Experiment

The `experiment/typed-memory-families` branch changes the synthetic memory
distribution from mostly string-valued traces to first-class structured payloads
on `MemoryOp`. The goal is data and target quality: expose policy, authority,
expiry, disclosure, stale-state replacement, and tool-routing structure to the
existing prefix/q/o comparison harness without adding native transformer memory
gates yet.

The rich families are:

- `standing_order`
- `active_expiry`
- `authority_conflict`
- `tool_verified_override`
- `interrupted_task`
- `stale_state_replacement`
- `privacy_disclosure_conflict`
- `multi_step_tool_routing`

Example standing-order payload:

```yaml
kind: standing_order
default_action: hold_fire
trigger: hostile_posture
allowed_responses: [raise_shields, evade, hail]
forbidden_responses: [fire_weapons]
exceptions: [civilians_at_risk]
authority_level: captain
expiry_policy: mission_end
```

Smoke run for the rich LLM path:

```bash
uv run python -m hotbob.llm.generate_data \
  --train-out data/llm_rich_train_smoke.jsonl \
  --eval-out data/llm_rich_eval_smoke.jsonl \
  --train-n 100 \
  --eval-n 40 \
  --seed 7

uv run python -m hotbob.llm.train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --traces data/llm_rich_train_smoke.jsonl \
  --steps 10 \
  --batch-size 1 \
  --integration-mode prefix \
  --freeze-base \
  --structured-loss-weight 0.2 \
  --out runs/qwen_memory/rich_smoke.pt

uv run python -m hotbob.llm.evaluate \
  --checkpoint runs/qwen_memory/rich_smoke.pt \
  --traces data/llm_rich_eval_smoke.jsonl \
  --limit 20 \
  --mode teacher_forced \
  --decode-strategy score_answers
```

Expected Modal/T4 comparison shape:

```bash
uv run python -m hotbob.llm.architecture_compare \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train-traces data/llm_rich_train.jsonl \
  --eval-traces data/llm_rich_eval.jsonl \
  --steps 10000 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --eval-modes teacher_forced \
  --decode-strategy score_answers \
  --structured-loss-weight 0.2 \
  --variant prefix \
  --variant attention_q:last4 \
  --variant attention_o:last4 \
  --variant attention_qo:last4 \
  --variant attention_qo:by_type:last4
```

## Prefix+LoRA Memory Reader Experiment

The `experiment/prefix-lora-memory-reader` branch tests whether a small PEFT
LoRA adapter on Qwen can learn to consume HotBob's structured memory-prefix
embeddings better than frozen Qwen. The memory controller, prefix adapter, write
supervision, and answer loss stay intact; training optimizes HotBob memory heads
plus Qwen LoRA weights while the base Qwen weights remain frozen.

PEFT is the first backend because it is the clean reference implementation for
the current custom loop, which already trains through `inputs_embeds` with
memory-prefix attachment and write supervision. Unsloth is intentionally left as
a later backend once the PEFT baseline is proven.

Generate the comparison data:

```bash
uv run python -m hotbob.llm.generate_data \
  --train-out data/llm_rich_lora_train_50000.jsonl \
  --eval-out data/llm_rich_lora_eval_5000.jsonl \
  --train-n 50000 \
  --eval-n 5000 \
  --seed 7
```

Train the frozen-prefix baseline:

```bash
uv run python -m hotbob.llm.train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --traces data/llm_rich_lora_train_50000.jsonl \
  --steps 10000 \
  --batch-size 1 \
  --integration-mode prefix \
  --freeze-base \
  --structured-loss-weight 0.2 \
  --out runs/qwen_memory/prefix_frozen_50k_10k.pt
```

Train prefix+LoRA:

```bash
uv run python -m hotbob.llm.train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --traces data/llm_rich_lora_train_50000.jsonl \
  --steps 10000 \
  --batch-size 1 \
  --integration-mode prefix \
  --freeze-base \
  --structured-loss-weight 0.2 \
  --lora-backend peft \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj k_proj v_proj o_proj \
  --memory-lr 1e-4 \
  --lora-lr 2e-4 \
  --out runs/qwen_memory/prefix_lora_50k_10k.pt
```

For a stricter contamination check, train the memory controller and LoRA on
separate data. First train or reuse a frozen-prefix memory-controller checkpoint,
then train LoRA on a different trace file while loading and freezing those
memory heads:

```bash
uv run python -m hotbob.llm.train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --traces data/llm_rich_lora_train_lora.jsonl \
  --steps 10000 \
  --batch-size 1 \
  --integration-mode prefix \
  --freeze-base \
  --memory-heads-checkpoint runs/qwen_memory/prefix_memory_controller.pt \
  --freeze-memory-heads \
  --write-loss-weight 0 \
  --lora-backend peft \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj k_proj v_proj o_proj \
  --lora-lr 2e-4 \
  --out runs/qwen_memory/prefix_lora_split_10k.pt
```

Compare aggregate behavior:

```bash
uv run python -m hotbob.llm.compare \
  --traces data/llm_rich_lora_eval_5000.jsonl \
  --checkpoints \
    prefix_frozen=runs/qwen_memory/prefix_frozen_50k_10k.pt \
    prefix_lora=runs/qwen_memory/prefix_lora_50k_10k.pt \
  --modes predicted context_only teacher_forced \
  --decode-strategy score_answers \
  --batch-size 1 \
  --out runs/qwen_memory/prefix_lora_comparison.json
```

`prefix_lora / context_only` is the memory-ablated result: the same LoRA
checkpoint is active, but memory is reset/unused and evaluated with
`use_memory=False`. This distinguishes memory use from prompt-pattern learning.

Generate authority-focused reports:

```bash
uv run python -m hotbob.llm.authority_report \
  --checkpoint runs/qwen_memory/prefix_lora_50k_10k.pt \
  --traces data/llm_rich_lora_eval_5000.jsonl \
  --mode predicted \
  --decode-strategy score_answers \
  --out runs/qwen_memory/prefix_lora_authority_predicted.json

uv run python -m hotbob.llm.authority_report \
  --checkpoint runs/qwen_memory/prefix_frozen_50k_10k.pt \
  --traces data/llm_rich_lora_eval_5000.jsonl \
  --mode predicted \
  --decode-strategy score_answers \
  --out runs/qwen_memory/prefix_frozen_authority_predicted.json
```

Success criteria:

- teacher-forced authority conflict accuracy `>= 0.65`
- predicted authority conflict accuracy `>= 0.40`
- `secret_leak_failures == 0`
- `prefix_lora / context_only` remains materially lower than
  `prefix_lora / predicted`

## Authority Memory Reader Follow-Up

The `experiment/authority-memory-reader` branch is now a controller-first
authority probe. The frozen-Qwen and LoRA path can still be explored separately,
but this branch's core question is simpler: can the compact HotBob controller
write the right typed memories, retrieve the active scoped authority memory, and
produce the right action under predicted memory?

Generate authority-heavy controller traces:

```powershell
uv run python -m hotbob.data.generate --train-out data/controller_authority_train.jsonl --eval-out data/controller_authority_eval.jsonl --train-n 50000 --eval-n 5000 --seed 51 --family-weight authority_conflict=10 --family-weight tool_verified_override=3 --family-weight scope_isolation=1 --family-weight privacy_disclosure_conflict=1
```

Train the compact controller:

```powershell
uv run python -m hotbob.training.train --traces data/controller_authority_train.jsonl --steps 5000 --batch-size 32 --structured-loss-weight 0.3 --retrieval-contrastive-weight 0.25
```

Evaluate and emit the controller-only authority report:

```powershell
uv run python -m hotbob.training.evaluate --checkpoint runs/latest.pt --traces data/controller_authority_eval.jsonl --batch-size 64 --debug-families authority_conflict tool_verified_override --debug-dumps-out runs/controller_authority_debug.jsonl
uv run python -m hotbob.training.authority_report --checkpoint runs/latest.pt --traces data/controller_authority_eval.jsonl --out runs/controller_authority_report.json
```

`hotbob.training.authority_report` reports sequential `context_only`,
`teacher_forced`, `predicted`, and predicted oracle slot/scope/value ablations.
For each mode it highlights authority accuracy, authority probe accuracy
(`authority_conflict` plus `tool_verified_override`), target-slot read mass,
boundary F1, and write-head accuracies.

Success criteria for this branch are predicted authority materially above
context-only, predicted authority close to teacher-forced, oracle ablations that
identify any remaining write/read bottleneck, no secret leaks, and no major
regression on the mixed controller evaluation. The cloud workflow for this probe
lives in `controller_authority_colab.ipynb`.

## Delta-HotBob

`docs/delta_hotbob.md` describes the delta-HotBob experiment inspired by
`delta-mem`: use active scoped HotBob memory to steer a frozen decoder with
low-rank query/output corrections, then compare that against the existing prefix
adapter.

The current q/o implementation patches selected Qwen-style attention modules
ending in `self_attn.q_proj` and `self_attn.o_proj` with scoped forward pre-hooks.
That keeps the Hugging Face module signatures and `generate()` cache behavior
while injecting memory-conditioned low-rank corrections inside the attention
path. Fake or unsupported models fall back to a residual approximation for CPU
tests.

This does not claim native working memory or long-context replacement. It is a
first integration experiment: can typed, scoped operational memory condition a
frozen decoder more deeply than a prefix while preserving HotBob's audit surface?

## Recent LLM Results

The most useful current prefix result used a larger synthetic LLM corpus:

```bash
uv run python -m hotbob.llm.generate_data \
  --train-out data/llm_train_50k.jsonl \
  --eval-out data/llm_eval_5k.jsonl \
  --train-n 50000 \
  --eval-n 5000 \
  --seed 41

uv run python -m hotbob.llm.train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --traces data/llm_train_50k.jsonl \
  --steps 10000 \
  --batch-size 1 \
  --integration-mode prefix \
  --freeze-base \
  --out runs/qwen_memory/prefix_50k_10k.pt
```

The 10k-step frozen-Qwen prefix run completed in about 25 minutes on the local
CUDA setup. Loss kept improving past the earlier 1k-step check:

```text
final loss: 0.1024
mean loss:  0.2958

00001-01000: mean=0.7446
01001-02000: mean=0.3680
02001-03000: mean=0.2901
03001-04000: mean=0.3045
04001-05000: mean=0.2336
05001-06000: mean=0.2220
06001-07000: mean=0.1919
07001-08000: mean=0.2175
08001-09000: mean=0.1877
09001-10000: mean=0.1978
```

Held-out 5k score-based eval:

```text
context-only:
  action accuracy:       0.3152
  scope_isolation:       0.519
  hidden_colour:         0.337
  standing_order:        0.239
  expiry:                0.000
  symbol_binding:        0.481

teacher-forced memory:
  action accuracy:       0.961
  scope_isolation:       1.000
  hidden_colour:         0.805
  standing_order:        1.000
  expiry:                1.000
  symbol_binding:        1.000
  secret leak failures:  0

predicted memory:
  action accuracy:       0.484
  scope_isolation:       0.000
  hidden_colour:         0.676
  standing_order:        1.000
  expiry:                0.744
  symbol_binding:        0.000
  secret leak failures:  0
```

Interpretation:

- the final prompt alone still underperforms, so the task is not solved by prompt
  leakage
- a frozen Qwen decoder can use dense scoped HotBob memory through a learned
  prefix
- longer prefix training has real headroom on the current synthetic distribution
- predicted-memory evaluation is currently bottlenecked by upstream memory
  prediction/scope selection, not only by the decoder adapter
- q/o correction is a deeper integration path, but prefix remains a strong
  baseline and should continue to run in parallel

## Commands

Install and test:

```bash
uv sync
uv run pytest
```

## Experiment Workflow

Project operating rules live in `AGENTS.md`. In short, keep `master` as the
last-known-good branch, start core memory-mechanism changes on
`experiment/<short-topic>` branches, do not commit generated data/checkpoints,
and verify merges with `uv run pytest`.

Generate synthetic controller traces:

```bash
uv run python -m hotbob.data.generate \
  --train-out data/train.jsonl \
  --eval-out data/eval.jsonl \
  --train-n 1000 \
  --eval-n 250 \
  --seed 3
```

Train and evaluate the compact neural memory controller:

```bash
uv run python -m hotbob.training.train \
  --traces data/train.jsonl \
  --steps 1000

uv run python -m hotbob.training.evaluate \
  --checkpoint runs/latest.pt \
  --traces data/eval.jsonl
```

Generate LLM traces:

```bash
uv run python -m hotbob.llm.generate_data \
  --train-out data/llm_train.jsonl \
  --eval-out data/llm_eval.jsonl \
  --train-n 10000 \
  --eval-n 1000 \
  --seed 3
```

Train frozen-Qwen prefix memory:

```bash
uv run python -m hotbob.llm.train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --traces data/llm_train.jsonl \
  --steps 1000 \
  --batch-size 1 \
  --integration-mode prefix \
  --freeze-base
```

Small performance guard run:

```bash
uv run python -m hotbob.llm.generate_data \
  --train-out data/llm_train.jsonl \
  --eval-out data/llm_eval.jsonl \
  --train-n 10000 \
  --eval-n 1000 \
  --seed 3

uv run pytest

uv run python -m hotbob.llm.train \
  --traces data/llm_train.jsonl \
  --steps 10 \
  --batch-size 2 \
  --freeze-base

uv run python -m hotbob.llm.evaluate \
  --traces data/llm_eval.jsonl \
  --limit 20 \
  --mode all \
  --decode-strategy score_answers

uv run python -m hotbob.llm.compare \
  --traces data/llm_eval.jsonl \
  --mode teacher_forced \
  --decode-strategy score_answers
```

Run the explicit memory-architecture comparison harness:

```bash
uv run python -m hotbob.llm.architecture_compare \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train-traces data/llm_train.jsonl \
  --eval-traces data/llm_eval.jsonl \
  --steps 1000 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --eval-modes teacher_forced \
  --variant prefix \
  --variant attention_q:last4 \
  --variant attention_o:last4 \
  --variant attention_qo:last4
```

The harness trains every variant with the same frozen model, data, step count,
batch size, correction rank, and write-loss weight. It writes one checkpoint per
variant under `runs/qwen_memory/architecture_compare/checkpoints/` plus
`comparison.json` containing loss buckets, final/mean loss, aggregate accuracy,
by-family accuracy, and leakage/scope/expiry failure counts. Variants use:

```text
--variant mode[:layers]
--variant mode:by_type[:layers]
```

For example, add typed q/o layer-routing ablations with:

```bash
--variant attention_qo:by_type:last1 \
--variant attention_qo:by_type:last4 \
--variant attention_qo:by_type:all
```

Single-run frozen-Qwen q/o correction training is still available:

```bash
uv run python -m hotbob.llm.train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --traces data/llm_train.jsonl \
  --steps 1000 \
  --batch-size 1 \
  --integration-mode attention_qo \
  --attention-patch-layers last4 \
  --freeze-base
```

Evaluate one or all LLM modes:

```bash
uv run python -m hotbob.llm.evaluate \
  --checkpoint runs/qwen_memory/latest.pt \
  --traces data/llm_eval.jsonl

uv run python -m hotbob.llm.evaluate \
  --checkpoint runs/qwen_memory/latest.pt \
  --traces data/llm_eval.jsonl \
  --mode teacher_forced
```

Useful LLM flags:

- `--integration-mode prefix|attention_q|attention_o|attention_qo`
- `--memory-state-mode shared|by_type`
- `--correction-rank 16`
- `--attention-patch-layers all|last4|0,1,2`
- `--mode all|context_only|teacher_forced|predicted`
- `--decode-strategy score_answers|generate`
- `hotbob.llm.compare` evaluates named checkpoints through the same result path
- `hotbob.llm.architecture_compare` trains and evaluates a matched variant matrix

For private or rate-limited Hugging Face downloads, create a local `.env` file:

```text
HF_TOKEN=your_hugging_face_token
HUGGING_FACE_HUB_TOKEN=your_hugging_face_token
```

`.env` is gitignored and should not be committed.

## Current Research Questions

The repo now exposes several concrete questions:

- Should working memory enter a decoder as prefix tokens, attention corrections,
  or a deeper native memory pathway?
- How should typed/scoped memories be grouped to reduce interference?
- What metadata should be embedded into the memory readout path?
- How should standing orders and expiry be represented so they become policy,
  not just facts?
- How much of predicted-memory failure is write/value/scope prediction versus
  decoder consumption?
- Can q/o-style correction overtake prefix once trained and tuned with the same
  data budget?

Near-term work:

- use the comparison harness for prefix, q-only, o-only, and q+o layer/type
  ablations
- profile and batch the slow eval paths
- improve predicted memory scope/value selection
- add corrupted-memory and wrong-scope-memory ablations
- expand trace diversity around authority, privacy, stale state, and tool use
- keep hidden memory out of prompts

## Attribution

The q/o correction experiment is architecturally inspired by:

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

HotBob uses delta-mem as an architectural and methodological reference, not as a
change in research lane. HotBob remains focused on scoped, typed, auditable
operational working memory.

## License And Contributions

HotBob is licensed under the Apache License 2.0. The project is intended to be
easy to fork, port, cite, extend, or repurpose.

Useful contributions include reproductions, negative results, new decoder
adapters, better write controllers, value encoders, retrieval losses,
evaluators, docs, examples, and cleanup.
