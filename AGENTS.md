# HotBob Operating Rules

- `master` is the last-known-good branch.
- Any core memory mechanism change starts on an experiment branch named `experiment/<short-topic>`.
- Experiment branches may contain messy commits and temporary instrumentation.
- Before merging to `master`, decide explicitly which behavior is hard-coded core behavior, exposed through CLI/config, retained only as experiment code, or discarded.
- Do not commit generated datasets, checkpoints, cloud logs, secrets, or `.env` files.
- Any change affecting training/eval speed must include either a test or a documented benchmark command.
- Hidden/private memory values must never be injected into prompts.
- Normal verification before merge is `uv run pytest`.
