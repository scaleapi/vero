# SWE-bench-Pro (scaffold)

This benchmark optimizes a code-editing agent on SWE-bench-Pro tasks. Each task
checks a real project out at `/app/repo` and describes a bug fix or feature. The
editable agent (`baseline/target/src/swebench_pro_agent/agent.py`) edits the
repository in place; the **task-source verifier** applies the resulting repo state
and runs the task's hidden test suite for the reward. The agent does not
self-grade and writes no answer file: reward comes entirely from the suite.

## Status: this is a SCAFFOLD, not a runnable benchmark

Everything except the actual task data is in place: a robust Harbor-native agent,
a `build.yaml`, partition scaffolding, a split script, and this README. It will
**not run yet** because the real SWE-bench-Pro task dataset must come from a Harbor
task source that is not committed here.

### DONE (working skeleton)

- `baseline/target/src/swebench_pro_agent/agent.py`: a tool-using code-editing
  agent on the OpenAI Responses API (`run_shell`, `read_file`, `write_file`,
  `apply_patch`, `run_tests`, `submit`), `MAX_TURNS = 50`. The
  `self._client.responses.create(...)` call is wrapped in a retry-with-backoff
  helper from the start (the GAIA baseline scored 0.0 because an unguarded call
  crashed on a transient error; that retry was the optimizer's winning fix).
- `baseline/build.yaml`: the full VeRO/Harbor optimization config (agent import
  path, inference gateway, partitions wiring, timeouts, secrets, W&B).
- `baseline/target/pyproject.toml`, smoke test, `secrets.env.example`, `.gitignore`.
- `scripts/partition_swe_bench_pro.py`: a deterministic, sha256-keyed stratified
  split with a `--check` mode, modeled on `gaia/scripts/partition_gaia.py`.

### STUBBED (must be completed before it runs)

Two things must be finished:

1. **Pin the real `task_source`.** `baseline/build.yaml` carries a clearly-marked
   placeholder:

   ```
   task_source: scale-ai/swe-bench-pro@sha256:REPLACE_WITH_REAL_DIGEST
   ```

   SWE-bench-Pro is a Scale benchmark. Find the real pinned task package in the
   Harbor task registry (ask the DEX-harness / Harbor team if it is not obvious).
   The baseline in this directory provides only the **agent + config**; the task
   package provides the **repos, tests, gold patches, and grading**. Replace the
   placeholder digest here and the `DATASET_DIGEST` constant in
   `scripts/partition_swe_bench_pro.py`.

2. **Regenerate the partitions from the real task list.** The committed
   `partitions/{development,validation,test}.json` are empty placeholders and
   `partitions/manifest.json` is a stub (`"_stub": true`). Once the task source is
   pinned, set `TOTAL_TASKS`/`TARGET_COUNTS` and confirm `_read_tasks` against the
   real `task.toml` layout in `scripts/partition_swe_bench_pro.py`, then generate
   and verify the split:

   ```bash
   uv run --python 3.12 \
     harness-engineering-bench/swe-bench-pro/scripts/partition_swe_bench_pro.py \
     --tasks-dir /path/to/exported/swe-bench-pro \
     --fetch-registry

   uv run --python 3.12 \
     harness-engineering-bench/swe-bench-pro/scripts/partition_swe_bench_pro.py \
     --tasks-dir /path/to/exported/swe-bench-pro \
     --check
   ```

   Also update the placeholder `total_cases` under `agent_access` in
   `baseline/build.yaml` to real disclosure counts, and run `uv lock` in
   `baseline/target/` so the compiled task pins an exact resolution.

## Split design

The committed split is intended to be deterministic and stratified by source
repository (so each represented codebase appears across all three partitions):

- development: 20% (full result disclosure to the optimizer)
- validation: 40% (aggregate-only; used to select candidates)
- test: 40% (held out until Harbor grades the completed outer task)

Exact counts are TODO until the task source is pinned.

## Build the outer Harbor task

From the repository root:

```bash
cd vero
VERO_SKIP_SECRET_CHECK=1 uv run vero harbor build \
  --config ../harness-engineering-bench/swe-bench-pro/baseline/build.yaml \
  --output ../harness-engineering-bench/swe-bench-pro/baseline/compiled
```

Omit `VERO_SKIP_SECRET_CHECK=1` for a real build so VeRO verifies that the OpenAI
and Modal credentials declared in `build.yaml` are present. Set `OPENAI_BASE_URL`
to your OpenAI-compatible endpoint. The `compiled/` directory is generated and
intentionally ignored. (This build will fail to resolve tasks until the
placeholder `task_source` above is replaced with the real digest.)

## Run the optimization

`vero harbor run` compiles the `build.yaml` to a temporary task, wires credentials
(relocating the upstream inference key to the gateway and handing the optimizer a
scoped producer token), and invokes Harbor. Copy the secrets template first:

```bash
cp harness-engineering-bench/swe-bench-pro/baseline/secrets.env.example \
   harness-engineering-bench/swe-bench-pro/baseline/secrets.env   # then edit it

cd vero
uv run vero harbor run \
  --config ../harness-engineering-bench/swe-bench-pro/baseline/build.yaml \
  --env-file ../harness-engineering-bench/swe-bench-pro/baseline/secrets.env \
  --environment modal \
  --agent codex \
  --model gpt-5.3-codex \
  --yes \
  -o ../runs/swe-bench-pro-full/jobs
```

`--env-file` values override the ambient shell and are passed only through the
subprocess environment, never on the command line. Anything after the known
options (here `--yes -o ...`) is forwarded verbatim to `harbor run`.

The producer scope's allow-list in `build.yaml` is `gpt-5.3-codex`, so pass
`--model gpt-5.3-codex` (or an aligned model) to keep the optimizer's model and
the gateway allow-list in lockstep and avoid a 403 mismatch. The evaluation scope
is pinned to `gpt-4o` (matching `model: openai/gpt-4o`).

**Modal endpoint requirement:** `--environment modal` runs eval sandboxes on
Modal, which must be able to reach the model endpoint in `OPENAI_BASE_URL`. Use a
Modal-reachable endpoint (for example the Azure
`https://<resource>.openai.azure.com/openai/v1` endpoint), **not** a VPN-internal
host that only resolves on the Scale network. Use `--environment docker` to run
the optimizer locally against Modal eval backends instead.
