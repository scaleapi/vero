---
name: run-benchmark
description: >-
  Run one harness-engineering-bench optimization end-to-end (compile → inference
  gateway → optimizer agent → sandboxed evals → finalize on held-out test),
  including the preflight to confirm before launching and the health checks to
  verify after. Use when launching, reproducing, or debugging a benchmark run.
---

# Running a harness-engineering-bench optimization

This is a runbook for launching one benchmark's optimization run and confirming
it is healthy. It is written to be provider-agnostic: fill in your own inference
endpoint, Modal account, and W&B account. Treat every `<placeholder>` as
something you supply. **Never commit real keys, tokens, endpoints, or absolute
personal paths** — they belong only in a local, git-ignored `secrets.env`.

## What a run is

Each benchmark compiles from `harness-engineering-bench/<benchmark>/baseline/build.yaml`,
the single source of truth. `vero harbor run` compiles it into a Harbor task and
stands up three things: an **inference gateway** (holds the real upstream key,
enforces per-scope model allow-lists), an **evaluation sidecar** (owns the cases,
scoring, and final candidate selection), and the **optimizer agent** (a coding
agent that edits only `target/`, commits candidates, and scores them via the
`evals` CLI). When the optimizer finishes, the trusted verifier scores the
selected candidate on the held-out `test` partition and writes the final reward.

The optimizer never sees the real upstream key — it gets a scoped producer token
pointed at the gateway. Keep it that way.

## Prerequisites (confirm these exist first)

- The repo checkout containing both the `vero/` CLI package and
  `harness-engineering-bench/`. Run commands from the `vero/` subdirectory.
- `uv` installed (the CLI is invoked as `uv run vero ...`).
- **Docker** running — needed for `--environment docker` (the outer optimizer
  compose runs locally).
- A **Modal** account + tokens — the inner evaluation sandboxes run there by
  default (`environment_name: ${inner_env:-modal}`).
- A **Weights & Biases** account + API key (self-hosted or cloud) for telemetry.
- An **OpenAI-compatible inference endpoint** + key that can serve both your
  optimizer model and each benchmark's target model. The gateway proxies to it.
- A local `secrets.env` (copy from a benchmark's `secrets.env.example` where
  present). Required keys (names only — never values in any committed file):
  `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`,
  `WANDB_API_KEY`, `WANDB_BASE_URL`. A few benchmarks that run an in-container
  judge/user-sim also need `OPENAI_API_BASE` (set it equal to `OPENAI_BASE_URL`).
  See each benchmark's `baseline/README.md` and `CONFIGURATION.md`.

## Launch command (template)

```bash
cd <repo-root>/vero
uv run vero harbor run \
  --config ../harness-engineering-bench/<benchmark>/baseline/build.yaml \
  --env-file secrets.env \
  --environment docker \            # outer optimizer location (see tradeoff below)
  --agent <agent-name> \            # e.g. codex or claude-code (exact registry name)
  --model <optimizer-model> \       # must match the gateway producer allow-list
  --param wandb_run=<benchmark>__<optimizer-model> \
  --yes \
  -o ../runs/<benchmark>/<optimizer-label>/jobs
```

Notes that bite if you get them wrong:

- **`--agent` is the exact Harbor registry name**, not a friendly alias. If it is
  wrong you get `Agent name <x> is not valid. Valid agent names: {...}` at init
  (a fast, harmless crash — no containers start). The Claude Code agent is
  `claude-code` (not `claude`); the OpenAI coding agent is `codex`. If unsure,
  run with a deliberately bogus `--agent` once and read the valid-names list.
- **`--model` must be spelled exactly as the producer allow-list** in the
  build.yaml (`inference_gateway.producer.allowed_models`, usually
  `${optimizer_model:-<default>}`). `--model` is threaded into that placeholder,
  so the agent's model and the allow-list stay in lockstep; a mismatch is a
  gateway 403 on the optimizer's first request. If a coding agent rewrites the
  model string on the wire (some strip a provider prefix), pass the bare form
  that both the upstream serves and the allow-list expects.
- **`-o <dir>`** is forwarded to Harbor; use the systematic layout
  `runs/<benchmark>/<optimizer-label>/jobs`.
- **`--environment`** controls where the *optimizer* runs: `docker` = local,
  fully observable, but the run dies if the machine sleeps; `modal` = survives a
  local sleep/disconnect, less local visibility. Inner evals are on Modal either
  way (override with `--param inner_env=docker` only for a local shakedown).

## Preflight — confirm BEFORE launching

Do a dry compile and inspect the artifacts. This catches almost every
misconfiguration for free:

```bash
cd <repo-root>/vero
VERO_SKIP_SECRET_CHECK=1 uv run vero harbor build \
  --config ../harness-engineering-bench/<benchmark>/baseline/build.yaml \
  --output /tmp/precompile
```

Then check, in the compiled `/tmp/precompile`:

1. **`environment/sidecar/serve.json` → `backends`**: each partition backend has
   the `n_attempts` / `aggregate_attempts` you intend. If you want the noisy
   held-out finalize averaged over N, only the **test** backend should show
   `n_attempts: N` / `aggregate_attempts: mean`; development and validation stay
   at the global (usually 1). This is set per target in build.yaml
   (`targets[].n_attempts` / `aggregate_attempts`).
2. **`serve.json` → `wandb`**: `project`, `group`, `name` are what you expect.
3. **Model allow-lists** (`serve.json` and `environment/gateway/config.json`):
   the **evaluation** allow-list is the benchmark's target model; the
   **producer** allow-list is your optimizer model (it shows the build default
   here — at launch `--model` overrides it).
4. **`instruction.md`** (the optimizer's task prompt): read it end to end. The
   objective, the `evals run --backend ... --partition ...` command, and the
   exposed partitions should be right, and it must not advertise tools/skills/
   subagents that aren't actually shipped in the workspace.

Also confirm, out of band:

- **The optimizer model actually serves on your upstream** — send one tiny
  request to your inference endpoint with that exact model string. A model the
  upstream doesn't recognize 403/404s the optimizer immediately.
- **W&B auth works** and the project name is correct (a viewer query against your
  W&B host with the key).
- **Docker is up** (`docker info`) and **Modal tokens are valid**.
- **The target model and `baseline_reward`** pinned in build.yaml are the ones
  you mean to compare against (`CONFIGURATION.md` records the held-out baselines).

## Launch discipline

- **Launch detached** so the run outlives your shell/session. Redirect to a log
  and record the PID, e.g. `nohup bash launch.sh > run.log 2>&1 &` (or `setsid`;
  on macOS, which lacks `setsid`, a Python double-fork + `os.setsid()` daemonizer
  works). Keep `launch.sh`, `run.log`, `run.pid` alongside the `jobs/` output.
- Prefer to **confirm the launch** (of a full-budget run) before firing — these
  spend real target-model and optimizer tokens plus Modal compute.

## Post-launch health check — verify AFTER launching

Watch for a fast crash first (invalid agent name, missing secret, model 403,
Docker down), then confirm forward progress. A tail filtered for failure
signatures catches the crashes:

```bash
tail -f run.log | grep -E 'Traceback|Exception|Error|denied|40[0-9]|429|Killed|OOM|not valid|Cannot connect to the Docker'
```

After a few minutes (first build can be slow), confirm the good path:

- Outer process still alive; no traceback in `run.log`.
- The compose containers are up (gateway, sidecar, optimizer/agent).
- The optimizer is issuing **gateway-authorized** requests — no 403s in the
  gateway request log (a 403 storm = producer model ≠ allow-list).
- Inner eval sandboxes are dispatching (Modal auth is good) and `result.json`
  files begin appearing under the `jobs/` tree.
- A W&B run shows up under the expected project, not immediately failed.
- Provider **429s**: the seed agents retry transient rate limits, but sustained
  429s mean you are hitting a shared upstream quota — throttle concurrency.

If you see a clear failure signal, **stop the run and diagnose from disk** rather
than letting it burn budget. Detached/daemon-owned sandboxes can keep running
after the parent dies, so check for and clean up orphans when you kill a run.

## What "done / green" looks like

- **`finalize.json`** (admin volume) / **`harbor-finalization.json`** (session):
  `shipped: true`, a `rewards` map (keyed by the target's `reward_key`), and
  `baseline_rewards`. `shipped: false` means selection produced nothing.
- Session artifacts exported: a portable `experiment.html` report, the session
  archive, and the candidates repo.
- The W&B run is finished (not failed); its summary carries `shipped`.
- Compare the candidate's held-out reward against the pinned `baseline_reward`
  for that benchmark — an improvement is `shipped: true` with a higher reward.

## Cost awareness

A full run spends: target-model tokens on development + validation (bounded by
each partition's `total_cases`), the optimizer-model session, plus finalization
(`n_attempts × test_size` case-evaluations; the pinned baseline is not re-scored
when `score_baseline: false`). Read `budgets.json` in the session for actuals.

The reported unit is tokens, not dollars: the trusted per-evaluation split lands
in `reward_metrics` as `inference_input_tokens` / `inference_cached_input_tokens`
/ `inference_output_tokens` / `inference_total_tokens` (→ W&B). Because the target
model is fixed per benchmark, dollars are a downstream linear function of that
triple (per-model rate vector) and are not stored.

Read the per-case statistics, not just the totals: token and latency
distributions are heavy-tailed, so `reward_metrics` carries a mean, median, and
max per case (`mean/median/max_case_wall_seconds`,
`mean/median/max_case_agent_reported_*_tokens`, plus a derived
`mean_case_inference_*`). A mean far above the median means a few cases dominate
the spend; the max is the case most likely to hit its wall budget and score the
failure value. For per-trial breakdowns, run the post-hoc aggregator on the
exported session:

```bash
python harness-engineering-bench/scripts/per_trial_tokens.py <session-dir> --json
# or roll several runs into one flat table:
python .../per_trial_tokens.py <run1> <run2> ... --csv tokens.csv
```

Check its `coverage_pct` (should be ~100% and `residual` ~0 when the build sets
`inference_gateway.request_log_attribution: true`; low coverage means per-trial
numbers are lower bounds and the per-evaluation totals are the envelope).

## Per-benchmark gotchas

Don't hardcode assumptions — read `CONFIGURATION.md` and the benchmark's
`baseline/build.yaml` and `baseline/README.md`. Common differences: a benchmark
may pin a **different target model**; some run an **in-container judge or
user-simulator** on the real upstream (extra credentials + cost, and reduced
isolation); some pull **large prebuilt images or corpora** (long first build);
timeouts and partition sizes vary widely. When in doubt, dry-compile and read the
rendered `instruction.md` and `serve.json`.
