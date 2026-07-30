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

The optimizer itself gets a scoped producer token pointed at the gateway, not the
real upstream key. Keep it that way.

Two caveats, because this is easy to over-trust. Benchmarks that run an
in-container judge or user-simulator set `task_services_use_upstream`, which puts
the **raw upstream credential** into the evaluation sub-run; those benchmarks also
run the candidate harness without a separate user, so optimizer-authored code
shares that environment. And the per-scope model allow-list confines the target
model in the normal case but is not a hard guarantee — the optimizer holds the
producer token and writes the candidate. Both are known, deferred, and recorded
in the affected `build.yaml` files and in `vero/src/vero/gateway/inference.py`.
Treat the boundary as "an honest optimizer cannot reach the key by accident",
not "an adversarial one cannot reach it at all".

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
  local sleep/disconnect, less local visibility. **Inner evals must be on Modal.**
  `inner_env=docker` fails every case: the inner evaluation runs
  `harbor run -e docker` inside the sidecar, which has no docker CLI or socket, so
  harbor produces no trials and the evaluation returns 502.

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

- **Use `scripts/launch_cell.sh`** rather than assembling this by hand. It writes
  `launch.sh`, `run.log`, `run.pid` and `daemonize.py` into
  `runs/<benchmark>/<label>/`, double-forks into a new session, and — importantly
  — **fails non-zero if no live pid appears**, so a launch that did not start
  cannot be mistaken for one that did.

  ```bash
  bash harness-engineering-bench/scripts/launch_cell.sh \
    <benchmark>/<label> ../harness-engineering-bench/<benchmark>/baseline/build.yaml \
    <envfile> modal <agent> <launch-model> <wire-model> <benchmark>__<label>
  ```

  Positions 6 and 7 are the launch form and the wire form of the model, which
  differ for `opencode` and `kimi-cli` — see the routing table in
  `CONFIGURATION.md`.

- **Launch detached.** A plain `nohup ... &` is not enough: supervising harnesses
  terminate process groups belonging to an idle session, and macOS has no
  `setsid(1)`. The launcher's double-fork handles both.
- **Never wrap launches in `|| true` inside a loop.** It converts every failure
  into silence and the loop still exits 0, so you report N runs started when zero
  did. This has cost a full grid.
- Prefer to **confirm the launch** (of a full-budget run) before firing — these
  spend real target-model and optimizer tokens plus Modal compute.

## Watching a run while it is in flight

`agent/` and `verifier/` artifacts are written only when a run *ends*, and
`run.log` holds nothing but the outer progress spinner. So mid-run, W&B is the
only view of whether the optimizer is actually evaluating anything:

```bash
uv run --project vero python harness-engineering-bench/scripts/wandb_progress.py \
  <benchmark> <suffix>
```

It reports, per cell, evaluations completed, evaluations terminated, upstream
error counts and generated tokens. **`terminated` should be 0**; a non-zero value
means evaluations are being invalidated mid-search, which costs the optimizer
evidence without necessarily corrupting the reported score.

Two traps when checking liveness:

- **`ps aux | grep` truncates its command column to the terminal width** and will
  report 0 processes for perfectly healthy runs, because the match text sits past
  the cutoff. Use `pgrep -f` instead. Under a kill-on-failure policy this
  false negative is an instruction to destroy working runs.
- **Phase text is not liveness.** A status derived from the log will happily
  report a phase for a dead run. Compare `stat -f %z run.log` across a sleep.

## Concurrency

Runs share one upstream provider account, so they contend. Steer by measurement,
not arithmetic:

**Steer on the marginal error rate across a check interval** — Δerrors ÷ Δrequests
between two consecutive readings — and specifically on the rate imposed on the
runs that were *already* going when load changed. Measured on one officeqa pass:
4 concurrent cells sat at 0.15%; adding a fifth took the pass to 23.1% overall and
put 17.1% on the four incumbents. Steady state at 4 later fell to 0.00%.

**Do not derive a ceiling from tokens-per-minute against a published quota.** That
was attempted twice on the same pass and mispredicted in both directions — first
from anchoring elapsed time to process start rather than to when work began, then
by reading a coincidental match as confirmation. The marginal error rate is the
signal that held up.

**Spikes are usually startup bursts, not a standing ceiling.** New cells hit their
opening evaluation waves together. Wait a full interval after any load change
before concluding anything.

**Replace on completion rather than adding**, and **never kill a healthy run to
reduce load** — see below.

## Stopping a run is two operations

Killing the local process does **not** stop the work. The compose topology is
daemon-owned and Modal sandboxes are server-side (default 24h timeout), so a
killed run keeps executing and keeps billing — only the local collector is gone,
which makes the results unrecoverable. Killing therefore converts expensive useful
work into expensive useless work.

An intentional stop needs both halves; `scripts/cleanup_orphans.sh --dry-run`
handles the remote half, but read its header first — all runs share one Modal app,
so a bulk teardown mid-pass destroys healthy runs alongside orphans. Run it only
between passes.

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

### Copy the session out BEFORE you kill anything

Every commit the optimizer made — including the one it submitted — lives *only*
inside the sidecar container until the verifier's `export-session` step writes
`session.tar.gz`. **Kill a run before that step and the candidates are gone**,
because stopping the stack removes the containers. A run killed mid-verifier
leaves an empty `verifier/` directory and nothing to recover; the entire search
is lost even though it completed successfully.

So the first move when killing a run is always:

```bash
docker cp <sidecar-container>:/state/admin/session ./salvaged-session
```

That gives you `candidates/repository.git` — a real git repo, so
`git --git-dir=.../repository.git log --all` lists every candidate and
`git archive <sha>` extracts one — plus `database.json`, `budgets.json` and the
evaluation job records. With it, the submitted candidate can be re-scored on the
held-out set afterwards. Only once you have it should you stop the containers.

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

## Auditing a finished pass

A run that was degraded still emits a plausible in-range number, so a reward is
only a measurement once you have checked that it is one. Per finished cell,
confirm all of:

- `error_rate` is `0.0` in `jobs/*/task__*/verifier/finalization.json` — the
  held-out set scored completely — or account for what dropped;
- the verifier error block is empty (a non-empty one means scoring aborted and the
  number is an artefact, not a low score);
- `shipped: true`;
- no evaluations were terminated mid-search (this does not invalidate the score,
  but it means the search ran on less evidence — a caveat on the comparison);
- no two cells share a shipped content digest, which would mean they shipped the
  same tree, usually the unmodified baseline. That is a null result, not agreement.

```bash
cd runs && python3 audit_pass.py <benchmark> <suffix> <baseline_reward>
cd runs && python3 terminated_evals.py <benchmark> <suffix>
```

Both glob **relative to the working directory** and must be run from `runs/`. From
anywhere else they print an empty table with a header, which reads as "no results
yet" rather than "wrong directory".

Prefer `session/database.json` inside the session archive over convenience
sources: the agent's own log shows only the terminations it happened to print, and
W&B summary fields are last-value-wins, so a count there proves only "at least
one".

## Per-benchmark quick reference

Values drift; treat this as an index and confirm against each `baseline/build.yaml`
before launching. Held-out baselines are the pinned `baseline_reward`.

| benchmark | task data | target model | held-out baseline | dev / val cases |
|---|---|---|---|---|
| `officeqa` | **local, gitignored** | `fireworks_ai/deepseek-v4-flash` | 0.3412 | 196 / 392 |
| `browsecomp-plus` | **local, gitignored** | `fireworks_ai/deepseek-v4-flash` | 0.4619 | 132 / 264 |
| `gaia` | registry digest | `gpt-5.4-mini` | 0.6205 | 132 / 264 |
| `tau3` | registry digest | `fireworks_ai/deepseek-v4-flash` | 0.7321 | 300 / 600 |
| `swe-atlas-qna` | registry digest | `fireworks_ai/gpt-oss-120b` | 0.0676 | 100 / 196 |
| `swe-bench-pro` | registry | `gpt-4o` | not pinned | 146 / 292 |

All share the same shape: `selection_partition: validation`, `reward_mode: submit`,
`baseline_floor: false`, global `n_attempts: 1` with the **test** target overridden
to `n_attempts: 3` / `aggregate_attempts: mean`, and `environment_name:
${inner_env:-modal}`.

**The two benchmarks marked "local" cannot run from a fresh checkout.** Their task
directories are gitignored, so they exist only where someone has already
materialised them. Fetch them before launching (see the README) and do not assume a
clean clone can run them. The failure is badly misreported: the loader leaves an
unresolvable `task_source` untouched and the validator then reads it as a registry
reference, so you get `registry task_source must include an explicit version`
rather than anything pointing at a missing directory.

## Per-benchmark gotchas

Don't hardcode assumptions — read `CONFIGURATION.md` and the benchmark's
`baseline/build.yaml` and `baseline/README.md`. Common differences: a benchmark
may pin a **different target model**; some run an **in-container judge or
user-simulator** on the real upstream (extra credentials + cost, and reduced
isolation); some pull **large prebuilt images or corpora** (long first build);
timeouts and partition sizes vary widely. When in doubt, dry-compile and read the
rendered `instruction.md` and `serve.json`.
