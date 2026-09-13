---
name: run-benchmark
description: >-
  Run one HarnessOpt-Bench optimization end-to-end (compile, inference gateway,
  optimizer agent, sandboxed evaluations, finalization on the held-out test
  split), including the preflight to confirm before launching and the health
  checks to verify after. Use when launching, reproducing, or debugging a run.
---

# Running a HarnessOpt-Bench optimization

A runbook for launching one benchmark's optimization run and confirming it is
healthy. Fill in your own inference endpoint, Modal account and W&B account;
treat every `<placeholder>` as something you supply. Never commit real keys,
tokens or endpoints. They belong in a local, git-ignored env file.

## What a run is

Each benchmark compiles from `harness-opt-bench/<benchmark>/baseline/build.yaml`,
the single source of truth. `vero harbor run` compiles it into a Harbor task and
stands up three services: an **inference gateway** that holds the real upstream
key and enforces per-scope model allow-lists and token budgets, an **evaluation
sidecar** that owns the cases, the scoring and the final candidate selection,
and the **optimizer agent**, a coding agent that edits only `target/`, commits
candidates, and scores them through the `evals` CLI. When the optimizer finishes
or its clock runs out, the trusted verifier scores the selected candidate on the
held-out `test` partition and writes the final reward.

The optimizer holds a scoped producer token for the gateway, never the upstream
key. The per-scope allow-list confines the target model in the normal case but
is not a hard guarantee against an adversarial optimizer, which authors the
candidate and holds the producer token. Benchmarks whose tasks run an
in-container judge set `task_services_use_upstream`, which puts the raw upstream
credential into the evaluation sub-run. Both are recorded in the affected
`build.yaml` files and in `vero/src/vero/gateway/inference.py`.

## Prerequisites

- A checkout containing both `vero/` and `harness-opt-bench/`. Run commands from
  `vero/`.
- `uv`. The CLI is invoked as `uv run vero ...`.
- A **Modal** account and tokens. Inner evaluation sandboxes run there
  (`environment_name: ${inner_env:-modal}`), and the outer optimizer trial can too.
- **Docker**, if you run the outer trial locally with `--environment docker`.
- A **Weights & Biases** account and API key, cloud or self-hosted.
- An **OpenAI-compatible inference endpoint** and key that serves both your
  optimizer model and the benchmark's target model. The gateway proxies to it.
- A local env file (start from `secrets.env.example`) with `OPENAI_API_KEY`,
  `OPENAI_BASE_URL`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `WANDB_API_KEY`,
  `WANDB_BASE_URL`, and optionally `MODAL_ENVIRONMENT` to pick a Modal
  environment other than your workspace default. Every build declares these under
  `secrets`; the compiler refuses to compile if one is missing.
- The task data for officeqa and browsecomp-plus, which is not in git. Run
  `python3 scripts/task_data.py` and follow what it prints.

## Launch command

```bash
cd <repo-root>/vero
uv run vero harbor run \
  --config ../harness-opt-bench/<benchmark>/baseline/build.yaml \
  --env-file <your>.env \
  --environment modal \
  --agent <harness> \
  --model <launch-model> \
  --param optimizer_model=<wire-model> \
  --param wandb_run=<benchmark>__<label> \
  --yes \
  -o <somewhere>/<benchmark>/<label>/jobs
```

Things that bite:

- **`--agent` is the exact Harbor registry name**: `claude-code`, `codex`,
  `opencode`, `kimi-cli`, `goose`, `mini-swe-agent`. A wrong name fails at init
  before any container starts, and the error lists the valid names.
- **The model has two spellings.** `--model` is what the harness is launched
  with; `--param optimizer_model=` is what the gateway allow-list must contain,
  and it must equal the string the harness puts on the wire. claude-code sends
  the model bare, opencode requires `provider/model` but sends the bare id,
  codex sends the last segment, kimi-cli needs an `openai/` prefix. A mismatch is
  a 403 on the optimizer's first request. `CONFIGURATION.md` has the table.
- **opencode also calls a small auxiliary model** of the same provider family.
  Builds carry a third allow-list slot for it (`optimizer_aux_model`); for an
  Anthropic optimizer pass the dated Haiku id.
- **Inner evaluations must run on Modal.** `inner_env=docker` fails every case,
  because the sidecar container has no Docker socket.
- **`--environment`** picks where the optimizer itself runs. `modal` survives
  your machine sleeping; `docker` is local and fully observable.

## Preflight: confirm before launching

Compile without launching and read the artefacts. This catches almost every
misconfiguration for free:

```bash
VERO_SKIP_SECRET_CHECK=1 uv run vero harbor build \
  --config ../harness-opt-bench/<benchmark>/baseline/build.yaml \
  --output /tmp/precompile
```

Then check, in `/tmp/precompile`:

1. `environment/sidecar/serve.json`: each partition backend has the
   `n_attempts` / `aggregate_attempts` you intend. Only the **test** backend should
   carry `n_attempts: 3` / `aggregate_attempts: mean`.
2. `serve.json` and `environment/gateway/config.json`: the evaluation allow-list
   is the benchmark's target model, the producer allow-list is your optimizer
   model. At launch, `--param optimizer_model=` replaces the compiled producer list.
3. `task.toml`: the `[agent] timeout_sec` (the optimizer's clock) and the verifier
   timeout are what the build declares.
4. `instruction.md`, the optimizer's task prompt, end to end.

Also confirm out of band that the optimizer model answers one tiny request on
your endpoint under exactly the wire spelling, that W&B authenticates, that Modal
tokens are valid, and that `baseline_reward` in the build is the floor you mean to
compare against. `vero harbor run` refuses to launch when a configured model is
not deployed upstream.

A compiled task is deterministic. Each benchmark commits
`baseline/compiled.manifest.json`; `vero harbor build --check <manifest>`
confirms your checkout reproduces it.

## Launching

Runs take hours and must outlive the shell that started them. Start the process
in its own session (a double fork or `setsid`), redirect its output to a log,
write its pid to a file, and **verify a live pid exists** before treating the
launch as successful. Never wrap launches in `|| true` inside a loop: it turns
every failure into silence.

Launch one cell first and define the gate in advance, for example one evaluation
completed with zero terminations. Ramp only after the gate is observed, and
replace cells as they finish rather than adding in bursts.

## While a run is in flight

`agent/` and `verifier/` artefacts are written only when the trial ends, and the
outer log holds a progress spinner. W&B is the mid-run view. The sidecar logs
each scope's requests and upstream errors under `inference/<scope>/...` and each
partition's completed and terminated evaluations under
`<partition>/agent/evaluations/...`. **Terminated evaluations should be zero.**
Run names carry a random suffix, so query by prefix rather than exact name.

Two traps when checking liveness:

- `ps aux | grep` truncates the command column to the terminal width and reports
  zero processes for healthy runs. Use `pgrep -f`.
- Phase text is not liveness. Compare the log's size across an interval.

## Concurrency

Runs share one upstream account, so they contend. Steer on measurement: the
marginal error rate between two consecutive checks, Δerrors ÷ Δrequests, and in
particular the rate imposed on the runs that were already going when load
changed. Do not derive a ceiling from tokens per minute against a published
quota; that mispredicts. New cells spike as their opening evaluations land
together, so wait a full interval after any load change before concluding
anything. Replace on completion rather than adding, and never kill a healthy run
to reduce load.

## Stopping a run is two operations

Killing the local process does **not** stop the work. The compose topology is
daemon-owned and Modal sandboxes are server-side, so a killed run keeps
executing and keeps billing with nobody collecting the result. An intentional
stop is the local process **and** its sandboxes (`modal app list`, then stop the
run's own app). The outer trial's app is named after the build
(`vero-optimize-<benchmark>`); inner evaluation sandboxes share one app per
suite, so never sweep that one while any other run is live.

### Copy the session out before you kill anything

Every candidate the optimizer committed lives only inside the sidecar until the
verifier exports `session.tar.gz`. Kill a run before that and the search is
lost. First:

```bash
docker cp <sidecar-container>:/state/admin/session ./salvaged-session
```

That gives you `candidates/repository.git`, a real git repository, plus
`database.json`, `budgets.json` and the evaluation records. The submitted
candidate can then be re-scored on the held-out set with
`scripts/rescore_candidate.py`.

## Post-launch health check

Watch for a fast crash first: invalid agent name, missing secret, model 403,
Docker down. Then, after the images build, confirm the good path:

- the outer process is alive and the log has no traceback;
- the three containers are up;
- the optimizer's requests reach the gateway without 403s;
- inner evaluation sandboxes are dispatching and a W&B run exists;
- sustained 429s from the provider mean you are sharing a quota; stop adding.

On a clear failure signal, stop the run and diagnose from disk. A theory is not a
signal; ambiguity resolves cheaply in one more interval, killing does not.

## What done looks like

`jobs/<timestamp>/task__*/verifier/finalization.json` with `shipped: true`, a
real `candidate` (non-null `parent_id`; a null one means the seed was shipped),
`rewards` keyed by the target's `reward_key`, `baseline_rewards`, and an **empty
`errors` block**. Alongside it: `session.tar.gz`, `experiment.html`, and in
`artifacts/` the `session-rescue.tar.gz` collected even if the verifier failed.

A run whose optimizer hit its clock is still scored: the trial records
`AgentTimeoutError` and the verifier runs on whatever was submitted.

## Auditing a finished run

A degraded run still emits a plausible number, so a reward is a measurement only
after these checks, all read from `finalization.json`:

- `reward_metrics.<key>.error_rate` is `0.0`, or you can account for what dropped;
- `errors` is empty. A non-empty block means scoring aborted and the number is an
  artefact, not a low score;
- `shipped: true` and `candidate.parent_id` is set;
- the W&B `terminated_total` counters are zero. Terminations do not invalidate the
  score but mean the search ran on less evidence;
- across a grid, no two cells share `candidate.metadata.content_digest`. Two cells
  that shipped the same tree usually shipped the unmodified seed.

For per-trial token and latency accounting from the gateway's request log, run
`scripts/per_trial_tokens.py <session-dir> --json` on the extracted session and
check its `coverage_pct` is near 100.

## Per-benchmark quick reference

Values drift; confirm against each `baseline/build.yaml` before launching.

| benchmark | task data | target model | pinned baseline | dev / val cases |
|---|---|---|---|---|
| `officeqa` | local, fetched | see build.yaml | 0.3412 | 196 / 392 |
| `browsecomp-plus` | local, fetched | see build.yaml | 0.4619 | 132 / 264 |
| `terminal-bench` | registry digest | `xai/grok-build-0.1` | 0.2407 | 68 / 144 |
| `gaia` (shell seed, `build.shell.yaml`) | registry digest | `gpt-5.4-mini` | 0.0 by construction | 132 / 264 |

All share: `selection_partition: validation`, `reward_mode: submit`,
`baseline_floor: false`, global `n_attempts: 1` with the test target at
`n_attempts: 3` / `aggregate_attempts: mean`, and `environment_name:
${inner_env:-modal}`. `gaia/baseline/build.shell.e2e.yaml` is an eight-case
variant with a 25-minute optimizer clock for testing the machinery end to end;
its score means nothing.

## Per-benchmark gotchas

Read `CONFIGURATION.md` and the benchmark's `baseline/build.yaml` and
`baseline/README.md` rather than assuming. A benchmark may pin a different target
model; some run an in-container judge on the real upstream; some pull large
prebuilt images or corpora, so the first build is slow; timeouts and partition
sizes vary widely. When in doubt, compile and read the rendered `instruction.md`
and `serve.json`.
