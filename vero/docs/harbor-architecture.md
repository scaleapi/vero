# Harbor benchmark architecture

**Audience:** an engineer picking this up cold, or reviewing the PRs.
**Scope:** the *harbor* deployment path — how VeRO turns an untrusted optimizer
agent into a trustworthy benchmark number. (The in-process
`OptimizationSession`/`optimizer.py` path is a separate, simpler runner and is
not used by the harbor deployment.)

## The core idea

VeRO benchmarks an **optimizer** (an LLM coding agent such as codex/claude) at
*improving a candidate program*. The optimizer edits code and asks a **trusted
evaluator** to score it; the evaluator holds the held-out answers and never lets
the optimizer see them. The whole system exists to get a *trustworthy* number
out of an *untrusted* agent editing code against data it must not peek at.

Everything below hangs off one organizing idea: the **trust boundary** between
the optimizer and the evaluator.

## Topology: three containers

A run is a Docker Compose stack of three services (on Modal, or local docker):

```
        UNTRUSTED  │  TRUSTED
                   │
  ┌─────────────┐  │  ┌────────────────────┐     ┌──────────────────┐
  │    main     │──┼─▶│    eval-sidecar    │     │ inference-gateway│
  │ (optimizer) │  │  │  (FastAPI, root)   │     │  (metered LLM    │
  │  edits      │  │  │  held-out data +   │     │   proxy)         │
  │ /work/agent │  │  │  scores + budgets  │     │                  │
  └─────┬───────┘  │  └─────────┬──────────┘     └────────┬─────────┘
        │          │            │ runs inner `harbor run` │
        │ LLM      │            │ as unprivileged `harness`│
        └──────────┼────────────┴──── scoped token ───────┘
                   │
```

- **`main`** — runs the optimizer agent. *Untrusted*: it may write anything into
  `/work/agent` (the candidate repo). It reaches the sidecar only over HTTP and
  the LLM only through the gateway. Real secrets are scrubbed to `""` here.
- **`eval-sidecar`** — the *trusted control plane*. Owns the held-out task
  lists, the scores database, the budget ledger, and the admin token. Serves
  agent-facing endpoints (budgeted, disclosure-gated) and token-gated admin
  endpoints.
- **`inference-gateway`** — a credential-isolating, budgeted proxy in front of
  the real LLM upstream. Issues *scoped tokens* (producer / evaluation /
  finalization), each with its own model allow-list and budget, so the optimizer
  never sees the raw key and its LLM usage is metered.

Named volumes carry state between services: `agent_repo`, `agent_context`,
`admin_state`, `token_state`, `inference_state`.

## A run, end to end

1. **`vero harbor run --config gaia/build.yaml --agent codex --model gpt-5.5`.**
2. **Compile** (`harbor/build/compiler.py`): the `build.yaml` + the target repo +
   partitions → a self-contained task dir — three Dockerfiles,
   `docker-compose.yaml`, `serve.json` (the sidecar's deployment config), the
   case lists, `instruction.md` (what the agent reads), and `test.sh` (the
   verifier phase).
3. It shells out to **inner `harbor run -e modal`**, which builds the images,
   brings up the stack, and injects the optimizer into `main`.
4. **Optimizer loop**: the agent reads its `.vero` context, edits the candidate
   in `/work/agent`, and calls the sidecar `/eval` to score candidates — on
   **development** (full per-case feedback) and **validation** (aggregate score
   only). It iterates, and may `/submit` a final pick.
5. **Finalization** (`test.sh`, admin-token'd): `verifier.finalize()` selects a
   candidate, admin-re-scores it on the held-out **test** target, writes
   `reward.json` + `finalization.json`; `export-session` archives the whole
   session.
6. Harbor collects `/logs` back to **`jobs/<timestamp>/<task>/`** on disk and
   reports the reward.

## The evaluation core

- **`EvaluationEngine`** (`evaluation/engine.py`) runs every evaluation through a
  **backend** and records an **`EvaluationRecord`** (`evaluation/models.py`): the
  request (candidate, evaluation_set, limits) + a report (per-case results,
  metrics, diagnostics, artifacts) + an objective value + a **principal**. It
  enforces the **`BudgetLedger`** (`evaluation/budget.py`) and fires **listeners**
  (where the W&B sink hooks in).
- **`HarborBackend`** (`harbor/backend.py`) is how a candidate is actually
  scored: it runs a *nested* `harbor run` over the tasks (`uv run` the
  candidate's package, execute the agent per task, collect trial records),
  aggregating attempts (`mean` by default, zero-filling dead attempts and
  recording `n_dead_infra`/`n_clean` so an outage-diluted mean is distinguishable
  from a clean low one).
- **Principals** are the key access concept:
  - **`agent`** — the optimizer's search evals: metered, disclosure-limited.
  - **`admin`** — trusted finalization / re-scores: unmetered, full disclosure.
    This is the authoritative number.

## Disclosure & the anti-cheating contract

Each partition has a **disclosure level** (`harbor/sidecar.py`):
- **development → FULL**: per-case results shown (the disclosed set to iterate
  on).
- **validation → AGGREGATE**: only an *average over ≥ k cases* is returned (the
  k-anonymity floor, default 5), so individual held-out labels can't be read off
  one case at a time.
- **test → held-out**: not agent-evaluable; only admin scores it, at
  finalization.

## Selection / finalization (what ships)

`CanonicalVerifier.finalize()` (`harbor/verifier.py`) decides what to ship and
computes the reward:
- **Chain**: the agent's explicit `submission` → else `auto_best` (rank
  candidates by *coverage-qualified* validation evals, admin-re-score the top-K
  on the full set) → else **pick-last** (the current candidate). It ships nothing
  only if there is no candidate at all.
- **`baseline_floor`** (opt-in, off by default): ship only if the candidate
  beats the seed. Because the seed is fixed, its score can be **pinned**
  (`baseline_reward` / `baseline_selection_score`) — or produced reproducibly via
  `vero harbor score-baseline --replicates N`.
- Result → `VerificationResult` (shipped, rewards, baseline_rewards) →
  `reward.json` (what harbor consumes) + the durable session archive.

## The security model (five mechanisms)

1. **Credential isolation** — the gateway's scoped tokens; the raw upstream key
   lives only in the sidecar. (`harbor/inference.py`, compose template.)
2. **Harness isolation** — the inner `harbor run` (arbitrary candidate code) runs
   as unprivileged `harness`; the session dir is `0700 root`, case lists `700`,
   `serve.json` `600` — so the candidate cannot read the answer key it is graded
   on. (`harbor/backend.py` `run_as`, `Dockerfile.sidecar.j2`,
   `harbor/deployment.py`.)
3. **k-anonymity floor** — aggregate subset evals must cover ≥ 5 cases.
   (`harbor/sidecar.py` `_enforce_aggregate_floor`.)
4. **Budget metering** — per-scope gateway budgets + a case/run ledger, with a
   *reserved finalization scope* so the mandatory admin re-score can't be starved
   by the agent's search. (`evaluation/budget.py`, gateway config.)
5. **Fail-safe finalization** — never ship an unverified candidate on an infra
   blip; a session export never fails wholesale. (`harbor/verifier.py`,
   `harbor/session.py`.)

## The build pipeline

`build.yaml` (`HarborBuildConfig`, `harbor/build/config.py`) →
`compile_harbor_task` (`harbor/build/compiler.py`) renders the whole deployable
task (Dockerfiles + compose + `serve.json` + cases + instructions). `serve.json`
deserializes into `HarborDeploymentConfig` (`harbor/deployment.py`), which
`build_harbor_components` turns into the live sidecar + verifier. So a benchmark
is *fully specified by one YAML* (see `harness-engineering-bench/gaia/baseline/`).

## Observability

`SidecarWandbSink` (`runtime/wandb.py`) subscribes to the engine's listeners and
streams every evaluation to Weights & Biases from the trusted side: scoped
metrics per `partition/principal`, `num_cases`, remaining budget, optional full
trace artifacts, and a run summary at finalize. Each `vero harbor run` gets its
own W&B run.

---

## Suggested review order

Read outside-in — from *what a benchmark declares* down to *how it runs and stays
honest*. Paths are under `vero/src/vero/` unless noted.

1. **What a benchmark is** — `harness-engineering-bench/gaia/baseline/build.yaml`
   and `harbor/build/config.py` (`HarborBuildConfig`, `AgentAccessSpec`,
   `VerificationTargetSpec`). This is the declarative surface; everything else
   serves it.
2. **Compile → deployable task** — `harbor/build/compiler.py`
   (`compile_harbor_task`) and the templates in `harbor/build/templates/`
   (`docker-compose.yaml.j2`, `Dockerfile.sidecar.j2`, `instruction.md.j2`,
   `test.sh.j2`). See how one YAML becomes the three-container stack + `serve.json`.
3. **The topology at runtime** — `harbor/deployment.py`
   (`HarborDeploymentConfig`, `build_harbor_components`): how `serve.json` becomes
   a live engine + sidecar + verifier.
4. **The evaluation core** — `evaluation/models.py` (`EvaluationRecord`,
   `EvaluationPrincipal`, `DisclosureLevel`, `EvaluationSet`), then
   `evaluation/engine.py` and `evaluation/budget.py`. This is the vocabulary the
   rest of the system speaks.
5. **How a candidate is scored** — `harbor/backend.py` (`HarborBackend`): the
   nested `harbor run`, attempt aggregation, and the infra-vs-candidate taxonomy.
   The most intricate file; skim `_case_result`, `_command`, `_environment`.
6. **The sidecar** — `harbor/sidecar.py` (access policies, disclosure floor,
   tracked eval jobs, submission) and `harbor/app.py` (the HTTP surface: agent
   `/eval` vs admin `/finalize` `/score/baseline` `/session/export`).
7. **Selection & finalization** — `harbor/verifier.py` (`CanonicalVerifier`): the
   submit → auto_best → pick-last chain, coverage-qualified selection, the
   baseline floor + pinning, `measure_baseline`. This decides the shipped number.
8. **Security specifics** — the five mechanisms: `run_as`/`harness_user` in
   `harbor/backend.py` + `Dockerfile.sidecar.j2`; the gateway in
   `harbor/inference.py`; the session archive in `harbor/session.py`.
9. **Observability** — `runtime/wandb.py` (`SidecarWandbSink`).
10. **The CLI glue** — `harbor/cli.py` (`vero harbor run`, `finalize`,
    `export-session`, `score-baseline`).

Tests mirror this order (`tests/test_v05_harbor_*.py`) and are a good
executable spec for each layer.
