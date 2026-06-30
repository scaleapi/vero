# Harbor integration — architecture

The Harbor integration turns a **vero optimization run into a [Harbor](https://www.harborframework.com)
task**. The agent-under-test of that Harbor task is an *optimizer*: any Harbor agent
(Claude Code, an oracle script, …) edits a target repository and spends an evaluation
budget; the reward is the best candidate's score on a hidden test split.

This lets anyone optimize a coding agent with plain `harbor run`, and aims to make the
result leaderboard-gradeable. On a best-effort, OS/process-level basis the integration
withholds per-sample labels from the agent volume, meters every agent evaluation against
a budget, applies read-only paths, and gates final hidden-split scoring behind a
`root:600` token the de-privileged optimizer cannot read (a container escape is out of
scope). See [Leaderboard integrity](#leaderboard-integrity-the-trust-boundary) for the
exact mechanisms and their limits.

```
harbor run -p <task> -a <optimizer> -m <model> -e <provider>
        │
        ▼  one optimization trial (a Docker Compose environment):
  ┌────────────────────────┐        ┌────────────────────────────────────┐
  │ main (optimizer bench)  │  HTTP  │ eval-sidecar (the evaluation engine) │
  │  • target repo (rw*)    │ ─────► │  • dataset + scorer + baseline repo  │
  │  • `vero harbor` client │        │  • budget ledger + creds             │
  │  • runs the -a optimizer│        │  • `vero harbor serve` (FastAPI)     │
  └────────────────────────┘        └────────────────────────────────────┘
        │  (trial end, shared verifier)            ▲
        └── `vero harbor finalize` (admin token) ──┘ → /logs/verifier/reward.json
```

## The optimization loop

1. **`vero harbor build`** compiles a `build.yaml` into a Harbor task directory
   (`environment/` compose + Dockerfiles, `instruction.md`, `tests/test.sh`), baking
   the dataset, scorer, baseline repo, and a `ServeConfig`.
2. At trial start, **`main`** seeds the target repo onto a shared volume and applies
   write-access rules; the **eval-sidecar** starts `vero harbor serve` and writes a
   per-trial admin token.
3. The **optimizer** (the `-a` agent) edits the repo, commits, and calls
   `vero harbor eval --split <train|validation>` to measure a commit. The sidecar
   fetches that commit, evaluates it (metered against the budget), and returns an
   **aggregate** score (never per-sample labels).
4. At trial end, Harbor runs `tests/test.sh` in `main` (shared verifier mode). It
   reads the admin token and calls the sidecar's **`finalize`**: the sidecar selects
   the winning commit and scores it on the **hidden** test split, producing the reward.

## Two evaluation modes

The seam is a single injection point on the `Evaluator` (`eval_strategy`):

- **Mode A — vero scores** (`task_project`/`task` + dataset). vero runs the agent's
  inference and a vero scoring function against vero-side labels. Example agent:
  [`examples/gsm8k-agent`](../../examples/gsm8k-agent) (note: it ships the agent and
  vero task but not yet a `build.yaml`; see the Mode A `build.yaml` snippet in the
  [tutorial](./tutorial.md) for a runnable config).
- **Mode B — Harbor scores** (`HarborConfig`). Inference is delegated: for each
  candidate, `HarborRunner` runs a *nested* `harbor run` of the agent on a set of
  Harbor tasks (e.g. on Modal) and collates the verifier rewards. One Harbor task =
  one sample. Example: [`examples/gaia-optimization`](../../examples/gaia-optimization).

Both modes share the same topology, trust boundary, budget, and verifier — only the
"produce sample results" step differs.

## Leaderboard integrity (the trust boundary)

The optimizer is untrusted. Integrity rests on a few mechanisms, all best-effort at
the OS/process level (a container escape is out of scope):

- **3-tier split visibility** (`SplitAccessLevel`): `visible` (aggregate + per-sample
  results), `non_viewable` (aggregate score only, no labels), `no_access` (hidden;
  never evaluable by the agent, never written to its volume). **A split not listed in
  `splits:` currently defaults to viewable (fail-open).** List *every* split explicitly
  and give held-out splits `no_access`; do not rely on omission to hide a split. (Once
  the protocol fix lands this default becomes fail-closed: an unlisted split defaults to
  `no_access`.)
- **Write-routing by tier**: the sidecar writes only the agent-permitted projection of
  each result to the *agent-results* volume (read-only in `main`). Full results, the
  dataset, the ledger, and creds live on the *admin* volume, **never** mounted to `main`.
- **Token-gated finalize**: `finalize` (selection + hidden-split scoring) requires an
  admin token written `root:600` on a volume `main` mounts read-only. The optimizer
  runs as a de-privileged user and cannot read it, so it cannot trigger scoring or
  probe the test split; the verifier (root, shared mode) can.
- **Metered budget**: a durable `BudgetLedger` caps how much the agent can evaluate per
  split. Admin (verifier) evaluations bypass the meter.
- **Commit transfer**: the sidecar `git fetch`es the agent's commit from the mounted
  repo into its *own* repo with hooks disabled and `file://` (object copy, no
  alternates), so the evaluated tree is fully owned by the sidecar and tamper-evident.
- **Protected scorer / write-access** (mode-dependent): in **Mode B** the scorer,
  dataset, and creds are sidecar-only and never share a filesystem with the optimizer.
  In **Mode A** the scoring function is vero task code that lives inside the agent's
  *editable* `agent_repo` (e.g. `src/gsm8k_agent/vero_tasks`); it is protected only by
  `read_only_paths` in `build.yaml`, applied as unix perms in `main` before the
  optimizer runs (best-effort, not container isolation). Full Mode A scorer isolation
  requires baking the task project into the sidecar instead of the agent repo, which
  lands with the `serve.py` fix.

### Why a sidecar + shared verifier

The evaluation engine, dataset, and creds live in a separate container so the optimizer
never shares a filesystem or process space with them (in **Mode B** the scorer too; in
**Mode A** the scorer currently lives in the agent's editable repo and is guarded only by
`read_only_paths`, until the `serve.py` fix bakes a Mode A task project into the sidecar). We use Harbor's
**shared verifier** (the env, including the sidecar, stays up during `tests/test.sh`)
so the verifier can reach the live engine over HTTP and stay the single source of
truth — avoiding shipping the repo/dataset/ledger into a fresh verifier container. The
agent/admin split is enforced by the `root:600` token rather than separate services.

## Component map

```
vero/harbor/
├── build/            `vero harbor build`: BuildConfig → Harbor task dir
│   ├── config.py       BuildConfig (the build.yaml schema)
│   ├── compiler.py     renders the task dir; bakes dataset/scorer/repo/ServeConfig
│   └── templates/      compose, two Dockerfiles, instruction.md, test.sh, seed.sh, solve.sh
├── serve.py          `vero harbor serve`: assemble engine+sidecar+verifier from ServeConfig
├── app.py            FastAPI surface: /eval /submit /status (agent), /finalize (admin)
├── server.py         EvaluationSidecar: commit transfer + tier write-routing (transport-agnostic)
├── verifier.py       Verifier: commit selection (submit | auto_best) + hidden-split scoring
├── auth.py           per-trial admin token (generate / root:600 write / verify)
├── cli.py            `vero harbor` group: build | run | serve | eval | submit | status | finalize
├── config.py         HarborConfig (Mode B)
├── runner.py         HarborRunner (Mode-B EvalStrategy): nested `harbor run` → collate
├── dataset.py        Mode-B {split: [task_names]} partition → DatasetDict
└── protocol.py       aggregate-safe wire types + the redaction of an Experiment
                      (note: an unlisted split currently defaults to viewable / fail-open)

vero/evaluation/
├── engine.py         EvaluationEngine: budget metering + the single evaluate() entry point
├── evaluator.py      Evaluator: checkout + run; the eval_strategy seam (Mode A vs B)
└── strategy.py       EvalStrategy protocol
```

The compiler↔sidecar contract is `ServeConfig` (baked as `environment/sidecar/serve.json`);
the optimizer↔sidecar contract is the HTTP API in `app.py` (+ the `vero harbor` CLI clients).

## See also

- [Tutorial](./tutorial.md) — build and run an optimization task end to end.
- [`examples/gsm8k-agent`](../../examples/gsm8k-agent) (Mode A agent; no `build.yaml` yet, use the tutorial's Mode A snippet).
- [`examples/gaia-optimization`](../../examples/gaia-optimization) — Mode B (nested Harbor on Modal).
