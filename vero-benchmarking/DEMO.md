# Vero Demo: End-to-End on MATH

This walks you through running Vero on a single small dataset so you can watch the
optimizer improve an agent and inspect the recorded results. The demo uses a tiny
subsample of **MATH-Hard** (competition math word problems) via the `math-mini` task:
it targets the `generic-agent`, scores answers with an LLM math-judge, and its dataset
downloads from HuggingFace only (no Kaggle/Modal credentials needed).

Everything below is run from the `vero-benchmarking/` directory.

---

## What you're about to do

1. Install deps and set API credentials.
2. Build the `aflow_math_mini` dataset.
3. Sanity-check the setup (`vero check`).
4. (Optional) Smoke-test inference + scoring on a few samples (`vero evaluate`).
5. Run a real optimization loop — the actual "Vero working."
6. Inspect the recorded results and see exactly what Vero changed.

> **How it fits together:** `generic-agent` (in `../vero-agents/`) is the agent being
> optimized. The `math` task tells Vero which dataset, which agent, and how to score.
> Vero runs the agent on the data, scores each answer, then iteratively edits the
> agent's code (prompts, logic) to push the score up — recording each attempt.

> **A note on scale and cost.** This guide is tuned to be *cheap to run through* — a tiny
> dataset and a small evaluation budget, so a full run costs on the order of a few
> hundred LLM requests. That same frugality limits how much the optimizer can discover
> and prove out: fewer samples and fewer evaluation runs mean noisier scores and less
> room to try, break, and recover from ideas. Treat the small settings here as a
> *walkthrough*, not a representative optimization run. For real work, scale the dataset
> and budgets up to suit the task (see the full `math` task and the knobs in step 5).

---

## 1. Prerequisites

```bash
cd vero-benchmarking

# Install the benchmarking harness (editable install of scale-vero from ../vero)
uv sync --all-extras
```

### Sync the agent project

`generic-agent` is a **separate uv project** with its own virtualenv, and it declares
`scale-vero` in its `dev` dependency-group. You must sync it once so its `.venv`
contains `vero` — otherwise task discovery fails with `No module named 'vero'`:

```bash
(cd ../vero-agents/agents/generic-agent && uv sync)
```

> Don't try to point at `../vero-agents` itself — its top-level `pyproject.toml` is
> only a ruff config (no `[project]`), so uv would create an empty venv there.

We'll refer to the agent by **absolute path** throughout (a relative path breaks
inside Vero's discovery subprocess, which runs from a different working directory):

```bash
GENERIC_AGENT="$(cd ../vero-agents/agents/generic-agent && pwd)"
```

### Credentials

Both agent inference and the LLM judge call models through a LiteLLM proxy (or any
OpenAI-compatible endpoint). Export these in your shell (or put them in a `.env` and
`source` it):

```bash
export LITELLM_BASE_URL="https://your-litellm-proxy/v1"
export LITELLM_API_KEY="sk-..."
# OPENAI_API_KEY is used as a fallback if LITELLM_API_KEY is unset.
```

The VeroAgent optimizer (step 5) also builds a `WebSearch` tool at startup, which
requires a Serper key to be **present** (it isn't validated at construction). The math
task never actually searches, so a dummy value is enough to start the run; use a real
key only if you want web search to function:

```bash
export SERPER_API_KEY="unused-for-math-demo"
```

> These are the two vars declared as `required_env_vars` by the math task. They must be
> set **before `vero check`**, not just before inference: the agent's task module builds
> an OpenAI client at import time, and discovery imports it — so an unset key makes even
> `check` fail. W&B is **not** required for this demo (the low-level runner leaves it
> opt-in).

---

## 2. Build the dataset

Build the cheap `math-mini` variant — a tiny deterministic subsample of MATH-Hard
(train=16, validation=8, test=30):

```bash
uv run python -m vero_benchmarking.datasets --dataset-name aflow_math_mini
```

The rest of this guide uses `--task math-mini` / `--dataset ./datasets/aflow_math_mini`.
To run the full paper-scale task instead, build the full dataset and use `--task math`
/ `--dataset ./datasets/aflow_math` everywhere below:

```bash
uv run python -m vero_benchmarking.datasets --dataset-name aflow_math   # full (test=1055)
```

Both download from HuggingFace, apply the AFLOW curated split mapping, and save a
`DatasetDict` (train/validation/test) under `datasets/`. Other benchmark datasets
(`aflow_gsm8k`, `aflow_drop`, `aflow_humaneval`, …) build the same way; see the task
table in `README.md`.

---

## 3. Sanity-check the setup

```bash
uv run vero check \
  --project-path "$GENERIC_AGENT" \
  --task math \
  --dataset ./datasets/aflow_math_mini
```

> `--task math` here is the agent's *task name* (both the `math` and `math-mini`
> benchmark entries reuse the same `math` task in `generic-agent`); the `mini` vs full
> choice is made by which `--dataset` you point at. In the optimization step (5) the
> `--task` value is the *benchmark registry key*, so there you use `math-mini`.

Expect all `[OK]` lines: uv project found, git repo, task discovery (`math: OK`),
env vars set, and the dataset splits with their sizes.

---

## 4. (Optional) Smoke test — a few samples, no optimization

Fast and cheap. Confirms the inference → scoring contract works before you spend money
on a full loop. `--isolate` is important here: `generic-agent` lives inside the `vero`
monorepo, so this copies it into a fresh throwaway git repo to evaluate cleanly.

```bash
uv run vero evaluate \
  --project-path "$GENERIC_AGENT" \
  --task math \
  --dataset ./datasets/aflow_math_mini \
  --split test \
  --num-samples 5 \
  --isolate \
  --task-params '{"model": "openai/gpt-4.1-mini-2025-04-14"}'
```

`--task-params.model` is the model the *agent under test* runs on (the example above is
`generic-agent`'s default); swap it for any model your proxy exposes. You'll see
per-sample scores (0.0–1.0) and judge feedback (`Expected: … Extracted Prediction: …
Is Equivalent: …`). Confirm the **average is below 1.0** — that's the headroom the
optimizer will work with. (If it's already 1.0, the optimizer will make no changes;
raise `--num-samples`, or the base model is too strong for this slice.)

---

## 5. Run the optimization loop

This is the real demo: Vero evaluates the baseline, then edits `generic-agent`'s code
over several rounds to improve its MATH score. Each run creates an isolated git
worktree of the agent, so your working tree is never touched.

### What Vero may and may not edit

The optimizer's writable surface is deliberately narrow, enforced by the `.veroaccess`
file in the agent project (rules work like `.gitignore` — last match wins). The two
files worth optimizing are:

- **`src/generic_agent/prompts.py`** — the per-task prompt templates
  (`get_math_prompt`, etc.). Highest-leverage knob.
- **`src/generic_agent/agent.py`** — the agent scaffold: tools, model settings, the
  `run_agent` workflow.

Everything under **`src/generic_agent/vero_tasks/`** is **read-only** by design — it
holds the task *definitions and grader* (`math.py` = the inference entry + LLM judge,
`utils.py` = judge templates). Letting the optimizer edit these would let it game its
own score, so writes there are rejected. 

### Run it

```bash
uv run python -m vero_benchmarking.runner vero \
  --task math-mini \
  --model anthropic/claude-sonnet-4-5-20250929 \
  --max-turns 60 \
  --evaluation-concurrency 2 \
  --evaluation-timeout 1800
```

> **Pick a capable optimizer model.** `--model` is the *optimizer* (the agent editing
> the code), not the agent under test. This is a hard reasoning-about-code task, so it
> needs a strong model — the value above is the runner's default; any frontier-class
> model your proxy exposes works. A weak model (e.g. an older/small model) tends to
> misdiagnose the code and fixate on the read-only task files, burning the run with no
> accepted change. Use whichever strong model your endpoint serves.

In our runs on `math-mini`, the optimizer typically edits the MATH prompt in
`prompts.py` — tightening the "final answer must be a single `\boxed{…}` line" contract
— and lifts the score by a few points over the baseline. Exactly what it changes (and
whether it improves at all) varies run to run; optimization is stochastic.


### Note on the demo run

The `math-mini` task sets small budgets so a full run stays around a few hundred LLM
requests: `train_budget=8` (evaluation runs the optimizer gets on train), `batch_size=16`
(samples per run), `validation_budget=2`, and a 30-sample test split.

These are a balance, and the balance matters:

- **Smaller = cheaper, but weaker signal.** Fewer samples per evaluation make scores
  noisier (a one-answer swing on 16 samples is ~6 points), and a tiny `train_budget`
  gives the optimizer little room to try an idea, see it fail, and recover — a single
  broken edit can consume the remaining budget before it fixes itself, so the run ends
  with nothing worth showing. `train_budget=8` is a deliberate floor for that reason; at
  `4` the demo often produces no improvement at all.
- **`validation_budget=2`** gives the optimizer the *option* to validate its final
  version on a **held-out** split. When it uses that budget, `vero session inspect`
  shows a `validation` row you can trust over a (possibly overfit) `train` score — but
  the optimizer won't always spend it, so you may see only `train` rows.
- **For real optimization, scale up.** The full `math` task uses `train_budget=8`,
  `batch_size=512`, and the 1055-sample test split (several thousand requests). Pick
  dataset size and budgets to fit the task and your cost tolerance — the mini settings
  are for learning the workflow, not for a representative result.

**Cost / time knobs** (all runtime flags, independent of the task's budgets):

- `--evaluation-concurrency 2` — cap simultaneous model calls. Keep this **low** if your
  provider rate-limits you. Symptom of it being too high: repeated `Retrying request to
  /responses in 60 seconds` (the judge) or `/chat/completions` (inference), sometimes
  ending in a `SubprocessTimeoutError`. Trickling 1–2 at a time usually avoids throttling;
  the default of 100 will not survive a tight rate limit. (You may not hit any limit —
  it depends on your provider/key.)
- `--evaluation-timeout 1800` — overall per-eval-subprocess timeout (seconds). The
  default of 180 is too short if any rate-limit backoffs occur; raise it so a slow eval
  isn't killed mid-run.
- `--max-turns 60` — total turns the optimizer agent may take over the whole run (one
  turn = one model call + its tool calls); the SDK stops at this ceiling. Default 200.
- `--skip-initial-eval` — skip the baseline pass if you just want to see it iterate.
- `--env-file .env` / `--subprocess-env-file .env` — pass credentials via file instead
  of exporting.

To run through the higher-level batch harness instead (adds config presets but
**force-enables W&B**, so it needs `WANDB_API_KEY`):

```bash
uv run python scripts/run_benchmark.py --scaffold vero-default --model haiku --task math
```

When the run finishes it prints the **Session ID**; use `vero session inspect` (step 6)
to find the best commit and see what changed.

---

## 6. Inspect the recorded results

```bash
# List all sessions (most recent last)
uv run vero session list

# Full detail for one session: config, every experiment attempt, and scores
uv run vero session inspect <session_id>
```

Results are recorded in several places:

| Location | What's there |
| -------- | ------------ |
| `~/.vero/sessions/<session_id>/` | Session DB (every experiment + score), config, run logs |
| `logs/session_manifest.jsonl` | One line per run: session id, task, agent type, best commit. Note: `best_commit` is only populated when a commit is validated on the run's held-out `eval_split`; for these benchmark tasks (the agent optimizes on train/validation, never the hidden `test` split) it's typically `null`. Use `vero session inspect` to find the best commit — see below. |
| `~/.vero/sessions/<session_id>/generic-agent-<task>-<slug>/` | The isolated git worktree holding the optimized agent code (branch `generic-agent-<task>-<slug>`) — `git log`/`git diff` it to see exactly what Vero changed (see below) |
| `results/` | Parquet output (baseline-evaluation runs) |
| W&B | Metrics per run — only if you opted in (`--enable-wandb` / `run_benchmark.py`) |

### See exactly what Vero changed

Each run leaves its optimized agent in an **isolated git worktree** under the session
directory. To see what the optimizer did, use `vero session inspect` to list every
evaluated commit with its score, pick the best one, and `git diff` it against the
original code in that worktree.

**Step 1 — find the best commit.** Inspect the session and read the `Experiments`
section:

```bash
SESSION=<session_id>          # printed at the end of the run (or: uv run vero session list)
uv run vero session inspect "$SESSION"
```

```text
Experiments (4):
  c5dfba3f-f7f  commit=71726129  split=test   status=success  samples=30  score=0.833
  8c28d21e-...  commit=8c28d21e  split=train  status=success  samples=16  score=0.812
  85cb85fa-...  commit=fe61b21c  split=train  status=failed   samples=16  score=0.000
  abd0d6fb-...  commit=9c0c695f  split=train  status=success  samples=16  score=0.875
```

Pick the **highest-scoring `train`/`validation` commit** — that's the optimizer's best
version (here `9c0c695f @ 0.875`). The `test` row is the *baseline* (the original code,
evaluated on the held-out split); don't pick it. Skip `status=failed` rows.

**Step 2 — diff it against the original.** The base version is in the session config:

```bash
SDIR="$HOME/.vero/sessions/$SESSION"
BASE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['base_version'])" "$SDIR/config.json")
WORKTREE=$(echo "$SDIR"/generic-agent-*)   # the isolated worktree for this session
BEST=<paste the best commit hash from step 1>

# Full diff, excluding generated __pycache__ noise:
git -C "$WORKTREE" diff --stat "$BASE" "$BEST" -- . ':(exclude)**/__pycache__/**'
git -C "$WORKTREE" diff       "$BASE" "$BEST" -- . ':(exclude)**/__pycache__/**'
```

> **Pick from the score list, not the branch tip.** The worktree branch's `HEAD` is
> often a *later, unevaluated* commit the optimizer made just before running out of
> budget — diffing against it can show a change whose score was never measured, or one
> that scored *worse*. Always pick the highest-scoring commit from the `Experiments`
> list.
>
> **No commit beats the baseline? That's a real outcome, not an error.** Optimization is
> stochastic, and small budgets make it more likely: the optimizer sometimes makes a
> change that *lowers* the score (e.g. a prompt edit that breaks answer formatting) and
> doesn't recover within budget. If the best `train`/`validation` score doesn't exceed
> the baseline, there's nothing worth diffing — re-run, or raise the task's budget /
> `--max-turns`.

---

## Cleanup

```bash
# Remove demo sessions
uv run vero session clear --all      # or: clear <session_id>

# Remove the worktree branches Vero created (from the generic-agent repo)
git -C ../vero-agents/agents/generic-agent worktree list
```

---

## Troubleshooting

- **`Path to vero-agents does not exist`** — run from `vero-benchmarking/`, or set
  `VERO_AGENTS_PATH` to your `vero-agents` checkout.
- **`No module named 'vero'` during task discovery** — the `generic-agent` project
  wasn't synced. Run `(cd ../vero-agents/agents/generic-agent && uv sync)` (step 1).
  If uv created a `.venv` under `vero-agents/` (the parent), delete it — that dir has
  no `[project]` and shouldn't have a venv.
- **`Project directory ... does not exist` / uv picks the wrong project** — you passed
  a **relative** `--project-path`. Use an absolute path (`$GENERIC_AGENT` above);
  Vero's discovery subprocess runs from a different cwd, so relative paths break.
- **`OpenAIError: Missing credentials` during `vero check`** — credentials aren't set.
  The agent builds its LLM client at import time, so discovery needs `LITELLM_API_KEY`
  (or `OPENAI_API_KEY`) present. Set them (step 1) before running `check`.
- **Missing env var errors** — `LITELLM_BASE_URL` and `LITELLM_API_KEY` must be set
  (see step 1); `vero check` reports exactly which are missing.
- **Optimizer crashes on a `429 … token` rate-limit error** — this is a rate limit on
  the *optimizer model's* provider route (not the eval subprocess, so
  `--evaluation-concurrency` won't help). A single VeroAgent turn sends a large context,
  which can blow past a low per-minute token cap; the model retry backoff can't wait out
  a reset window, so the run dies. Such caps are often **route-** or **key-specific**
  (one provider route throttled while another isn't). If you hit this, switch `--model`
  to a strong model on a route without that cap, or get the throttled key's limit
  raised. Many proxies won't rate-limit you at all — this only applies if yours does.
