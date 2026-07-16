# Vero Demo: End-to-End on GSM8K

This walks you through running Vero on a single small dataset (**GSM8K**, grade-school
math word problems) so you can watch the optimizer improve an agent and inspect the
recorded results. GSM8K is chosen because it's the cheapest, fastest task: it targets
the `generic-agent`, scores with an LLM math-judge, and its dataset downloads from
HuggingFace only (no Kaggle/Modal credentials needed).

Everything below is run from the `vero-benchmarking/` directory.

---

## What you're about to do

1. Install deps and set API credentials.
2. Build the `aflow_gsm8k` dataset.
3. Sanity-check the setup (`vero check`).
4. (Optional) Smoke-test inference + scoring on 3 samples (`vero evaluate`).
5. Run a real optimization loop — the actual "Vero working."
6. Inspect the recorded results.

> **How it fits together:** `generic-agent` (in `../vero-agents/`) is the agent being
> optimized. The `gsm8k` task tells Vero which dataset, which agent, and how to score.
> Vero runs the agent on the data, scores each answer, then iteratively edits the
> agent's code (prompts, logic) to push the score up — recording each attempt.

---

## 1. Prerequisites

```bash
cd /Users/gregk/repos/scaleapi/vero/vero-benchmarking

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

Both agent inference and the LLM judge call models through a LiteLLM proxy. Export
these in your shell (or put them in a `.env` and `source` it):

```bash
export LITELLM_BASE_URL="https://your-litellm-proxy/v1"
export LITELLM_API_KEY="sk-..."
# OPENAI_API_KEY is used as a fallback if LITELLM_API_KEY is unset.
```

> These are the two vars declared as `required_env_vars` by the gsm8k task. They must
> be set **before `vero check`**, not just before inference: the agent's task module
> builds an OpenAI client at import time, and discovery imports it — so an unset key
> makes even `check` fail. W&B is **not** required for this demo (the low-level runner
> leaves it opt-in).

---

## 2. Build the dataset

```bash
uv run python -m vero_benchmarking.datasets --dataset-name aflow_gsm8k
```

This downloads GSM8K from HuggingFace, applies the AFLOW curated split mapping, and
saves a `DatasetDict` (train/validation/test) to `datasets/aflow_gsm8k/`.

---

## 3. Sanity-check the setup

```bash
uv run vero check \
  --project-path "$GENERIC_AGENT" \
  --task gsm8k \
  --dataset ./datasets/aflow_gsm8k
```

Expect all `[OK]` lines: uv project found, git repo, task discovery (`gsm8k: OK`),
env vars set, and the dataset splits with their sizes.

---

## 4. (Optional) Smoke test — 3 samples, no optimization

Fast and cheap. Confirms the inference → scoring contract works before you spend money
on a full loop. `--isolate` is important here: `generic-agent` lives inside the `vero`
monorepo, so this copies it into a fresh throwaway git repo to evaluate cleanly.

```bash
uv run vero evaluate \
  --project-path "$GENERIC_AGENT" \
  --task gsm8k \
  --dataset ./datasets/aflow_gsm8k \
  --split test \
  --num-samples 3 \
  --isolate \
  --task-params '{"model": "anthropic/claude-haiku-4-5-20251001"}'
```

You'll see per-sample scores (0.0–1.0) and judge feedback (`Expected: … Extracted
Prediction: … Is Equivalent: …`). No errors + sensible scores = you're ready.

---

## 5. Run the optimization loop

This is the real demo: Vero evaluates the baseline, then edits `generic-agent`'s code
over several rounds to improve its GSM8K score. Each run creates an isolated git
worktree branch of the agent, so your working tree is never touched.

```bash
uv run python -m vero_benchmarking.runner vero \
  --task gsm8k \
  --model anthropic/claude-haiku-4-5-20251001 \
  --max-turns 30
```

**Cost/time knobs** (the gsm8k task defaults to `train_budget=8`, `batch_size=512`,
which is a substantial paper-scale run):

- `--model anthropic/claude-haiku-4-5-20251001` — cheapest/fastest optimizer (default is Sonnet).
- `--max-turns 30` — caps how long the optimizer agent works per round.
- `--skip-initial-eval` — skip the baseline pass if you just want to see it iterate.
- `--env-file .env` / `--subprocess-env-file .env` — pass credentials via file instead of exporting.

To run through the higher-level batch harness instead (adds config presets but
**force-enables W&B**, so it needs `WANDB_API_KEY`):

```bash
uv run python scripts/run_benchmark.py --scaffold vero-default --model haiku --task gsm8k
```

When the run finishes it prints the **Session ID** and the **best commit** (the git ref
of the improved agent).

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
| `logs/session_manifest.jsonl` | One line per run: session id, task, agent type, best commit |
| Git worktree branch | A branch named like `vero-aflow_gsm8k-<id>` holding the optimized agent code — `git log`/`git diff` it to see exactly what Vero changed |
| `results/` | Parquet output (baseline-evaluation runs) |
| W&B | Metrics per run — only if you opted in (`--enable-wandb` / `run_benchmark.py`) |

To see the actual code changes Vero made, check out the best commit in the
`generic-agent` project and diff it against `main`.

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
- **`VIRTUAL_ENV ... does not match` warning** — harmless; uv ignores the stale
  `VIRTUAL_ENV`. `unset VIRTUAL_ENV` to silence it.
- **Missing env var errors** — `LITELLM_BASE_URL` and `LITELLM_API_KEY` must be set
  (see step 1); `vero check` reports exactly which are missing.
- **Dirty-tree / git errors during evaluate** — add `--isolate` (already included above).
- **Want a different task?** Swap `--task gsm8k` and the matching
  `--dataset ./datasets/<name>`. See the task table in `README.md`; build any dataset
  with `python -m vero_benchmarking.datasets --dataset-name <name>`.
