# vero-benchmarking

`vero-benchmarking` connects the VeRO program-optimization runtime to the target
programs and datasets in `vero-agents`. It owns experiment configuration and
result collection; targets depend only on the narrow `scale-vero-tasks`
protocol, not on the optimizer.

For exact ICML 2026 paper reproduction, use the repository's `paper/v1` branch
or `paper-v1` tag. This directory follows the breaking v0.5 architecture.

## Setup

From this directory:

```bash
uv sync --all-extras
./scripts/build_datasets.sh
```

By default, the target repository is the sibling `../vero-agents` directory.
Set `VERO_AGENTS_PATH` to use another checkout. Evaluation and coding-agent
credentials are passed through the environment, including `LITELLM_API_KEY`,
`LITELLM_BASE_URL`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY`.

## Run a benchmark

Evaluate a baseline without starting a coding agent:

```bash
uv run python -m vero_benchmarking.runner baseline \
  --task math \
  --models gpt
```

Optimize the same target with either built-in coding-agent adapter:

```bash
uv run python -m vero_benchmarking.runner optimize \
  --task math \
  --agent vero \
  --model sonnet

uv run python -m vero_benchmarking.runner optimize \
  --task math \
  --agent claude \
  --model sonnet
```

The baseline consumes one evaluation run. `--max-candidates` controls the
number of optimization proposals; trial checkpoints requested by an agent use
the same durable evaluation budget.

For repeatable batches:

```bash
uv run python scripts/run_benchmark.py --list
uv run python scripts/run_benchmark.py \
  --config vero-sonnet \
  --task math \
  --batch-id july-benchmarks \
  -n 3
```

Batch manifests make reruns resumable. Budget ablations use the same runtime:

```bash
uv run python scripts/run_budget_ablation.py \
  --config vero-sonnet \
  --task math \
  --budgets 2 4 8 16
```

## Evaluate historical candidates

`vero_benchmarking.eval` accepts a CSV or Parquet manifest with `task`, `model`,
`commit`, and `split` columns. Each commit is mounted in a temporary Git
worktree and evaluated through the canonical backend:

```bash
uv run python -m vero_benchmarking.eval --input evaluations.csv -n 3
```

Per-case Parquet results, aggregate summaries, and canonical session state are
written under the selected output directory.

## Architecture

```text
OptimizationTask
    ├── target Git project in vero-agents
    ├── immutable materialized JSONL cases
    ├── one EvaluationSet and objective
    └── run and case budgets
             │
             ▼
minimal evaluator project + editable candidate overlay
             │
             ▼
schema-v1 EvaluationRecord + durable OptimizationSession
```

The small [`evaluator`](evaluator/) project is the subprocess harness. It
depends only on `scale-vero-tasks`, so evaluation does not install VeRO's coding
agents, notebooks, Docker integrations, or analysis stack. Dataset snapshots
are materialized into the session outside the editable target repository.

Current benchmark task adapters are still imported from the editable target
packages. This preserves the existing benchmark implementations during the
v0.5 migration, but it is not an adversarial security boundary: an optimizer
could edit an adapter along with the target. Scorers should move into the
external evaluator project before these benchmarks are treated as tamper-proof.
The matrix-multiplication example in `vero/examples/matmul-kernel` demonstrates
the fully external scorer layout.

## Tasks

The main registry contains `gaia`, `gpqa-nosplit`, `math`, `simpleqa`, and
`tau-bench`; additional tasks include `drop-single`, `gsm8k`, `hotpotqa`,
`humaneval-nosplit`, `mbpp`, `facts-search`, and `terminal-bench`.

An `OptimizationTask` specifies:

- the target project and Python task module;
- the dataset and single optimization partition;
- the objective metric and direction;
- total evaluation runs and optional case limits; and
- evaluation parameters passed to the target program.

The old independent train/validation budgets are intentionally gone. A v0.5
session optimizes one explicit, immutable `EvaluationSet`; a separate set or
holdout is evaluated in a separate session.

## Outputs

Canonical sessions default to
`$VERO_HOME/sessions/benchmarks/<task>/<session-id>` (or
`~/.vero/sessions/benchmarks/...`). Each session contains:

```text
manifest.json
database.json
budgets.json
events.jsonl
inputs/cases.jsonl
artifacts/
evaluations/
```

Candidate commits remain addressable by Git hash. VeRO does not push branches
or results to external services automatically.
