# vero-benchmarking

Benchmark Vero on open-source datasets.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync --all-extras
```

### Environment Variables

| Variable | Description |
| -------- | ----------- |
| `VERO_AGENTS_PATH` | Path to vero-agents directory (default: `../vero-agents` in monorepo) |
| `LITELLM_BASE_URL` | LiteLLM proxy base URL |
| `LITELLM_API_KEY` | LiteLLM API key (also used as fallback for `OPENAI_API_KEY` in agents) |
| `OPENAI_API_KEY` | OpenAI API key (fallback if `LITELLM_API_KEY` not set) |
| `WANDB_API_KEY` | Weights & Biases API key (for experiment logging) |
| `MODAL_TOKEN_ID` | Modal token ID (required for Terminal Bench with modal environment) |
| `MODAL_TOKEN_SECRET` | Modal token secret (required for Terminal Bench with modal environment) |

## Project Structure

```text
vero-benchmarking/
├── src/vero_benchmarking/
│   ├── constants.py        # Default paths (datasets dir, seed)
│   ├── runner.py           # Policy factories, optimization runner, baseline eval, CLI
│   ├── eval.py             # Batch evaluation from CSV/manifest
│   ├── gepa.py             # GEPA adapter for evolutionary optimization
│   ├── datasets.py         # Dataset building logic
│   ├── utils.py            # Model helpers, path utilities
│   ├── tasks/              # Task definitions
│   │   ├── __init__.py     # ALL_TASKS registry, BENCHMARK_TASKS, load_task()
│   │   ├── base.py         # OptimizationTask dataclass
│   │   ├── aflow.py        # AFLOW benchmark tasks (math, gsm8k, etc.)
│   │   ├── gaia.py         # GAIA task
│   │   ├── gpqa.py         # GPQA Diamond task
│   │   ├── simple_qa.py    # SimpleQA task
│   │   ├── tau_bench.py    # Tau Bench task
│   │   ├── facts_search.py # Facts Search task
│   │   └── terminal_bench.py # Terminal Bench 2.0 task
│   ├── analysis/           # Post-hoc analysis, plotting, W&B extraction
│   └── static_data/        # Static JSON files for dataset building
├── datasets/               # Built datasets (default output)
├── scripts/
│   ├── run_benchmark.py      # Batch experiment runner (scaffolds, configs)
│   ├── run_terminal_bench.py # Terminal Bench 2.0 optimization
│   └── build_datasets.sh    # Build all datasets
└── notebooks/              # Analysis notebooks
```

## Tasks

Tasks define what to optimize. Each task specifies a project path, dataset path, and evaluation budgets.

### Benchmark Tasks

These are the canonical tasks used for paper results:

| Task | Registry Key | Dataset | Agent Project |
| ---- | ------------ | ------- | ------------- |
| GAIA | `gaia` | GAIA pure language | generic-agent |
| GPQA Diamond | `gpqa-nosplit` | GPQA Diamond (no val split) | generic-agent |
| MATH | `math` | AFLOW MATH | generic-agent |
| SimpleQA | `simpleqa` | SimpleQA Verified Wiki | web_search_agent |
| Tau Bench | `tau-bench` | Tau Bench Retail | tau-bench |

### Other Tasks

| Task | Registry Key | Dataset | Agent Project |
| ---- | ------------ | ------- | ------------- |
| DROP | `drop-single` | AFLOW DROP (single answer) | generic-agent |
| GSM8K | `gsm8k` | AFLOW GSM8K | generic-agent |
| HotpotQA | `hotpotqa` | AFLOW HotpotQA | generic-agent |
| HumanEval | `humaneval-nosplit` | AFLOW HumanEval | generic-agent |
| MBPP | `mbpp` | AFLOW MBPP | generic-agent |
| Facts Search | `facts-search` | Facts Search | web_search_agent |
| Terminal Bench | `terminal-bench` | Terminal Bench 2.0 | KIRA |

Tasks are defined in `src/vero_benchmarking/tasks/` and registered in `tasks/__init__.py`.

## Datasets

### Building All Datasets

```bash
./scripts/build_datasets.sh
```

### Building a Single Dataset

```bash
uv run python -m vero_benchmarking.datasets --dataset-name <name>
```

Dataset names correspond to the builder classes in `datasets.py` (e.g. `aflow_math`, `gpqa_diamond_no_split`, `simple_qa_verified_wiki_unanswered`, `tau_bench_retail`, `gaia_pure_language`).

## Scaffolds

Scaffolds define optimizer configurations (agent type + tool sets + instructions):

| Scaffold | Agent | Description |
| -------- | ----- | ----------- |
| `vero-default` | VeroAgent | Default settings |
| `vero-prompts-only` | VeroAgent | Restricted to resource edits only |
| `vero-cookbook` | VeroAgent | With pre-loaded Agent Cookbook skills |
| `vero-orchestrator` | VeroAgent | Orchestrator mode (sub-agents only) |
| `vero-orchestrator-cookbook` | VeroAgent | Orchestrator + cookbook |
| `claude-code-vmf` | ClaudeCodeAgent | With Vero measurement tools |
| `claude-code-vmf-cookbook` | ClaudeCodeAgent | VMF + cookbook |
| `claude-code-pure` | ClaudeCodeAgent | Pure Claude Code (no Vero tools) |

## Models

| Short Name | Full Model |
| ---------- | ---------- |
| `sonnet` | `anthropic/claude-sonnet-4-5-20250929` |
| `opus` | `anthropic/claude-opus-4-5-20251101` |
| `haiku` | `anthropic/claude-haiku-4-5-20251001` |
| `gpt` | `gpt-5.2-codex` |

## Running Experiments

### Batch CLI (run_benchmark.py)

```bash
# List available scaffolds, models, configs, and tasks
uv run python scripts/run_benchmark.py --list

# Run a specific pre-defined config on a task
uv run python scripts/run_benchmark.py --config vero-cookbook-sonnet --task math

# Run all default configs on a task
uv run python scripts/run_benchmark.py --all-configs --task math

# Run a scaffold with a specific model
uv run python scripts/run_benchmark.py --scaffold vero-orchestrator-cookbook --model haiku --task gpqa-nosplit

# Dry run (preview what would run)
uv run python scripts/run_benchmark.py --all-configs --task math --dry-run

# Run with multiple iterations
uv run python scripts/run_benchmark.py --config vero-cookbook-sonnet --task math -n 3
```

### Low-Level CLI (runner.py)

```bash
# VeroAgent optimization
uv run python -m vero_benchmarking.runner vero --task math

# ClaudeCodeAgent optimization
uv run python -m vero_benchmarking.runner claude-code --task math

# Baseline evaluation (no optimization)
uv run python -m vero_benchmarking.runner baseline --task math --models openai/gpt-4.1-mini-2025-04-14
```

### GEPA Optimization

GEPA optimizes `@resource()` decorated functions using evolutionary reflection-mutation.

```bash
uv run python -m vero_benchmarking.gepa --task math --model sonnet --skip-initial-eval
```

## Reproducing Paper Results

### 1. Build Datasets

```bash
./scripts/build_datasets.sh
```

### 2. Run All Configs on All Benchmark Tasks (3 iterations each)

```bash
for task in gaia gpqa-nosplit math simpleqa tau-bench; do
    uv run python scripts/run_benchmark.py \
        --all-configs \
        --task "$task" \
        -n 3 \
        --batch-id paper-results \
        --continue-on-error \
        --push-to-origin
done
```

This runs 7 configs x 5 tasks x 3 iterations = 105 experiments. The `--batch-id` flag enables resume support: if a run is interrupted, re-running the same command will skip already-completed experiments.

### 3. Review Results

Results are tracked in:
- **Wandb**: Experiment metrics logged per run (enabled by default in `run_benchmark.py`)
- **Session artifacts**: `~/.vero/sessions/{session_id}/` contains experiments DB, config, and run logs
- **Batch manifest**: `logs/batch_manifests/{batch_id}.jsonl` records completed experiments
- **Git branches**: Each run creates a worktree branch like `{repo}-{dataset}-{random_id}`

## Outputs

- **Sessions**: `~/.vero/sessions/{session_id}/` contains experiments, config, and run results
- **Wandb**: Experiment metrics logged to Weights & Biases (if `--enable-wandb`)
- **Git branches**: Each optimization run creates a worktree branch like `{repo}-{dataset}-{random_id}`
- **Batch manifests**: `logs/batch_manifests/{batch_id}.jsonl` tracks completed experiments for resume support
