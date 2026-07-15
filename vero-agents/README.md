# VeRO benchmark targets

This directory is VeRO's benchmark corpus: Python programs used as optimization
targets in the paper and in end-to-end tests. Agents are one important kind of
target program, but they are not part of the optimizer runtime.

Each target is a self-contained uv project. It depends on the narrow
[`scale-vero-tasks`](../vero-tasks/) protocol for Python evaluation ergonomics,
not on `scale-vero` itself.

## Agents

| Agent | Description | Benchmark Tasks |
|-------|-------------|-----------------|
| **generic-agent** | OpenAI Agents SDK agent for general-purpose benchmarks | MATH, GPQA, GAIA, GSM8K, HotpotQA, HumanEval, DROP, MBPP |
| **web_search_agent** | Web search agent using OpenAI Agents SDK | SimpleQA, Facts Search |
| **KIRA** | Terminal task agent built on Harbor/Terminus | Terminal Bench 2.0 |
| **tau-bench** | Tool-augmented customer service agent | Tau Bench Retail |
| **pharma_summarizer** | Pharmaceutical document summarization agent | Pharma Summarizer |

## Structure

```
vero-agents/
├── agents/
│   ├── generic-agent/       # General-purpose LLM agent
│   ├── web_search_agent/    # Web search + retrieval agent
│   ├── KIRA/                # Terminal Bench agent (Harbor-based)
│   ├── tau-bench/           # Customer service tool-use agent
│   └── pharma_summarizer/   # Document summarization agent
└── pyproject.toml
```

Each target contains:

- `pyproject.toml` with its own runtime dependencies and `scale-vero-tasks`
- Source code under `src/<agent_name>/` or `<agent_name>/`
- A `vero_tasks/` module defining inference and scoring functions

## Adding a new target

1. Create a directory under `agents/`
2. Initialize a `uv` package with `pyproject.toml`
3. Add the task protocol:

   ```toml
   [project]
   dependencies = ["scale-vero-tasks>=0.1.0"]

   [tool.uv.sources]
   scale-vero-tasks = { path = "../../../vero-tasks", editable = true }
   ```
4. Create a `vero_tasks/` module with inference and evaluation functions
5. Register the task in `vero-benchmarking/src/vero_benchmarking/tasks/`

## Running Evaluations

Agents are evaluated through `vero-benchmarking`. See the [vero-benchmarking README](../vero-benchmarking/README.md) for details.

```bash
# Quick example: optimize generic-agent on MATH
cd ../vero-benchmarking
uv run python scripts/run_benchmark.py --scaffold claude-code-vmf --model sonnet --task math
```
