# Vero Agents

A collection of Python-based agent implementations optimizable with [VeRO](../vero/). Each agent is a self-contained `uv` package with its own dependencies and evaluation tasks.

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

Each agent contains:
- `pyproject.toml` with dependencies and a dev dependency on `scale-vero`
- Source code under `src/<agent_name>/` or `<agent_name>/`
- A `vero_tasks/` module defining evaluation tasks for VeRO

## Adding a New Agent

1. Create a directory under `agents/`
2. Initialize a `uv` package with `pyproject.toml`
3. Add `scale-vero` as a dev dependency:
   ```toml
   [dependency-groups]
   dev = ["scale-vero[evaluate]"]

   [tool.uv.sources]
   scale-vero = { path = "../../../vero", editable = true }
   ```
4. Create a `vero_tasks/` module with inference and evaluation functions (see `vero init tasks`)
5. Register the task in `vero-benchmarking/src/vero_benchmarking/tasks/`

## Running Evaluations

Agents are evaluated through `vero-benchmarking`. See the [vero-benchmarking README](../vero-benchmarking/README.md) for details.

```bash
# Quick example: optimize generic-agent on MATH
cd ../vero-benchmarking
uv run python scripts/run_benchmark.py --scaffold claude-code-vmf --model sonnet --task math
```
