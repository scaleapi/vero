# VeRO: Versioning Rewards and Observations

[![Paper](https://img.shields.io/badge/arXiv-2602.22480-b31b1b.svg)](https://arxiv.org/abs/2602.22480)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

VeRO is an evaluation harness for using coding agents to optimize LLM-based agents and workflows. It treats agent code as a versioned artifact — making changes, evaluating results, and hill-climbing toward better performance using git version control.

> **Paper**: [VeRO: An Evaluation Harness for Agents to Optimize Agents](https://arxiv.org/abs/2602.22480)

## Repository Structure

```
vero/
├── vero/               # Core library (scale-vero)
├── vero-agents/        # Agent implementations (benchmarking targets)
├── vero-benchmarking/  # Benchmarking scripts and analysis
└── LICENSE
```

### [vero/](vero/)

The core optimization framework. Provides:

- **Policy** — orchestrates the optimization loop (agent + evaluator + git)
- **Agents** — VeroAgent (OpenAI Agents SDK) and ClaudeCodeAgent (Claude Agent SDK)
- **Evaluator** — runs task evaluations in isolated subprocess environments
- **Tools** — MCP-based tools for agents (bash, file I/O, experiment runner, dataset viewer, etc.)
- **Traces** — session analysis and LLM-based trace interpretation

```bash
cd vero && uv sync --extra optimize
```

See [vero/README.md](vero/README.md) for full documentation.

### [vero-agents/](vero-agents/)

Agent implementations used as optimization targets:

| Agent | Description |
|-------|-------------|
| **generic-agent** | General-purpose agent for MATH, GPQA, GAIA, GSM8K, etc. |
| **web_search_agent** | Web search agent for SimpleQA, Facts Search |
| **KIRA** | Terminal task agent for Terminal Bench 2.0 |
| **tau-bench** | Customer service tool-use agent |
| **pharma_summarizer** | Document summarization agent |

See [vero-agents/README.md](vero-agents/README.md) for details.

### [vero-benchmarking/](vero-benchmarking/)

Scripts and infrastructure for running optimization experiments:

```bash
cd vero-benchmarking && uv sync --all-extras

# Run an optimization experiment
uv run python scripts/run_benchmark.py --scaffold claude-code-vmf --model sonnet --task math

# Build datasets
./scripts/build_datasets.sh
```

See [vero-benchmarking/README.md](vero-benchmarking/README.md) for full documentation.

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Git
- Access to an LLM provider (via LiteLLM, OpenAI, Anthropic, etc.)

### Install

```bash
git clone <repo-url> && cd vero

# Install core library
cd vero && uv sync --extra optimize

# Install benchmarking tools
cd ../vero-benchmarking && uv sync --all-extras
```

### Run Your First Optimization

```python
from agents import Agent as OAIAgent
from vero.policy import Policy
from vero.agents.vero import VeroAgent

policy = Policy(
    project_path="/path/to/my-agent",
    dataset="/path/to/my-dataset",
    agent=VeroAgent(
        oai_agent=OAIAgent(name="VeroAgent", model="anthropic/claude-sonnet-4-5-20250929"),
    ),
    task="main",
    train_budget=10,
    max_turns=200,
)

best = await policy.run()
print(f"Best commit: {best.commit}, score: {best.score}")
```

## Citation

```bibtex
@article{ursekar2026vero,
  title={VeRO: An Evaluation Harness for Agents to Optimize Agents},
  author={Ursekar, Varun and Shanker, Apaar and Chatrath, Veronica and Xue, Yuan (Emily) and Denton, Sam},
  journal={arXiv preprint arXiv:2602.22480},
  year={2026}
}
```

## License

[MIT](LICENSE)
