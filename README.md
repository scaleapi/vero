# VeRO: a harness for agents to optimize programs, text, and agents
> **Looking for the code from the VeRO paper?** See
> [Paper reproduction](#paper-reproduction) — reproduce from the `paper-v1`
> tag, or read the same code in place under [`legacy/`](legacy/).

[![Paper](https://img.shields.io/badge/arXiv-2602.22480-b31b1b.svg)](https://arxiv.org/abs/2602.22480)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

VeRO gives a coding agent something to edit, an evaluation boundary, and durable
memory of every candidate it tried. The target is anything you can put under Git
and score — a **program** (a single function up to a whole codebase), **text**
(a prompt, spec, or config), or an **agent** (its scaffold, tools, and prompts).
VeRO was introduced to optimize agents, and the same version / evaluate / select
loop applies to any of these.

```
  ┌─────────────────────────┐   submit candidate    ┌─────────────────────────┐
  │  candidate production   ├──────────────────────►│  evaluation service     │
  │                         │                       │                         │
  │  coding agent, command, │◄──────────────────────┤  owns cases + scoring   │
  │  or custom strategy;    │  score + diagnostics  │                         │
  │  edits its own Git      │                       │  development: may ask   │
  │  worktree per candidate │                       │  validation:  aggregate │
  └───────────┬─────────────┘                       │  test:        withheld  │
              │ commit                              └───────────┬─────────────┘
              ▼                                                 │ report
  ┌─────────────────────────┐    next round     ┌───────────────▼─────────────┐
  │  candidate history:     │◄──────────────────┤  selection: keep the best   │
  │  every version kept,    │                   │  feasible candidate         │
  │  each one re-selectable │                   └─────────────────────────────┘
  └─────────────────────────┘

  Every model call on both sides goes through the inference gateway, which holds
  the provider key and meters spend in tokens against a per-scope budget.
```

The target and evaluator do not need to be Python. External evaluators and
candidate producers connect through command protocols; Python benchmarks can
use the optional, optimizer-independent `scale-vero-tasks` package.

Targets may live locally or in an isolated sandbox. VeRO keeps optimization
state and experiment tracking on the host while running Git worktrees, producer
commands, builds, and evaluation commands in the target sandbox. The core guide
includes a no-bind-mount `DockerSandbox` example.

## Repository layout

| Directory | Purpose |
| --- | --- |
| [`vero/`](vero/) | The `scale-vero` optimization kernel, runtime, CLI, and coding-agent adapters |
| [`vero-tasks/`](vero-tasks/) | Narrow Python task types and schema-v1 evaluation runner |
| [`harness-engineering-bench/`](harness-engineering-bench/) | Harbor-native target programs and end-to-end optimization benchmarks |
| [`legacy/`](legacy/) | The pre-v0.5 tree, i.e. the original VeRO paper code — reference only, not used by the current system |

Start with the [generic C matrix-multiplication quickstart](vero/examples/c-matmul/),
try the [26-circle packing benchmark](vero/examples/circle-packing/), or read the
[core guide](vero/README.md). The C example demonstrates the language-neutral
command protocol without model credentials. Circle packing is a substantive
coding-agent benchmark with exact geometry checks and inspectable search
artifacts.

For a full agent-optimization example, the [GAIA benchmark](harness-engineering-bench/gaia/)
pairs a tool-using GPT-5.4 mini target with Harbor's canonical GAIA verifier and
an immutable 20% / 40% / 40% development, validation, and test split.

```bash
cd vero
uv sync --all-extras
uv run vero --help
```

## Paper reproduction

VeRO was introduced in [*VeRO: A Harness for Agents to Optimize
Agents*](https://arxiv.org/abs/2602.22480), accepted at ICML 2026. The current
library generalizes that version/evaluate/select loop from agents to programs.

The paper-era code is available two ways, and they hold the same code:

**To reproduce the paper, use the frozen ref.** It is the repository exactly as
it stood at publication, with the original paths intact:

```bash
git checkout paper-v1          # tag; the paper/v1 branch points at the same commit
```

**To read that code alongside the current system, use [`legacy/`](legacy/).** The
v0.5 redesign relocated the paper-era tree into that directory rather than
deleting it, so it sits in this branch next to the code that replaced it. Both
the frozen ref and `legacy/` include the paper-era `vero-agents` and
`vero-benchmarking` directories; their Harbor-native replacement is
[`harness-engineering-bench/`](harness-engineering-bench/).

Prefer the frozen ref for reproduction, since it is the state that was actually
published. Either way, note that both packages are named `scale-vero` and both
import as `vero` (0.4.7 in `legacy/vero`, 0.5.0 in `vero/`), so **they cannot be
installed into the same environment** — give the legacy package its own
virtualenv. Development of the current system continues on `main`.

## Citation

```bibtex
@article{ursekar2026vero,
  title={VeRO: A Harness for Agents to Optimize Agents},
  author={Ursekar, Varun and Shanker, Apaar and Chatrath, Veronica and Xue, Yuan (Emily) and Denton, Sam},
  journal={arXiv preprint arXiv:2602.22480},
  year={2026}
}
```

VeRO is licensed under the [MIT License](LICENSE).
