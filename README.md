# VeRO: a harness for agents to optimize programs, text, and agents

[![Paper](https://img.shields.io/badge/arXiv-2602.22480-b31b1b.svg)](https://arxiv.org/abs/2602.22480)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

VeRO gives a coding agent something to edit, an evaluation boundary, and durable
memory of every candidate it tried. The target is anything you can put under Git
and score — a **program** (a single function up to a whole codebase), **text**
(a prompt, spec, or config), or an **agent** (its scaffold, tools, and prompts).
Agents are programs, but not everyone reads "program" that way, so VeRO names
them explicitly: it was introduced to optimize agents and generalizes the same
version / evaluate / select loop to any Git-versioned artifact.

```text
strategy -> candidate producers -> evaluation backends -> selection
                  |                       |
          isolated workspaces       versioned reports
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
| [`program-opt-bench/`](program-opt-bench/) | Harbor-native target programs and end-to-end optimization benchmarks |

Start with the [generic C matrix-multiplication quickstart](vero/examples/c-matmul/),
try the [26-circle packing benchmark](vero/examples/circle-packing/), or read the
[core guide](vero/README.md). The C example demonstrates the language-neutral
command protocol without model credentials. Circle packing is a substantive
coding-agent benchmark with exact geometry checks and inspectable search
artifacts.

For a full agent-optimization example, the [GAIA benchmark](program-opt-bench/gaia/)
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

The exact paper implementation is frozen separately from the v0.5 redesign:

```bash
git checkout paper-v1
```

Use the `paper/v1` branch or `paper-v1` tag for reproduction. Development of
the generic program optimizer continues on the `v0.5` branch. The frozen ref
also preserves the paper-era `vero-agents` and `vero-benchmarking` directories;
their Harbor-native replacement on `v0.5` is `program-opt-bench`.

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
