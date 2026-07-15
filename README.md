# VeRO: a harness for agents to optimize programs

[![Paper](https://img.shields.io/badge/arXiv-2602.22480-b31b1b.svg)](https://arxiv.org/abs/2602.22480)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

VeRO gives a coding agent a program to edit, an evaluation boundary, and durable
memory of every candidate it tried. The target can be an agent, a prompt, a
compiler pass, a CUDA kernel, a matrix multiplication function, or any other
Git-versioned program.

```text
strategy -> candidate producers -> evaluation backends -> selection
                  |                       |
          isolated workspaces       versioned reports
```

The target and evaluator do not need to be Python. External evaluators and
candidate producers connect through command protocols; Python benchmarks can
use the optional, optimizer-independent `scale-vero-tasks` package.

## Repository layout

| Directory | Purpose |
| --- | --- |
| [`vero/`](vero/) | The `scale-vero` optimization kernel, runtime, CLI, and coding-agent adapters |
| [`vero-tasks/`](vero-tasks/) | Narrow Python task types and schema-v1 evaluation runner |
| [`vero-agents/`](vero-agents/) | Benchmark target programs; not part of the optimizer runtime |
| [`vero-benchmarking/`](vero-benchmarking/) | Reproducible experiment configurations and drivers |

Start with the [generic C matrix-multiplication quickstart](vero/examples/c-matmul/)
or the [core guide](vero/README.md). The C target has no Python or VeRO
dependency; it is compiled, checked, benchmarked, and optimized through the
language-neutral command protocol.

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
the generic program optimizer continues on the `v0.5` branch.

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
