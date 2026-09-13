<p align="center">
  <img src="https://raw.githubusercontent.com/scaleapi/vero/main/vero/docs/assets/vero-banner.png" alt="Illustration of VeRO's iterative optimization loop" width="100%">
</p>

<h1 align="center">VeRO</h1>

<p align="center"><strong>A harness for agents to optimize programs, text, and agents</strong></p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.22480"><img src="https://img.shields.io/badge/arXiv-2602.22480-b31b1b.svg" alt="Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
</p>

> **Looking for the code from the VeRO paper?** See
> [Paper reproduction](#paper-reproduction) — reproduce from the `paper-v1`
> tag, or read the same code in place under [`legacy/`](legacy/).

VeRO gives a coding agent something to edit, an evaluation boundary, and durable
memory of every candidate it tried. The target is anything you can put under Git
and score — a **program** (a single function up to a whole codebase), **text**
(a prompt, spec, or config), or an **agent** (its scaffold, tools, and prompts).
VeRO was introduced to optimize agents, and the same version / evaluate / select
loop applies to any of these.

<p align="center">
  <img src="https://raw.githubusercontent.com/scaleapi/vero/main/vero/docs/assets/loop.png" alt="The VeRO loop: the optimizer proposes a candidate, the evaluator scores it, and the score and diagnostics return to the optimizer" width="720">
</p>

The optimizer is a coding agent, a command, or a custom strategy, running
locally or inside a Harbor container. The evaluator owns the cases and the
scoring, and every candidate it has scored stays selectable.
Every model call on both sides goes through the inference gateway, which holds
the provider key and meters spend in tokens against a per-scope budget.

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
| [`vero/`](vero/) | The `scaleapi-vero` optimization kernel, runtime, CLI, and coding-agent adapters |
| [`vero-tasks/`](vero-tasks/) | Narrow Python task types and schema-v1 evaluation runner |
| [`harness-opt-bench/`](harness-opt-bench/) | Harbor-native target programs and end-to-end optimization benchmarks |
| [`legacy/`](legacy/) | The pre-v0.5 tree, i.e. the original VeRO paper code — reference only, not used by the current system |

Start with the [generic C matrix-multiplication quickstart](vero/examples/c-matmul/),
try the [26-circle packing benchmark](vero/examples/circle-packing/), or read the
[core guide](vero/README.md). The C example demonstrates the language-neutral
command protocol without model credentials. Circle packing is a substantive
coding-agent benchmark with exact geometry checks and inspectable search
artifacts.

For a full agent-optimization example, the [GAIA benchmark](harness-opt-bench/gaia/)
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
[`harness-opt-bench/`](harness-opt-bench/).

Prefer the frozen ref for reproduction, since it is the state that was actually
published. Either way, note that both packages import as `vero` (`scale-vero`
0.4.7 in `legacy/vero`, `scaleapi-vero` 0.5.0 in `vero/`), so **they cannot be
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
