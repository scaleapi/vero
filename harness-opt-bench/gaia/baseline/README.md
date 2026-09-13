# GAIA editable targets

This directory contains two starting points for the same GAIA evaluation.

| Target | Build file | Starting behavior |
| --- | --- | --- |
| `target/` | `build.yaml` | Searches, uses the task shell, inspects images, and submits an answer |
| `target-shell/` | `build.shell.yaml` | Loads successfully but submits an empty answer |

The optimizer may change prompts, tools, control flow, file handling, and
dependencies inside the selected target. The dataset, partitions, target model,
verifier, and evaluation policy remain fixed.

## Why keep a shell target?

The working seed measures improvement to an existing harness. The shell target
measures whether an optimizer can build the harness itself. It deliberately
submits an empty answer so every case receives a valid zero rather than becoming
an evaluation error.

The shell-specific instruction extends the shared task instructions and changes
only the opening frame from “improve” to “build, then optimize.” Design notes stay
outside the editable target so they do not become hints.

## Compile a build

From the repository root:

~~~bash
cd vero
uv run vero harbor build \
  --config ../harness-opt-bench/gaia/baseline/build.yaml \
  --param inner_env=<evaluation-environment> \
  --output <output-directory>
~~~

Use `build.shell.yaml` in place of `build.yaml` to compile the shell variant.

For a real optimization run, follow the shared
[`run-benchmark` guide](../../skills/run-benchmark/SKILL.md). Dataset and split
details are in the [GAIA overview](../README.md).
