# Matrix Multiply Kernel Optimization

End-to-end example of vero optimizing a simple Python function for speed.

## What's here

Two packages:

- **matmul-kernel/** — The agent project. Contains a naive `multiply()` function that the optimizer will improve.
- **matmul-eval/** — The evaluation task. Measures correctness and execution time (score = avg ms per call, lower is better).

`run.py` copies both into a temp directory, initializes git, creates a dataset of test matrices, and runs the optimization.

## Quick start

```bash
# Evaluate the baseline kernel (no optimization, no LLM needed)
uv run run.py --eval-only

# Run full optimization with VeroAgent
uv run run.py

# Use Claude Code agent instead
uv run run.py --agent claude-code
```

## Modes

### `--eval-only`

Just evaluate the current kernel implementation — no LLM, no optimization loop. Useful for verifying the evaluation pipeline works.

```bash
uv run run.py --eval-only
```

### Default (no flags)

Full optimization loop with VeroAgent (OpenAI Agents SDK). The agent gets the default tool set: `BashTool`, `FileRead`, `FileWrite`, `Grep`, `GitViewer`, `GitControl`, `DatasetViewer`, `ExperimentViewer`, `ExperimentRunnerTool`, `SubAgentTool`, `TodoList`.

The agent can read dataset samples via `DatasetViewer`, view past experiment results via `ExperimentViewer`, edit files directly with `FileWrite`, and run evaluations with `ExperimentRunnerTool`.

```bash
uv run run.py
```

### `--artifacts`

Replace `DatasetViewer` and `ExperimentViewer` with filesystem artifacts. Dataset samples are materialized as JSON files in `_vero/datasets/`, and experiment traces are written to `_vero/traces/` after each evaluation. The agent reads these with `FileRead` / `BashTool` instead of dedicated viewer tools.

This is useful when you want the agent to have file-based access to data rather than structured tool calls.

```bash
uv run run.py --artifacts
```

### `--resources`

Use `ResourceControl` instead of `FileWrite`. The `multiply()` function is decorated with `@resource("kernel")`, so the agent edits it by resource name (`kernel.multiply`) rather than by file path. This constrains the agent to only modify registered resources.

```bash
uv run run.py --resources
```

### `--agent claude-code`

Use the Claude Code agent (Claude Agent SDK) instead of VeroAgent. Automatically enables filesystem artifacts since Claude Code reads files natively.

```bash
uv run run.py --agent claude-code
```

### `--work-dir`

Specify a working directory instead of creating a temp dir.

```bash
uv run run.py --work-dir ./my-run
```

## Flags can be combined

```bash
# Artifacts + resources + Claude Code
uv run run.py --artifacts --resources --agent claude-code

# Evaluate only, custom work dir
uv run run.py --eval-only --work-dir ./my-run
```

## What the agent does

1. Reads the naive `multiply()` implementation
2. Runs an initial evaluation to get the baseline score
3. Modifies the kernel (e.g., replaces with numpy, optimizes the algorithm)
4. Commits the change and runs evaluation
5. Iterates up to the budget (5 evaluation runs)

The score is average execution time in milliseconds across the test matrix sizes. Incorrect results get a penalty score of 999999.0.
