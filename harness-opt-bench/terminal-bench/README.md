# Terminal-Bench 2.1

Terminal-Bench evaluates a shell-based agent on long-running command-line tasks.
Each task grades the final container state with its own tests, so the benchmark
score is a pass rate.

## At a glance

| Item | Value |
| --- | --- |
| Task source | Terminal-Bench 2.1, pinned by content digest |
| Cases | 89 |
| Development / validation / test | 17 / 36 / 36 |
| Editable harness | `baseline/target/` (working seed); `baseline/target-shell/` (skeleton); `baseline/target-opencode/` (opencode source) |
| Target model | `grok-build-0.1` |
| Pinned seed baselines | 0.2407 ± 0.0131 working target; 0.0 shell target |

Development exposes complete results and task resources. Validation is
aggregate-only, and test is held out for final scoring.

## Dataset and split

The build uses:

~~~text
terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a
~~~

Version 2.1 repairs task definitions and dependencies from 2.0. No local task
vendoring is required.

The split is deterministic and stratified by category and difficulty. To verify
it against an exported dataset:

~~~bash
uv run --project vero --with 'harbor==0.20.0' \
  python harness-opt-bench/scripts/partition_dataset.py terminal-bench \
  --tasks-dir <exported-tasks> \
  --output-dir harness-opt-bench/terminal-bench/partitions \
  --check
~~~

## Editable target

`TerminalBenchAgent` runs a function-calling shell loop with linear history and
a fixed step budget. The optimizer may change its prompt, tools, control flow,
context handling, and dependencies. Dataset tasks, task tests, partitions,
target model, and scoring remain fixed.

Keep design rationale outside `baseline/target/`: the optimizer can read that
directory, so implementation advice placed there would become part of the task.

## Evaluation notes

Terminal-Bench tasks declare different time limits. Equal
`case_timeout_seconds` and `task_agent_timeout_seconds` preserve those
task-specific limits through Harbor's timeout multiplier.

Final scoring runs each test case three times. Candidate-caused failures receive
zero; evaluation-system failures are excluded. The pinned baseline includes a
known seed decoding failure, so fixing that behavior is valid optimization
headroom and requires no special scoring rule.

## Build variants

| File | Purpose |
| --- | --- |
| `baseline/build.yaml` | Standard benchmark: improve the working seed agent |
| `baseline/build.routed.yaml` | Same benchmark with one explicit optimizer-model alias |
| `baseline/build.shell.yaml` | Non-solving skeleton seed: build the agent from scratch |
| `baseline/build.opencode.yaml` | Full-harness seed: opencode, vendored as source in `baseline/target-opencode/opencode/` and compiled per candidate |

The configuration tests ensure the routed variant differs only by that alias,
and that the shell variant shares the tasks, model, budgets and evaluation policy
of the standard build and differs only in its seed and framing. The shell seed
runs no commands, so its baseline is zero by construction; confirm it with one
baseline round before quoting deltas against it.

Together the three seeds form a ladder of starting complexity -- nothing, a
hand-written loop, a full coding harness -- on the same tasks, model and budget.
The opencode seed is a git submodule pinned to one upstream commit; run
`git submodule update --init` before compiling. Its wrapper subclasses harbor's
opencode runner and, instead of installing the npm release, compiles the vendored
tree on the evaluation host (Bun, cached by source hash) and uploads the binary
into each task container, so edits anywhere in the source are what gets scored.

Compile from the repository root:

~~~bash
cd vero
uv run vero harbor build \
  --config ../harness-opt-bench/terminal-bench/baseline/build.yaml \
  --param inner_env=<evaluation-environment> \
  --output <output-directory>
~~~

For a real optimization run, follow the shared
[`run-benchmark` guide](../skills/run-benchmark/SKILL.md).

## Sources

- [Terminal-Bench 2.1 announcement](https://www.tbench.ai/news/terminal-bench-2-1)
- [Dataset](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1)
- [Repository](https://github.com/harbor-framework/terminal-bench)
