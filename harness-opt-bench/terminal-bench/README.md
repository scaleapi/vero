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
| Editable harness | `baseline/target/` |
| Target model | `grok-build-0.1` |
| Pinned seed baseline | 0.2407 ± 0.0131 |

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
| `baseline/build.yaml` | Standard benchmark |
| `baseline/build.routed.yaml` | Same benchmark with one explicit optimizer-model alias |

The configuration test ensures the routed variant differs only by that alias.

Compile from the repository root:

~~~bash
cd vero
VERO_SKIP_SECRET_CHECK=1 uv run vero harbor build \
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
