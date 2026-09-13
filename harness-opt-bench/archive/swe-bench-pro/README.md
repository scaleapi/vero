# SWE-bench-Pro

SWE-bench-Pro evaluates a code-editing agent. Each task checks out a project,
describes a change, and grades the edited repository with hidden tests.

## At a glance

| Item | Value |
| --- | --- |
| Task source | `swebenchpro@1.0` |
| Cases | 731 across 11 projects |
| Development / validation / test | 146 / 292 / 293 |
| Editable harness | `baseline/target/` |
| Split strategy | Source repository |
| Status | Archived; not part of the reported suite |

The dataset version and `partitions/manifest.json` pin the task definitions.
Development resources are not exposed through VeRO for this registry-backed
dataset. Each task grades itself without a judge model.

## Configurations

| Build | Split | Intended use |
| --- | --- | --- |
| `baseline/build.yaml` | 146 / 292 / 293 | Full benchmark |
| `baseline/build.sample.yaml` | 33 / 66 / 66 | Shorter optimizer runs |

The sample is nested inside the full split, so no test case moves into
development. It remains an experimental configuration with a provisional
baseline.

## Verify the split

~~~bash
uv run --python 3.12 \
  python harness-opt-bench/archive/swe-bench-pro/scripts/partition_swe_bench_pro.py \
  --tasks-dir <exported-dataset> \
  --check
~~~

## Editable target

The agent in `baseline/target/` inspects and edits the checked-out project. The
optimizer may change its prompt, tools, control flow, context handling, and
dependencies. The task source, partitions, target model, and hidden tests remain
fixed.

## Compile

From the repository root:

~~~bash
cd vero
VERO_SKIP_SECRET_CHECK=1 uv run vero harbor build \
  --config ../harness-opt-bench/archive/swe-bench-pro/baseline/build.yaml \
  --param inner_env=<evaluation-environment> \
  --output <output-directory>
~~~

The `VERO_SKIP_SECRET_CHECK` setting is appropriate only for compile-time
validation. For a real optimization run, follow the shared
[`run-benchmark` guide](../../skills/run-benchmark/SKILL.md).
