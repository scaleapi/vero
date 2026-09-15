# GAIA

GAIA evaluates a multimodal research agent that can search the web, inspect
files and images, run shell commands, and submit an exact answer.

## At a glance

| Item | Value |
| --- | --- |
| Task source | Pinned GAIA Harbor package |
| Editable harness | `baseline/target/` |
| Development / validation / test | 33 / 66 / 66 |
| Split strategy | GAIA level and attachment presence |
| Target model | `gpt-5.4-mini` |
| Pinned seed baselines | 0.6205 working target; 0.0 shell target; opencode target unpinned (measure before use) |
| Scoring | Canonical task verifier |

Development cases expose full results and task resources. Validation exposes
aggregate scores, and test remains hidden until final scoring.

## Variants

| Build | Starting point | Purpose |
| --- | --- | --- |
| `baseline/build.yaml` | Working tool-using agent | Measure harness improvement |
| `baseline/build.shell.yaml` | Minimal non-solving skeleton | Measure harness construction from scratch |
| `baseline/build.shell.e2e.yaml` | Skeleton with eight cases | Exercise the complete pipeline quickly |
| `baseline/build.opencode.yaml` | opencode, vendored as source and compiled per candidate | Measure improvement of a full coding harness |

The full and shell builds use the same tasks, model, budgets, and evaluation
policy. See [the baseline guide](baseline/README.md) for their editable surfaces
and build commands.

## Data and split

The 165 task references are pinned in
[`partitions/manifest.json`](partitions/manifest.json). To verify the committed
split against downloaded tasks, run from the repository root:

~~~bash
uv run --python 3.12 harness-opt-bench/gaia/scripts/partition_gaia.py \
  --tasks-dir <downloaded-gaia-tasks> \
  --check
~~~

Changing the dataset revision is a benchmark change. Update the source pin,
regenerate the partitions, and review the manifest before using new results.

## Where to look

| Path | Contents |
| --- | --- |
| `baseline/build*.yaml` | Benchmark variants and evaluation settings |
| `baseline/target/` | Working editable agent |
| `baseline/target-shell/` | Minimal editable skeleton |
| `baseline/target-opencode/` | opencode source (git submodule) plus a thin wrapper |
| `partitions/` | Reported development, validation, and test split |
| `partitions-e2e/` | Small smoke-test split |
| `scripts/partition_gaia.py` | Split verification and regeneration |
