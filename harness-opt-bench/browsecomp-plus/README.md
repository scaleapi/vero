# BrowseComp-Plus

BrowseComp-Plus evaluates a research agent against a fixed document corpus and
BM25 index. The agent searches local evidence rather than the live web.

## At a glance

| Item | Value |
| --- | --- |
| Available queries | 830 |
| Evaluated cases | 165 |
| Development / validation / test | 33 / 66 / 66 |
| Editable harness | `baseline/target/` |
| Target model | `deepseek-v4-flash` |
| Pinned seed baseline | 0.4619 ± 0.0283 |
| Search data | Pinned corpus and BM25 index |
| Scoring | Canonical semantic answer evaluator |

Development exposes complete task results. Validation is aggregate-only, and
test is held out for final scoring.

## Pinned inputs

| Input | Revision |
| --- | --- |
| BrowseComp-Plus repository | `046949032b0328319cc9a02663a759ec601d9402` |
| `Tevatron/browsecomp-plus` | `144cff8e35b5eaef7e526346aa60774a9deb941f` |
| `Tevatron/browsecomp-plus-indexes` | `b3f37f70c33829eb09d04784a54277a31871fd63` |

The task builder rejects a different repository revision. The dataset and index
revisions are immutable inputs to the benchmark.

## Prepare the tasks

~~~bash
git submodule update --init harness-opt-bench/browsecomp-plus/upstream

cd harness-opt-bench/browsecomp-plus
uv run --no-project --python 3.12 --with datasets==4.0.0 -- \
  python scripts/build_tasks.py
~~~

The command writes all 830 generated tasks to the ignored `tasks/` directory.
It assigns a deterministic 165-case subset to the committed partitions and
leaves the other tasks unassigned. Use `--check` to verify an existing tree or
`--force` to replace one.

The first task-image build also retrieves the pinned search index. Later tasks
reuse the same image.

## Evaluation

Answers contain an explanation, exact answer, and confidence value. The
task-owned evaluator applies the benchmark's pinned judge prompt and model. The
editable target has access only to the local search index and document reader.

See [the baseline guide](baseline/README.md) for the editable surface and build
command.

## Where to look

| Path | Contents |
| --- | --- |
| `baseline/` | Build configuration and editable target |
| `partitions/` | Committed split and task manifest |
| `scripts/build_tasks.py` | Deterministic task generator |
| `task-template/` | Shared task environment and verifier |
