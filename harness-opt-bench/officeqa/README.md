# OfficeQA

OfficeQA evaluates a grounded question-answering agent over the Treasury
Bulletin corpus. The agent searches local documents and writes one exact answer.

## At a glance

| Item | Value |
| --- | --- |
| Cases | 246 |
| Development / validation / test | 49 / 98 / 99 |
| Editable harness | `baseline/target/` |
| Target model | `deepseek-v4-flash` |
| Pinned seed baseline | 0.3412 ± 0.0330 |
| Scoring | Numeric or text match, with 1% numeric tolerance |

Development exposes complete task results and corpus resources. Validation is
aggregate-only, and test is held out for final scoring.

## Prepare the tasks

OfficeQA task directories are too large to store in this repository. Fetch the
pinned dataset into the ignored `tasks/` directory:

~~~bash
cd harness-opt-bench/officeqa
bash scripts/vendor_tasks.sh
~~~

The script fetches only the OfficeQA subset, verifies that all 246 tasks are
present, and adds the canonical task names required by local staging.

Check task availability from the repository root:

~~~bash
python3 harness-opt-bench/scripts/task_data.py --check
~~~

## Editable target

`OfficeQaAgent` can search the corpus with shell commands, inspect images, and
submit an answer. The optimizer may change its prompts, tools, control flow,
context handling, and dependencies. The corpus, partitions, target model,
verifier, and scoring policy remain fixed.

## Compile

From the repository root:

~~~bash
cd vero
VERO_SKIP_SECRET_CHECK=1 uv run vero harbor build \
  --config ../harness-opt-bench/officeqa/baseline/build.yaml \
  --param inner_env=<evaluation-environment> \
  --output <output-directory>
~~~

The `VERO_SKIP_SECRET_CHECK` setting is appropriate only for compile-time
validation. For a real optimization run, follow the shared
[`run-benchmark` guide](../skills/run-benchmark/SKILL.md).

## Where to look

| Path | Contents |
| --- | --- |
| `baseline/build.yaml` | Evaluation configuration |
| `baseline/target/` | Editable seed agent |
| `partitions/` | Committed split and task manifest |
| `scripts/vendor_tasks.sh` | Dataset preparation |
