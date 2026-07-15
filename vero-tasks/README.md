# scale-vero-tasks

`scale-vero-tasks` is the optional Python task protocol used by VeRO benchmark
targets. It contains no optimizer, workspace, session, dataset-store, or
experiment-database code.

```python
from vero_tasks import TaskOutput, TaskResult, create_task

task = create_task("exact_match")

@task.inference()
async def infer(case, context):
    return TaskOutput(output=case["question"].upper())

@task.evaluation()
async def evaluate(case, output, context):
    return TaskResult.from_task_output(
        output,
        score=float(output.output == case["answer"]),
    )
```

The runner accepts a VeRO command-evaluation request and an external JSON/JSONL
case file, imports a task module, and writes a schema-v1 evaluation report. The
standard VeRO adapter runs this package in a trusted evaluator project while
overlaying the candidate package as an editable dependency. This keeps Python
benchmark ergonomics separate from both the target program and VeRO's
language-neutral evaluation kernel.
