from __future__ import annotations

import asyncio

import pytest

from vero_tasks import (
    TaskContext,
    TaskOutput,
    TaskParameters,
    TaskResult,
    create_task,
)


class Parameters(TaskParameters):
    multiplier: float = 1.0


@pytest.mark.asyncio
async def test_task_runs_cases_with_typed_parameters_and_concurrency():
    task = create_task("score", register=False, task_parameters_type=Parameters)
    active = 0
    maximum_active = 0

    @task.inference()
    async def infer(case, context):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1
        return TaskOutput(output=case["value"] * 2)

    @task.evaluation()
    async def evaluate(case, output, context):
        parameters = context.parse_task_params(Parameters)
        return TaskResult.from_task_output(
            output,
            score=output.output * parameters.multiplier,
            metrics={"raw": output.output},
        )

    results = await task.run(
        [{"value": 1}, {"value": 2}, {"value": 3}],
        TaskContext(parameters={"multiplier": 0.5}, max_concurrency=2),
    )

    assert [result.score for result in results] == [1.0, 2.0, 3.0]
    assert maximum_active == 2


@pytest.mark.asyncio
async def test_task_captures_inference_and_evaluation_errors():
    task = create_task("errors", register=False)

    @task("run_inference")
    async def infer(case, context):
        if case == "inference":
            raise RuntimeError("inference failed")
        return TaskOutput(output=case)

    @task("run_evaluation")
    async def evaluate(case, output, context):
        if case == "evaluation":
            raise RuntimeError("evaluation failed")
        return TaskResult.from_task_output(output, score=1.0)

    results = await task.run(
        ["inference", "evaluation", "ok"],
        TaskContext(),
    )

    assert results[0].error == "inference failed"
    assert results[0].error_traceback is not None
    assert results[1].eval_error == "evaluation failed"
    assert results[1].evaluation_error_traceback is not None
    assert results[2].score == 1.0


@pytest.mark.asyncio
async def test_batch_evaluation_preserves_inference_errors():
    task = create_task("batch-errors", register=False)

    @task.inference()
    async def infer(case, context):
        if case == "broken":
            raise RuntimeError("inference failed")
        return TaskOutput(output=case)

    @task.evaluation(batch=True)
    async def evaluate(cases, outputs, context):
        return [TaskResult(score=1.0) for _ in cases]

    results = await task.run(["broken", "ok"], TaskContext())

    assert results[0].error == "inference failed"
    assert results[1].error is None
