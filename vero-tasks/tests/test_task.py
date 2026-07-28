from __future__ import annotations

import asyncio

import pytest

from vero_tasks import (
    RetryPolicy,
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


@pytest.mark.asyncio
async def test_task_retries_transient_inference_errors_and_preserves_history():
    task = create_task("retry-inference", register=False)
    attempts = 0

    @task.inference()
    async def infer(case, context):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("rate limit from provider")
        return TaskOutput(output=case)

    @task.evaluation()
    async def evaluate(case, output, context):
        return TaskResult.from_task_output(output, score=1.0)

    results = await task.run(
        ["ok"],
        TaskContext(
            retry=RetryPolicy(
                max_attempts=3,
                initial_delay_seconds=0,
            )
        ),
    )

    assert attempts == 3
    assert results[0].score == 1.0
    assert results[0].is_error() is False
    assert [error.attempt for error in results[0].attempt_errors] == [1, 2]
    assert all(not error.terminal for error in results[0].attempt_errors)


@pytest.mark.asyncio
async def test_task_retries_timeouts_and_evaluation_errors():
    task = create_task("retry-timeout-and-evaluation", register=False)
    inference_attempts = 0
    evaluation_attempts = 0

    @task.inference()
    async def infer(case, context):
        nonlocal inference_attempts
        inference_attempts += 1
        if inference_attempts == 1:
            await asyncio.sleep(0.02)
        return TaskOutput(output=case)

    @task.evaluation()
    async def evaluate(case, output, context):
        nonlocal evaluation_attempts
        evaluation_attempts += 1
        if evaluation_attempts == 1:
            raise RuntimeError("too many requests")
        return TaskResult.from_task_output(output, score=1.0)

    results = await task.run(
        ["ok"],
        TaskContext(
            case_timeout_seconds=0.01,
            retry=RetryPolicy(max_attempts=2, initial_delay_seconds=0),
        ),
    )

    assert inference_attempts == 2
    assert evaluation_attempts == 2
    assert results[0].score == 1.0
    assert [
        (error.phase, error.attempt, error.terminal)
        for error in results[0].attempt_errors
    ] == [
        ("inference", 1, False),
        ("evaluation", 1, False),
    ]


@pytest.mark.asyncio
async def test_task_does_not_retry_non_transient_errors():
    task = create_task("no-retry", register=False)
    attempts = 0

    @task.inference()
    async def infer(case, context):
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid candidate")

    @task.evaluation()
    async def evaluate(case, output, context):
        return TaskResult.from_task_output(output)

    results = await task.run(
        ["broken"],
        TaskContext(retry=RetryPolicy(initial_delay_seconds=0)),
    )

    assert attempts == 1
    assert results[0].error == "invalid candidate"
    assert len(results[0].attempt_errors) == 1
    assert results[0].attempt_errors[0].retryable is False
    assert results[0].attempt_errors[0].terminal is True
