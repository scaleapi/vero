"""Matrix multiply evaluation task.

Measures correctness and execution speed of a matmul kernel.
Score = average time per multiply (lower is better).
Incorrect results get a penalty score of 999999.0.
"""

import time

from vero_tasks import (
    TaskContext,
    TaskOutput,
    TaskParameters,
    TaskResult,
    create_task,
)


class MatmulParameters(TaskParameters):
    n_repeats: int = 100


matmul_task = create_task("matmul", task_parameters_type=MatmulParameters)


@matmul_task.inference()
async def run_inference(task: dict, context: TaskContext) -> TaskOutput:
    from matmul_kernel import multiply

    a = task["matrix_a"]
    b = task["matrix_b"]

    parameters = context.parse_task_params(MatmulParameters)

    # Warmup
    multiply(a, b)

    # Timed runs (like timeit)
    start = time.perf_counter()
    for _ in range(parameters.n_repeats):
        result = multiply(a, b)
    elapsed_ms = (time.perf_counter() - start) / parameters.n_repeats * 1000

    return TaskOutput(output={"result": result, "time_ms": elapsed_ms})


@matmul_task.evaluation()
async def run_evaluation(
    task: dict, output: TaskOutput, context: TaskContext
) -> TaskResult:
    expected = task["expected"]
    actual = output.output["result"]
    time_ms = output.output["time_ms"]

    # Check correctness (within floating point tolerance)
    correct = all(
        abs(a_val - e_val) < 1e-6
        for a_row, e_row in zip(actual, expected)
        for a_val, e_val in zip(a_row, e_row)
    )

    # Score = time if correct, penalty if wrong (lower is better)
    score = time_ms if correct else 999999.0

    return TaskResult.from_task_output(
        output,
        score=score,
        metrics={"time_ms": time_ms, "correct": 1.0 if correct else 0.0},
        feedback=f"{'Correct' if correct else 'Wrong'}. Time: {time_ms:.2f}ms",
    )
