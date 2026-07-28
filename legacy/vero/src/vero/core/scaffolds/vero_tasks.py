"""VeroTask evaluation scaffold.

This is a working starter task that does exact-match evaluation.
Adapt run_inference and run_evaluation to your use case.

To verify setup works:
  1. Create a dataset with "input" and "expected" fields
  2. Run: vero check --project-path . --task main
  3. Run: vero evaluate --project-path . --task main --dataset ./data --split test
"""

from vero.core.db.result import TaskOutput, TaskResult
from vero.core.evaluation import EvaluationParameters
from vero.core.task import create_task

# Create and register the task.
# The task name here should match what you pass to Policy(task="...").
# Add required_env_vars if your inference needs API keys, e.g.:
#   create_task("main", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])
task = create_task("main")


@task.inference()
async def run_inference(
    task: dict, evaluation_parameters: EvaluationParameters
) -> TaskOutput:
    """Run inference on a single task.

    Replace this with your agent logic. This default echoes the input.

    Args:
        task: Raw dict from the dataset row.
        evaluation_parameters: Eval config (timeout, task_params, etc.)

    Returns:
        TaskOutput wrapping your agent's output.
    """
    # TODO: Replace with your agent logic, e.g.:
    #   from my_agent import run
    #   result = await run(task["input"])
    #   return TaskOutput(output=result)
    return TaskOutput(output=task.get("input", ""))


@task.evaluation()
async def run_evaluation(
    task: dict,
    output: TaskOutput,
    evaluation_parameters: EvaluationParameters,
) -> TaskResult:
    """Evaluate the inference output against ground truth.

    Replace this with your scoring logic. This default does exact match.

    Args:
        task: Raw dict from the dataset row.
        output: Output from run_inference.
        evaluation_parameters: Eval config.

    Returns:
        TaskResult with score (0-1) and optional feedback.
    """
    # TODO: Replace with your evaluation logic, e.g.:
    #   score = my_custom_scorer(output.output, task["expected"])
    expected = task.get("expected", "")
    prediction = output.output
    score = 1.0 if prediction == expected else 0.0
    return TaskResult(
        output=prediction,
        score=score,
        feedback=f"expected={expected}, got={prediction}",
    )
