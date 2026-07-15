"""HLE benchmark task definition."""

from vero_tasks import TaskContext, TaskOutput, TaskResult, TaskT, create_task

from .utils import TaskType, grade_answer, run_inference

hle_task = create_task("hle", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])


@hle_task("run_inference")
async def hle_run_inference(
    task: TaskT,
    evaluation_parameters: TaskContext,
) -> TaskOutput:
    """Run inference on a single HLE task."""
    return await run_inference(task, evaluation_parameters)


@hle_task("run_evaluation")
async def evaluate_sample(
    task: TaskT,
    output: TaskOutput,
    evaluation_parameters: TaskContext,
) -> TaskResult | Exception:
    """Evaluate the inference output for a single HLE task."""
    score, feedback = await grade_answer(
        question=task["question"],
        gold_answer=task["answer"],
        predicted_answer=output.output,
        task_type=TaskType.HLE,
    )
    return TaskResult.from_task_output(task_output=output, score=score, feedback=feedback)
