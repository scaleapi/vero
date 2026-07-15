"""Facts Search benchmark task definition."""

from vero_tasks import TaskContext, TaskOutput, TaskResult, TaskT, create_task

from .utils import TaskType, grade_answer, run_inference

facts_search_task = create_task("facts_search", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])


@facts_search_task("run_inference")
async def facts_search_run_inference(
    task: TaskT,
    evaluation_parameters: TaskContext,
) -> TaskOutput:
    """Run inference on a single Facts Search task."""
    return await run_inference(task, evaluation_parameters)


@facts_search_task("run_evaluation")
async def evaluate_sample(
    task: TaskT,
    output: TaskOutput,
    evaluation_parameters: TaskContext,
) -> TaskResult | Exception:
    """Evaluate the inference output for a single Facts Search task."""
    score, feedback = await grade_answer(
        question=task["problem"],
        gold_answer=task["gold answer"],
        predicted_answer=output.output,
        task_type=TaskType.FACTS_SEARCH,
    )
    return TaskResult.from_task_output(task_output=output, score=score, feedback=feedback)
