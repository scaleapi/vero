"""SimpleQA benchmark task definition."""

from vero_tasks import TaskContext, TaskOutput, TaskResult, TaskT, create_task

from .utils import TaskType, grade_answer, run_inference

simple_qa_task = create_task("simple_qa", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])


@simple_qa_task("run_inference")
async def simple_qa_run_inference(
    task: dict[str, str],
    evaluation_parameters: TaskContext,
) -> TaskOutput:
    """Run inference on a single SimpleQA task."""
    return await run_inference(task, evaluation_parameters)


@simple_qa_task("run_evaluation")
async def evaluate_sample(
    task: TaskT,
    output: TaskOutput,
    evaluation_parameters: TaskContext,
) -> TaskResult | Exception:
    """Evaluate the inference output for a single SimpleQA task."""
    score, feedback = await grade_answer(
        question=task["problem"],
        gold_answer=task["answer"],
        predicted_answer=output.output,
        task_type=TaskType.SIMPLE_QA,
    )
    return TaskResult.from_task_output(task_output=output, score=score, feedback=feedback)
