"""Pharma Summarizer benchmark task definition."""

from vero.core.db.result import TaskOutput, TaskResult
from vero.core.evaluation import EvaluationParameters
from vero.core.task import TaskT, create_task

from pharma_summarizer.agent import run_agent

from .utils import evaluate_summary

pharma_summarizer_task = create_task("pharma_summarizer", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])


@pharma_summarizer_task("run_inference")
async def run_inference(
    task: TaskT,
    evaluation_parameters: EvaluationParameters,
) -> TaskOutput:
    """Run inference on a single task.

    Args:
        task: The task data (raw dict from the Dataset)
        evaluation_parameters: Evaluation parameters

    Returns:
        The inference output (e.g., agent response, model prediction)
    """
    section = task["content"]
    try:
        result = await run_agent(section)
    except Exception as e:
        return TaskOutput(error=e)
    return TaskOutput(output=result.final_output, execution_trace=result.to_input_list())


@pharma_summarizer_task("run_evaluation")
async def evaluate_sample(
    task: TaskT,
    output: TaskOutput,
    evaluation_parameters: EvaluationParameters,
) -> TaskResult | Exception:
    """Evaluate the inference output for a single task.

    Args:
        task: The task data (raw dict from the dataset)
        output: Output from run_inference
        evaluation_parameters: Evaluation parameters

    Returns:
        TaskResult with score and optional feedback
    """
    section = task["content"]
    summary = output.output
    score, feedback = await evaluate_summary(summary, section)
    return TaskResult.from_task_output(task_output=output, score=score, feedback=feedback)
