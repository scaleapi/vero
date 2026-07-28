"""GPQA benchmark task definition."""

from vero.core.db.result import TaskOutput, TaskResult
from vero.core.evaluation import EvaluationParameters
from vero.core.task import create_task

from .utils import (
    GenericAgentParameters,
    run_agent_with_tracing,
)

gpqa_task = create_task("gpqa", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])


def extract_answer_from_model_response(model_response: str) -> int | None:
    """Extract answer index from model response.

    Expects format like "ANSWER: A" (case/whitespace insensitive).
    Returns 0 for A, 1 for B, 2 for C, 3 for D, or None if not found.
    """
    import re

    match = re.search(r"answer:\s*([a-d])", model_response, re.IGNORECASE)
    if not match:
        return None
    return ord(match.group(1).upper()) - ord("A")


@gpqa_task("run_inference")
async def run_inference(
    task: dict,
    evaluation_parameters: EvaluationParameters,
) -> TaskOutput:
    """Run inference on a single task.

    Args:
        task: The task data (raw dict from the Dataset)
        evaluation_parameters: Evaluation parameters

    Returns:
        The inference output (e.g., agent response, model prediction)
    """
    params = evaluation_parameters.parse_task_params(GenericAgentParameters)
    model = params.model
    task_inputs = {"question": task["question"], "options": task["options"]}
    return await run_agent_with_tracing(task_inputs=task_inputs, task_name="gpqa", model=model)


@gpqa_task("run_evaluation")
async def evaluate_sample(
    task: dict,
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
    answer_index = task["answer_index"]
    explanation = task["explanation"]
    predicted_response = output.output

    try:
        predicted_answer_index = extract_answer_from_model_response(predicted_response)
    except Exception as e:
        return TaskResult.from_task_output(task_output=output, score=0.0, eval_error=str(e))

    score = 1.0 if predicted_answer_index == answer_index else 0.0
    feedback = f"Expected: {answer_index}, Extracted Prediction: {predicted_answer_index}, Correct Answer Explanation: {explanation}"
    return TaskResult.from_task_output(task_output=output, score=score, feedback=feedback)
