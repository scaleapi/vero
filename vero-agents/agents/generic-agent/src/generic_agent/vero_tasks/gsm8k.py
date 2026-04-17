"""GSM8K benchmark task definition."""

from vero.core.db.result import TaskOutput, TaskResult
from vero.core.evaluation import EvaluationParameters
from vero.core.task import create_task

from .utils import (
    EXTRACT_AND_JUDGE_TEMPLATE,
    GenericAgentParameters,
    MathJudgeLMOutput,
    get_completion,
    run_agent_with_tracing,
)

gsm8k_task = create_task("gsm8k", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])


@gsm8k_task("run_inference")
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
    task_inputs = {"question": task["question"]}
    return await run_agent_with_tracing(task_inputs=task_inputs, task_name="gsm8k", model=model)


@gsm8k_task("run_evaluation")
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
    ground_truth_solution = task["answer"]
    predicted_response = output.output

    prompt = EXTRACT_AND_JUDGE_TEMPLATE.format(
        ground_truth_solution=ground_truth_solution,
        predicted_response=predicted_response,
    )

    try:
        judge_output = await get_completion(prompt=prompt, text_format=MathJudgeLMOutput)
    except Exception as e:
        return TaskResult.from_task_output(task_output=output, score=0.0, eval_error=str(e))

    feedback = f"Expected: {judge_output.ground_truth_expression}, Extracted Prediction: {judge_output.predicted_expression}, Is Equivalent: {judge_output.is_equivalent}"
    return TaskResult.from_task_output(task_output=output, score=judge_output.score, feedback=feedback)
