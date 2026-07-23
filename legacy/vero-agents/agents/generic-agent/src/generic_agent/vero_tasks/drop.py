"""DROP benchmark task definition."""

from typing import TypedDict

from vero.core.db.result import TaskOutput, TaskResult
from vero.core.evaluation import EvaluationParameters
from vero.core.task import create_task

from .utils import (
    SHORT_FORM_QA_JUDGE_TEMPLATE,
    GenericAgentParameters,
    ShortFormQaJudgeLMOutput,
    get_completion,
    run_agent_with_tracing,
)

drop_task = create_task("drop", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])
drop_single_answer_task = create_task("drop_single_answer", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])


class AnswersSpans(TypedDict):
    spans: list[str]


class DropTaskItem(TypedDict):
    passage: str
    question: str
    answers_spans: AnswersSpans
    answer: str | None = None


# --------------------------------------------------
# DROP Task
# --------------------------------------------------


@drop_task("run_inference")
async def run_inference(
    task: DropTaskItem,
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
    task_inputs = {
        "passage": task["passage"],
        "question": task["question"],
    }
    return await run_agent_with_tracing(task_inputs=task_inputs, task_name="drop", model=model)


@drop_task("run_evaluation")
async def evaluate_sample(
    task: DropTaskItem,
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
    question = task["question"]
    target = " | ".join(task["answers_spans"]["spans"])
    predicted_answer = output.output

    prompt = SHORT_FORM_QA_JUDGE_TEMPLATE.format(question=question, target=target, predicted_answer=predicted_answer)

    try:
        judge_output = await get_completion(prompt=prompt, text_format=ShortFormQaJudgeLMOutput)
    except Exception as e:
        return TaskResult.from_task_output(task_output=output, score=0.0, eval_error=str(e))

    feedback = f"Expected: {target}, Extracted: {judge_output.extracted_answer}, Grade: {judge_output.grade}"
    return TaskResult.from_task_output(task_output=output, score=judge_output.score, feedback=feedback)


# --------------------------------------------------
# DROP Single Answer Task
# --------------------------------------------------


@drop_single_answer_task("run_inference")
async def run_inference_single_answer(
    task: DropTaskItem,
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
    task_inputs = {
        "passage": task["passage"],
        "question": task["question"],
    }
    return await run_agent_with_tracing(task_inputs=task_inputs, task_name="drop", model=model)


@drop_single_answer_task("run_evaluation")
async def evaluate_sample_single_answer(
    task: DropTaskItem,
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
    question = task["question"]
    target = task["answer"]
    predicted_answer = output.output

    prompt = SHORT_FORM_QA_JUDGE_TEMPLATE.format(question=question, target=target, predicted_answer=predicted_answer)

    try:
        judge_output = await get_completion(prompt=prompt, text_format=ShortFormQaJudgeLMOutput)
    except Exception as e:
        return TaskResult.from_task_output(task_output=output, score=0.0, eval_error=str(e))

    feedback = f"Expected: {target}, Extracted: {judge_output.extracted_answer}, Grade: {judge_output.grade}"
    return TaskResult.from_task_output(task_output=output, score=judge_output.score, feedback=feedback)
