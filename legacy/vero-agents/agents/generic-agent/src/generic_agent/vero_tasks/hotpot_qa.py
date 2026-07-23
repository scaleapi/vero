"""HotpotQA benchmark task definition."""

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

hotpot_qa_task = create_task("hotpot_qa", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])


class SupportingFacts(TypedDict):
    title: list[str]
    sent_id: list[int]


class Context(TypedDict):
    title: list[str]
    sentences: list[list[str]]


class AflowHotpotQAItem(TypedDict):
    id: str
    question: str
    answer: str
    type: str
    level: str
    supporting_facts: SupportingFacts
    context: Context


@hotpot_qa_task("run_inference")
async def run_inference(
    task: AflowHotpotQAItem,
    evaluation_parameters: EvaluationParameters,
) -> TaskOutput:
    """Run inference on a single task.

    Args:
        task: The task data (raw dict from the Dataset)
        evaluation_parameters: Evaluation parameters

    Returns:
        The inference output (e.g., agent response, model prediction)
    """
    question = task["question"]
    context = task["context"]
    formatted_context = "\n".join(" ".join(paragraph) for paragraph in context["sentences"])
    params = evaluation_parameters.parse_task_params(GenericAgentParameters)
    model = params.model
    task_inputs = {
        "formatted_context": formatted_context,
        "question": question,
        "context": context,
    }
    return await run_agent_with_tracing(task_inputs=task_inputs, task_name="hotpotqa", model=model)


@hotpot_qa_task("run_evaluation")
async def evaluate_sample(
    task: AflowHotpotQAItem,
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
    target = task["answer"]
    question = task["question"]
    predicted_answer = output.output

    prompt = SHORT_FORM_QA_JUDGE_TEMPLATE.format(question=question, target=target, predicted_answer=predicted_answer)

    try:
        judge_output = await get_completion(prompt=prompt, text_format=ShortFormQaJudgeLMOutput)
    except Exception as e:
        return TaskResult.from_task_output(task_output=output, score=0.0, eval_error=str(e))

    feedback = f"Expected: {target}, Extracted: {judge_output.extracted_answer}, Grade: {judge_output.grade}"
    return TaskResult.from_task_output(task_output=output, score=judge_output.score, feedback=feedback)
