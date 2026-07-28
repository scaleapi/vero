"""
Vero task definition for gsm8k-agent.

This file contains the main evaluation logic for testing
the performance of your package using the VeroTask decorator formalism.
"""

import re

from vero.core.db.result import TaskOutput, TaskResult
from vero.core.evaluation import EvaluationParameters
from vero.core.task import create_task

from gsm8k_agent.agent import gsm8k_agent

# Create the evaluation task instance.
gsm8k_task = create_task("gsm8k")


def parse_ground_truth_answer(ground_truth_answer: str) -> int:
    """Parse the answer string from the ground truth reasoning string."""
    return int(ground_truth_answer.split("####")[1].replace(",", "").strip())


def parse_model_predicted_answer(predicted_answer: str) -> float | None:
    """
    Extracts the numeric value after "Final answer:" from the given string.
    Returns a float if successful, or None if no number is found.
    """
    PATTERN = re.compile(
        r"Final answer:\s*"  # literal prefix
        r"(?:[^\d\s+-]\s*)?"  # optional currency symbol (not + or -)
        r"([+-]?[\d,]+(?:\.\d+)?)"  # now capture optional sign + digits/commas + opt. decimal
    )
    m = PATTERN.search(predicted_answer)
    if not m:
        return None

    num_str = m.group(1).replace(",", "")
    try:
        return int(num_str)
    except ValueError:
        pass

    try:
        return float(num_str)
    except ValueError:
        pass

    return None


def is_equal(predicted_answer: str, ground_truth_answer: str) -> bool:
    ans = parse_ground_truth_answer(ground_truth_answer)
    pred = parse_model_predicted_answer(predicted_answer)
    return pred == ans


@gsm8k_task.inference()
async def run_inference(
    task_data: dict[str, str],
    evaluation_parameters: EvaluationParameters,
) -> TaskOutput:
    """Run inference on a single task.

    Args:
        task_data: The task data (raw dict from the Dataset)
        evaluation_parameters: Evaluation parameters

    Returns:
        The inference output (e.g., agent response, model prediction)
    """
    try:
        messages = await gsm8k_agent(task_data["question"])
    except Exception as e:
        return TaskOutput(error=str(e))

    if not messages:
        return TaskOutput(error="No response from agent")

    if isinstance(messages[-1], dict):
        output = messages[-1]["content"]
    else:
        output = messages[-1].content

    return TaskOutput(output=output, execution_trace=messages)


@gsm8k_task.evaluation()
async def evaluate_sample(
    task_data: dict[str, str],
    output: TaskOutput,
    evaluation_parameters: EvaluationParameters,
) -> TaskResult:
    """Evaluate the inference output for a single task.

    Args:
        task_data: The task data (raw dict from the dataset)
        output: Output from run_inference
        evaluation_parameters: Evaluation parameters

    Returns:
        TaskResult with score and optional feedback
    """
    score = int(is_equal(output.output, task_data["answer"]))
    return TaskResult.from_task_output(
        task_output=output,
        score=score,
        feedback=f"Expected: {task_data['answer']}, Got: {output.output}",
    )
