import json
import os
from pathlib import Path

from vero_tasks import TaskContext, TaskOutput, TaskResult, TaskT, create_task

from agents.exceptions import MaxTurnsExceeded
from agents.extensions.models.litellm_model import LitellmModel
from agents.items import RunItemBase
from generic_agent.agent import run_agent

# Import the judge from gaia.eval
from generic_agent.vero_tasks.gaia.eval import judge_answer_async

from ..utils import (
    SHORT_FORM_QA_JUDGE_TEMPLATE,
    GenericAgentParameters,
    ShortFormQaJudgeLMOutput,
    get_completion,
    run_agent_with_tracing,
)

# Create and register the task
gaia_task = create_task("gaia", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])

PATH_TO_GAIA_ARTIFACTS = Path(__file__).parent.parent.parent.parent.parent / "data" / "gaia"


@gaia_task("run_inference")
async def run_inference(
    task: dict[str, str],
    evaluation_parameters: TaskContext,
) -> TaskOutput:
    """Run inference on a single task."""
    question = task["Question"]

    file_path = task.get("file_path")
    if file_path:
        file_path = (PATH_TO_GAIA_ARTIFACTS / file_path).as_posix()

    params = evaluation_parameters.parse_task_params(GenericAgentParameters)
    model = params.model
    task_inputs = {
        "question": question,
        "file_path": file_path,
    }

    return await run_agent_with_tracing(task_inputs=task_inputs, task_name="gaia", model=model)


@gaia_task("run_evaluation")
async def evaluate_sample(
    task: TaskT,
    output: TaskOutput,
    evaluation_parameters: TaskContext,
) -> TaskResult | Exception:
    """Evaluate the inference output for a single task."""
    question = task["Question"]
    ground_truth = task["Final answer"]
    predicted_answer = output.output

    try:
        # Use judge_answer_async from gaia.eval
        eval_result = await judge_answer_async(
            question=question,
            expected_answer=ground_truth,
            agent_answer=predicted_answer or "",
            model="openai/gpt-4.1-mini-2025-04-14",
            max_eval_chars=4096,
        )
        score = 1.0 if eval_result.is_correct else 0.0
        feedback = eval_result.reasoning
        return TaskResult.from_task_output(task_output=output, score=score, feedback=feedback)
    except Exception as e:
        return e
