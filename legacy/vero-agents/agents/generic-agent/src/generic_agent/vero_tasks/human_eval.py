"""HumanEval benchmark task definition."""

import asyncio
import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Optional, TypedDict

from vero.core.db.result import TaskOutput, TaskResult
from vero.core.evaluation import EvaluationParameters
from vero.core.task import create_task

from .utils import (
    EXECUTION_TIMEOUT,
    GenericAgentParameters,
    extract_code,
    run_agent_with_tracing,
)

human_eval_task = create_task("human_eval", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])


class HumanEvalTaskT(TypedDict):
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str


def _execute_check(check_func, entry_point_func) -> tuple[Any, str]:
    """Execute the check function synchronously (runs in thread pool). Returns result and captured output."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        result = check_func(entry_point_func)
    captured_output = stdout_capture.getvalue() + stderr_capture.getvalue()
    return result, captured_output


async def check_solution(solution: str, test: str, entry_point: str) -> tuple[bool, str]:
    """Check if the solution passes all test cases."""
    try:
        global_dict = {
            "math": __import__("math"),
            "hashlib": __import__("hashlib"),
            "re": __import__("re"),
            "List": list,
            "Dict": dict,
            "Tuple": tuple,
            "Optional": Optional,
            "Any": Any,
        }

        # Add handling for special cases (from AFlow)
        if entry_point == "decode_cyclic":
            solution = (
                '\n\ndef encode_cyclic(s: str):\n    """\n    returns encoded string by cycling groups of three characters.\n    """\n    # split string to groups. Each of length 3.\n    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n    # cycle elements in each group. Unless group has fewer elements than 3.\n    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n    return "".join(groups)'
                + "\n\n"
                + solution
            )
        elif entry_point == "decode_shift":
            solution = (
                '\n\ndef encode_shift(s: str):\n    """\n    returns encoded string by shifting every character by 5 in the alphabet.\n    """\n    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])\n\n\n'
                + solution
            )
        elif entry_point == "find_zero":
            solution = (
                "\n\ndef poly(xs: list, x: float):\n    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))\n\n"
                + solution
            )

        # Capture stdout/stderr during solution and test setup execution
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(solution, global_dict)

            if entry_point not in global_dict:
                raise ValueError(f"Function {entry_point} is not defined in the solution.")

            exec(test, global_dict)

        check = global_dict["check"]

        result, _ = await asyncio.wait_for(
            asyncio.to_thread(_execute_check, check, global_dict[entry_point]),
            timeout=EXECUTION_TIMEOUT,
        )

        if result is None:
            return True, "The solution passed all test cases."

    except TimeoutError:
        return (
            False,
            "Execution timed out. Please check if your solution contains infinite loops or overly time-consuming operations.",
        )
    except Exception as e:
        error_message = f"Error: {e!s}.\n Solution: {solution}.\n Test: {test}"
        return False, error_message

    return True, "The solution passed all test cases."


@human_eval_task("run_inference")
async def run_inference(
    task: HumanEvalTaskT,
    evaluation_parameters: EvaluationParameters,
) -> TaskOutput:
    """Run inference on a single task."""
    params = evaluation_parameters.parse_task_params(GenericAgentParameters)
    model = params.model
    task_inputs = {
        "question": task["prompt"],
        "timeout": EXECUTION_TIMEOUT,
        "entry_point": task["entry_point"],
    }
    return await run_agent_with_tracing(task_inputs=task_inputs, task_name="humaneval", model=model)


@human_eval_task("run_evaluation")
async def evaluate_sample(
    task: HumanEvalTaskT,
    output: TaskOutput,
    evaluation_parameters: EvaluationParameters,
) -> TaskResult | Exception:
    """Evaluate the inference output for a single task."""
    if output.error is not None:
        return TaskResult.from_task_output(task_output=output, score=0.0, eval_error=str(output.error))

    expected_output = (
        "\nCorrect Solution:\ndef "
        + task["entry_point"]
        + "(params you should put here):"
        + "\n\n"
        + task["canonical_solution"]
    )

    try:
        extracted_code = await extract_code(response=output.output, entry_point=task["entry_point"])
        if extracted_code is None:
            return TaskResult.from_task_output(
                task_output=output,
                score=0.0,
                eval_error="No code could be extracted from the response.",
            )
        passed, test_case_details = await check_solution(extracted_code, task["test"], task["entry_point"])
        score = 1.0 if passed else 0.0
        feedback = test_case_details + expected_output
        return TaskResult.from_task_output(task_output=output, score=score, feedback=feedback)
    except Exception as e:
        return TaskResult.from_task_output(task_output=output, score=0.0, eval_error=str(e))
