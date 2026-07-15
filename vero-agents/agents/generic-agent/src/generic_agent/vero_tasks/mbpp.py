"""MBPP benchmark task definition."""

import asyncio
import io
import re
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Optional, TypedDict

from vero_tasks import TaskContext, TaskOutput, TaskResult, create_task

from .utils import (
    EXECUTION_TIMEOUT,
    GenericAgentParameters,
    extract_code,
    run_agent_with_tracing,
)

mbpp_task = create_task("mbpp", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])


class MbppTaskT(TypedDict):
    task_id: int
    text: str
    code: str
    test_list: list[str]
    test_setup_code: str
    challenge_test_list: list[str]


def extract_function_name(test_list: list[str]) -> str | None:
    """Extract the expected function name from test assertions."""
    for test in test_list:
        match = re.search(r"assert\s+(\w+)\s*\(", test)
        if match:
            return match.group(1)
    return None


def _execute_tests(test_list: list[str], global_dict: dict) -> str:
    """Execute all tests synchronously (runs in thread pool). Returns captured output."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        for test in test_list:
            exec(test, global_dict)
    return stdout_capture.getvalue() + stderr_capture.getvalue()


async def check_solution(
    solution: str, test_list: list[str], entry_point: str | None, test_setup_code: str = ""
) -> tuple[bool, str]:
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

        # Capture stdout/stderr during setup and solution execution
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            if test_setup_code:
                exec(test_setup_code, global_dict)
            exec(solution, global_dict)

        if entry_point and entry_point not in global_dict:
            raise ValueError(f"Function {entry_point} is not defined in the solution.")

        await asyncio.wait_for(
            asyncio.to_thread(_execute_tests, test_list, global_dict),
            timeout=EXECUTION_TIMEOUT,
        )

        return True, "The solution passed all test cases."

    except TimeoutError:
        return (
            False,
            "Execution timed out. Please check if your solution contains infinite loops or overly time-consuming operations.",
        )
    except Exception as e:
        error_message = f"Error: {e!s}.\n Solution: {solution}.\n Tests: {test_list}"
        return False, error_message


@mbpp_task("run_inference")
async def run_inference(
    task: MbppTaskT,
    evaluation_parameters: TaskContext,
) -> TaskOutput:
    """Run inference on a single task."""
    params = evaluation_parameters.parse_task_params(GenericAgentParameters)
    model = params.model
    task_inputs = {
        "question": task["text"],
        "entry_point": extract_function_name(task["test_list"]),
        "test_list": task["test_list"],
        "timeout": EXECUTION_TIMEOUT,
    }
    return await run_agent_with_tracing(task_inputs=task_inputs, task_name="mbpp", model=model)


@mbpp_task("run_evaluation")
async def evaluate_sample(
    task: MbppTaskT,
    output: TaskOutput,
    evaluation_parameters: TaskContext,
) -> TaskResult | Exception:
    """Evaluate the inference output for a single task."""
    if output.error is not None:
        return TaskResult.from_task_output(task_output=output, score=0.0, eval_error=str(output.error))

    expected_output = "\nCorrect Solution:\n" + task["code"]
    entry_point = extract_function_name(task["test_list"])

    try:
        extracted_code = await extract_code(response=output.output, entry_point=entry_point)
        if extracted_code is None:
            return TaskResult.from_task_output(
                task_output=output,
                score=0.0,
                eval_error="No code could be extracted from the response.",
            )
        test_setup_code = task.get("test_setup_code", "") or ""
        passed, test_case_details = await check_solution(
            extracted_code, task["test_list"], entry_point, test_setup_code
        )
        score = 1.0 if passed else 0.0
        feedback = test_case_details + expected_output
        return TaskResult.from_task_output(task_output=output, score=score, feedback=feedback)
    except Exception as e:
        return TaskResult.from_task_output(task_output=output, score=0.0, eval_error=str(e))
