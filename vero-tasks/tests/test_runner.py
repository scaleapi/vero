from __future__ import annotations

import json
from pathlib import Path

import pytest

from vero_tasks import TaskAttemptError, TaskResult
from vero_tasks.runner import _errors, _load_cases, run_task
from vero_tasks.task import TaskDefinition


def test_load_cases_detects_staged_jsonl_without_a_suffix(tmp_path: Path):
    cases_path = tmp_path / "cases"
    cases_path.write_text('{"id": "a"}\n{"id": "b"}\n', encoding="utf-8")

    assert _load_cases(cases_path) == [{"id": "a"}, {"id": "b"}]


def test_errors_preserve_transient_history_and_explicit_terminal_errors():
    errors = _errors(
        TaskResult(
            eval_error="invalid score",
            attempt_errors=[
                TaskAttemptError(
                    message="rate limit",
                    phase="inference",
                    attempt=1,
                    retryable=True,
                    terminal=False,
                )
            ],
        )
    )

    assert [(error["phase"], error["terminal"]) for error in errors] == [
        ("inference", False),
        ("evaluation", True),
    ]


@pytest.mark.asyncio
async def test_runner_writes_canonical_report(tmp_path: Path, monkeypatch):
    module = tmp_path / "example_tasks.py"
    module.write_text(
        """
from vero_tasks import TaskOutput, TaskResult, create_task

task = create_task("double")

@task.inference()
async def infer(case, context):
    return TaskOutput(output=case["value"] * 2)

@task.evaluation()
async def evaluate(case, output, context):
    return TaskResult.from_task_output(
        output,
        score=float(output.output == case["expected"]),
        metrics={"distance": abs(output.output - case["expected"])},
    )
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {"id": "a", "value": 2, "expected": 4},
                {"id": "b", "value": 3, "expected": 7},
                {"id": "c", "value": 4, "expected": 8},
            ]
        ),
        encoding="utf-8",
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "request": {
                    "evaluation_set": {
                        "name": "test",
                        "partition": None,
                        "selection": {"kind": "ids", "ids": ["c", "a"]},
                    },
                    "parameters": {},
                    "limits": {
                        "max_concurrency": 2,
                        "case_timeout_seconds": 1.0,
                    },
                    "seed": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    await run_task(
        module="example_tasks",
        task_name="double",
        cases_path=cases_path,
        request_path=request_path,
        report_path=report_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["status"] == "success"
    assert report["metrics"] == {
        "distance": 0.0,
        "score": 1.0,
        "error_rate": 0.0,
    }
    assert [case["case_id"] for case in report["cases"]] == ["c", "a"]
    TaskDefinition.clear_registry()


@pytest.mark.asyncio
async def test_runner_applies_retry_policy_and_reports_attempts(
    tmp_path: Path, monkeypatch
):
    module = tmp_path / "retry_tasks.py"
    module.write_text(
        """
from vero_tasks import TaskOutput, TaskResult, create_task

task = create_task("retry")
attempts = 0

@task.inference()
async def infer(case, context):
    global attempts
    attempts += 1
    if attempts == 1:
        raise RuntimeError("rate limit")
    return TaskOutput(output=case["value"])

@task.evaluation()
async def evaluate(case, output, context):
    return TaskResult.from_task_output(output, score=1.0)
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([{"id": "a", "value": 2}]), encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "request": {
                    "evaluation_set": {
                        "name": "test",
                        "partition": None,
                        "selection": {"kind": "all"},
                    },
                    "parameters": {},
                    "limits": {
                        "max_concurrency": 1,
                        "case_timeout_seconds": 1.0,
                        "retry": {
                            "max_attempts": 2,
                            "initial_delay_seconds": 0,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    await run_task(
        module="retry_tasks",
        task_name="retry",
        cases_path=cases_path,
        request_path=request_path,
        report_path=report_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert report["metrics"]["error_rate"] == 0.0
    assert report["cases"][0]["status"] == "success"
    assert report["cases"][0]["errors"][0] == {
        "message": "rate limit",
        "code": "task_inference_error",
        "phase": "inference",
        "attempt": 1,
        "retryable": True,
        "terminal": False,
        "metadata": {
            "traceback": report["cases"][0]["errors"][0]["metadata"]["traceback"]
        },
    }
    TaskDefinition.clear_registry()
