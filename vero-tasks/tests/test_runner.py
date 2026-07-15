from __future__ import annotations

import json
from pathlib import Path

import pytest

from vero_tasks.runner import run_task
from vero_tasks.task import TaskDefinition


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
