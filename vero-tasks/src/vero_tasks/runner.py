"""Run a registered Python task through VeRO's schema-v1 command contract."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vero_tasks.models import TaskContext, TaskResult
from vero_tasks.task import TaskDefinition


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _load_cases(path: Path) -> list[Any]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and "cases" in value:
        value = value["cases"]
    if not isinstance(value, list):
        raise ValueError(
            "case file must contain a JSON list or an object with a cases list"
        )
    return value


def _case_id(case: Any, index: int) -> str:
    if isinstance(case, dict) and case.get("id") is not None:
        return str(case["id"])
    return str(index)


def _select(cases: list[Any], selection: dict[str, Any]) -> list[tuple[str, Any]]:
    indexed = [(_case_id(case, index), case) for index, case in enumerate(cases)]
    case_ids = [case_id for case_id, _ in indexed]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")
    kind = selection.get("kind", "all")
    if kind == "all":
        return indexed
    if kind == "range":
        return indexed[selection.get("start", 0) : selection["stop"]]
    if kind == "ids":
        by_id = dict(indexed)
        missing = [case_id for case_id in selection["ids"] if case_id not in by_id]
        if missing:
            raise ValueError(f"unknown case IDs: {missing}")
        return [(case_id, by_id[case_id]) for case_id in selection["ids"]]
    raise ValueError(f"unknown case selection kind: {kind!r}")


def _errors(result: TaskResult) -> list[dict[str, Any]]:
    errors = []
    if result.error is not None:
        metadata = (
            {"traceback": result.error_traceback}
            if result.error_traceback is not None
            else {}
        )
        errors.append(
            {
                "message": result.error,
                "code": "task_inference_error",
                "phase": "inference",
                "terminal": True,
                "metadata": metadata,
            }
        )
    if result.eval_error is not None:
        metadata = (
            {"traceback": result.evaluation_error_traceback}
            if result.evaluation_error_traceback is not None
            else {}
        )
        errors.append(
            {
                "message": result.eval_error,
                "code": "task_evaluation_error",
                "phase": "evaluation",
                "terminal": True,
                "metadata": metadata,
            }
        )
    return errors


def _report(
    selected: list[tuple[str, Any]],
    results: list[TaskResult],
) -> dict[str, Any]:
    cases = []
    totals: dict[str, list[float]] = defaultdict(list)
    error_count = 0
    for (case_id, case), result in zip(selected, results):
        metrics = dict(result.metrics)
        if result.score is not None:
            metrics.setdefault("score", result.score)
        errors = _errors(result)
        if errors:
            error_count += 1
        else:
            for name, value in metrics.items():
                totals[name].append(value)
        cases.append(
            {
                "case_id": case_id,
                "status": "error" if errors else "success",
                "metrics": metrics,
                "input": _json_value(case),
                "output": _json_value(result.output),
                "feedback": result.feedback,
                "errors": errors,
                "execution_trace": (
                    _json_value(list(result.execution_trace))
                    if result.execution_trace is not None
                    else None
                ),
                "evaluation_trace": (
                    _json_value(list(result.evaluation_trace))
                    if result.evaluation_trace is not None
                    else None
                ),
            }
        )
    metrics = {
        name: sum(values) / len(values)
        for name, values in totals.items()
        if values
    }
    metrics["error_rate"] = error_count / len(results) if results else 0.0
    return {
        "schema_version": 1,
        "status": "failed" if results and error_count == len(results) else "success",
        "metrics": metrics,
        "cases": cases,
    }


async def run_task(
    *,
    module: str,
    task_name: str,
    cases_path: Path,
    request_path: Path,
    report_path: Path,
) -> None:
    request_envelope = json.loads(request_path.read_text(encoding="utf-8"))
    if request_envelope.get("schema_version") != 1:
        raise ValueError("unsupported command evaluation request schema")
    request = request_envelope["request"]
    importlib.import_module(module)
    task = TaskDefinition.resolve(task_name)
    selected = _select(
        _load_cases(cases_path),
        request["evaluation_set"]["selection"],
    )
    limits = request["limits"]
    context = TaskContext(
        parameters=request.get("parameters", {}),
        max_concurrency=limits["max_concurrency"],
        case_timeout_seconds=limits["case_timeout_seconds"],
        seed=request.get("seed"),
    )
    results = await task.run([case for _, case in selected], context)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_report(selected, results), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(
        run_task(
            module=arguments.module,
            task_name=arguments.task,
            cases_path=arguments.cases,
            request_path=arguments.request,
            report_path=arguments.report,
        )
    )


if __name__ == "__main__":
    main()
