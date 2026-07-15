import json
import subprocess
from pathlib import Path

import pytest
from vero.evaluation import CaseRange

from vero_benchmarking import runner
from vero_benchmarking.tasks.base import OptimizationTask


def _git_target(path: Path) -> Path:
    path.mkdir()
    (path / "pyproject.toml").write_text(
        """[project]
name = "example-target"
version = "0.1.0"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["example_target"]
"""
    )
    package = path / "example_target"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "task.py").write_text(
        """from vero_tasks import TaskResult, create_task

example = create_task("example")

@example("run_inference")
def infer(case, context):
    return case["x"] * 2

@example("run_evaluation")
def evaluate(case, output, context):
    expected = case["x"] * 2
    return TaskResult(score=float(output.output == expected), output=output.output)
"""
    )
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "add", "--all"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=vero",
            "-c",
            "user.email=vero@localhost",
            "commit",
            "-m",
            "baseline",
        ],
        cwd=path,
        check=True,
    )
    return path


@pytest.mark.asyncio
async def test_build_baseline_session_uses_canonical_runtime(tmp_path, monkeypatch):
    target = _git_target(tmp_path / "target")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([{"x": 1}, {"x": 2}, {"x": 3}]))
    task = OptimizationTask(
        project_path=target,
        dataset_path=dataset,
        task="example",
        module="example_target.task",
        evaluation_budget=4,
        max_cases_per_evaluation=2,
        parameters={"temperature": 0},
    )
    monkeypatch.setattr(runner, "load_task", lambda _: task)

    session = await runner.build_benchmark_session(
        task_name="example",
        model="test-model",
        agent_name=None,
        session_dir=tmp_path / "session",
        evaluation_budget=2,
    )

    assert session.optimizer.max_candidates == 0
    assert session.optimizer.producers == {}
    assert session.optimizer.parameters == {
        "model": "test-model",
        "temperature": 0,
    }
    assert session.optimizer.evaluation_set.name == "example"
    assert session.optimizer.evaluation_set.partition == "test"
    assert session.optimizer.evaluation_set.selection == CaseRange(stop=2)
    budget = session.budget_ledger.get(
        runner.BACKEND_ID,
        session.optimizer.evaluation_set,
    )
    assert budget is not None
    assert budget.remaining_runs == 2
    assert budget.max_cases_per_run == 2
    assert (tmp_path / "session" / "inputs" / "cases.jsonl").is_file()
    backend = session.optimizer.engine.backends.resolve(runner.BACKEND_ID)
    assert Path(backend.config.harness_root).name == "evaluator"


def test_runner_has_no_legacy_policy_dependency():
    source = Path(runner.__file__).read_text()
    assert "vero.core" not in source
    assert "vero.policy" not in source
    assert "ExperimentRunnerTool" not in source


@pytest.mark.asyncio
async def test_baseline_runs_end_to_end_through_external_harness(
    tmp_path,
    monkeypatch,
):
    target = _git_target(tmp_path / "target")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([{"x": 2}, {"x": 4}]))
    task = OptimizationTask(
        project_path=target,
        dataset_path=dataset,
        task="example",
        module="example_target.task",
        evaluation_budget=1,
    )
    monkeypatch.setattr(runner, "load_task", lambda _: task)
    session = await runner.build_benchmark_session(
        task_name="example",
        model="test-model",
        agent_name=None,
        session_dir=tmp_path / "session",
    )

    result = await session.run()

    assert result.baseline.objective is not None
    assert result.baseline.objective.value == 1.0
    assert [case.metrics["score"] for case in result.baseline.report.cases] == [
        1.0,
        1.0,
    ]
