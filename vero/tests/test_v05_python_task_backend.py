from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vero.evaluation import (
    AllCases,
    CaseIds,
    CaseRange,
    EvaluationSet,
    PythonTaskBackend,
    PythonTaskBackendConfig,
    MetricSelector,
    ObjectiveSpec,
)
from vero.optimization import (
    CommandCandidateProducer,
    CommandCandidateProducerConfig,
)
from vero.runtime import create_local_optimization_session


def initialize_repository(path: Path) -> None:
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "--all"],
        cwd=path,
        check=True,
        capture_output=True,
    )
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
        capture_output=True,
    )


@pytest.mark.asyncio
async def test_python_task_backend_builds_uv_command_and_resolves_cost(tmp_path: Path):
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps([{"id": "a"}, {"id": "b"}, {"id": "c"}]),
        encoding="utf-8",
    )
    backend = PythonTaskBackend(
        PythonTaskBackendConfig(
            harness_root=str(tmp_path),
            module="target.tasks",
            task="quality",
            cases_path=str(cases),
            target_project_directory="packages/target",
            uv_executable=sys.executable,
        )
    )

    command = backend._command.config.command
    assert command[:6] == [
        sys.executable,
        "run",
        "--project",
        "{harness}",
        "--with-editable",
        "{workspace}/packages/target",
    ]
    assert command[-4:] == ["--request", "{request}", "--report", "{report}"]
    assert "{input:cases}" in command
    assert (await backend.resolve_cost(EvaluationSet(selection=AllCases()))).cases == 3
    assert (
        await backend.resolve_cost(EvaluationSet(selection=CaseRange(start=1, stop=3)))
    ).cases == 2
    assert (
        await backend.resolve_cost(EvaluationSet(selection=CaseIds(ids=["a", "c"])))
    ).cases == 2

    with pytest.raises(ValueError, match="contains 3 cases"):
        await backend.resolve_cost(EvaluationSet(selection=CaseRange(stop=4)))

    with pytest.raises(ValueError, match="unknown Python task case IDs"):
        await backend.resolve_cost(EvaluationSet(selection=CaseIds(ids=["missing"])))


def test_python_task_backend_requires_external_case_file(tmp_path: Path):
    missing = tmp_path / "missing.json"

    with pytest.raises(ValueError, match="existing file"):
        PythonTaskBackendConfig(
            harness_root=str(tmp_path),
            module="target.tasks",
            task="quality",
            cases_path=str(missing),
            uv_executable=sys.executable,
        )


@pytest.mark.asyncio
async def test_python_task_backend_optimizes_target_task_module(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "factor.txt").write_text("1\n", encoding="utf-8")
    (target / "target_program.py").write_text(
        """
from pathlib import Path

def multiply(value):
    factor = int(Path(__file__).with_name("factor.txt").read_text())
    return value * factor
""",
        encoding="utf-8",
    )
    (tmp_path / "evaluation_tasks.py").write_text(
        """
from vero_tasks import TaskOutput, TaskResult, create_task
from target_program import multiply

task = create_task("multiply")

@task.inference()
async def infer(case, context):
    return TaskOutput(output=multiply(case["value"]))

@task.evaluation()
async def evaluate(case, output, context):
    return TaskResult.from_task_output(
        output,
        score=float(output.output == case["expected"]),
    )
""",
        encoding="utf-8",
    )
    initialize_repository(target)

    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {"id": "one", "value": 2, "expected": 4},
                {"id": "two", "value": 3, "expected": 6},
            ]
        ),
        encoding="utf-8",
    )
    tasks_source = Path(__file__).parents[2] / "vero-tasks" / "src"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        f"""#!{sys.executable}
import runpy
import sys

arguments = sys.argv[1:]
project = arguments[arguments.index("--project") + 1]
target = arguments[arguments.index("--with-editable") + 1]
module_index = arguments.index("-m")
module = arguments[module_index + 1]
sys.path[:0] = [project, target, {str(tasks_source)!r}]
sys.argv = [module, *arguments[module_index + 2:]]
runpy.run_module(module, run_name="__main__")
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    producer_root = tmp_path / "producer"
    producer_root.mkdir()
    producer_script = producer_root / "improve.py"
    producer_script.write_text(
        """
import sys
from pathlib import Path
Path(sys.argv[1], "factor.txt").write_text("2\\n")
""",
        encoding="utf-8",
    )
    backend = PythonTaskBackend(
        PythonTaskBackendConfig(
            harness_root=str(tmp_path),
            module="evaluation_tasks",
            task="multiply",
            cases_path=str(cases),
            uv_executable=str(fake_uv),
        )
    )
    producer = CommandCandidateProducer(
        CommandCandidateProducerConfig(
            root=str(producer_root),
            command=[sys.executable, str(producer_script), "{workspace}"],
        )
    )
    session = await create_local_optimization_session(
        project_path=target,
        session_dir=tmp_path / "sessions" / "python-task",
        backend_id="python-task",
        backend=backend,
        objective=ObjectiveSpec(
            selector=MetricSelector(metric="score"),
            direction="maximize",
        ),
        producers={"default": producer},
        max_candidates=1,
    )

    result = await session.run()

    assert result.baseline.objective.value == 0.0
    assert result.best.objective.value == 1.0
    assert [case.case_id for case in result.best.report.cases] == ["one", "two"]
    assert (target / "factor.txt").read_text(encoding="utf-8") == "1\n"
