"""E2E test: task module in a separate uv project from the agent.

Creates:
- my-agent: a trivial agent package with solve() → "42"
- my-eval-tasks: a separate task package that imports my_agent.solve and scores it

Verifies:
1. Evaluator can run with task_project + task_module + --with-editable
2. Agent code is imported from the correct worktree (not stale)
3. Changing agent code and re-evaluating produces different scores
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from vero.evaluation.evaluator import run_evaluation


def _init_git(path: Path) -> None:
    """Initialize a git repo with an initial commit."""
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@test", "commit", "-m", "init"],
        cwd=path, capture_output=True, check=True,
    )


def _create_agent(root: Path) -> Path:
    """Create a minimal agent package."""
    agent_dir = root / "my-agent"
    src = agent_dir / "src" / "my_agent"
    src.mkdir(parents=True)

    (agent_dir / "pyproject.toml").write_text(textwrap.dedent("""\
        [project]
        name = "my-agent"
        version = "0.1.0"
        requires-python = ">=3.11"

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["src/my_agent"]
    """))

    (src / "__init__.py").write_text('def solve(question: str) -> str:\n    return "42"\n')

    _init_git(agent_dir)
    return agent_dir


def _create_task_project(root: Path, vero_path: Path) -> Path:
    """Create a separate task project that imports and scores the agent."""
    task_dir = root / "my-eval-tasks"
    src = task_dir / "src" / "my_eval_tasks"
    vero_tasks = src / "vero_tasks"
    vero_tasks.mkdir(parents=True)

    (task_dir / "pyproject.toml").write_text(textwrap.dedent(f"""\
        [project]
        name = "my-eval-tasks"
        version = "0.1.0"
        requires-python = ">=3.11"
        dependencies = ["scale-vero[evaluate]"]

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["src/my_eval_tasks"]

        [tool.uv.sources]
        scale-vero = {{ path = "{vero_path}", editable = true }}
    """))

    (src / "__init__.py").write_text("")

    (vero_tasks / "__init__.py").write_text("from . import math_task  # noqa: F401\n")

    (vero_tasks / "math_task.py").write_text(textwrap.dedent("""\
        from vero.core.task import create_task
        from vero.core.db.result import TaskOutput, TaskResult
        from vero.core.evaluation import EvaluationParameters

        math_task = create_task("math")

        @math_task("run_inference")
        async def run_inference(task: dict, evaluation_parameters: EvaluationParameters) -> TaskOutput:
            from my_agent import solve
            answer = solve(task["question"])
            return TaskOutput(output=answer)

        @math_task("run_evaluation")
        async def run_evaluation(task: dict, output: TaskOutput, evaluation_parameters: EvaluationParameters) -> TaskResult:
            score = 1.0 if output.output == task["expected"] else 0.0
            return TaskResult(output=output.output, score=score)
    """))

    # Sync the task project
    subprocess.run(["uv", "sync"], cwd=task_dir, capture_output=True, check=True)

    return task_dir


def _create_dataset(root: Path) -> Path:
    """Create a minimal HuggingFace dataset."""
    from datasets import Dataset, DatasetDict

    ds = DatasetDict({
        "test": Dataset.from_dict({
            "question": ["What is 6 * 7?", "What is 2 + 2?"],
            "expected": ["42", "4"],
        })
    })
    dataset_path = root / "dataset"
    ds.save_to_disk(str(dataset_path))
    return dataset_path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Create agent, task project, and dataset in a temp dir.

    Also redirects VERO_HOME_DIR so sessions are written to tmp_path,
    not ~/.vero. The env var is inherited by subprocesses.
    """
    from vero.core.constants import PACKAGE_DIR

    vero_home = tmp_path / "vero_home"
    vero_home.mkdir()
    (vero_home / "sessions").mkdir()
    (vero_home / "datasets").mkdir()
    monkeypatch.setenv("VERO_HOME_DIR", str(vero_home))

    agent_dir = _create_agent(tmp_path)
    task_dir = _create_task_project(tmp_path, vero_path=PACKAGE_DIR)
    dataset_path = _create_dataset(tmp_path)

    yield agent_dir, task_dir, dataset_path, vero_home


@pytest.mark.asyncio
async def test_external_task_evaluates_agent(workspace):
    """Evaluate agent using a task from a separate project."""
    agent_dir, task_dir, dataset_path, vero_home = workspace

    result = await run_evaluation(
        project_path=agent_dir,
        dataset=str(dataset_path),
        split="test",
        task="math",
        task_project=task_dir,
        task_module="my_eval_tasks.vero_tasks",
        sample_ids=[0],
        timeout=120,
        vero_home=vero_home,
    )

    assert result is not None
    assert len(result.sample_results) == 1
    # Agent returns "42", first sample expects "42" → score 1.0
    assert result.sample_results[0].score == 1.0


@pytest.mark.asyncio
async def test_external_task_sees_correct_agent_version(workspace):
    """Changing agent code changes eval results (worktree versioning works)."""
    agent_dir, task_dir, dataset_path, vero_home = workspace

    # Evaluate with original agent (solve returns "42")
    result_v1 = await run_evaluation(
        project_path=agent_dir,
        dataset=str(dataset_path),
        split="test",
        task="math",
        task_project=task_dir,
        task_module="my_eval_tasks.vero_tasks",
        sample_ids=[0],
        timeout=120,
        vero_home=vero_home,
    )
    assert result_v1.sample_results[0].score == 1.0

    # Change agent: solve now returns "wrong"
    agent_src = agent_dir / "src" / "my_agent" / "__init__.py"
    agent_src.write_text('def solve(question: str) -> str:\n    return "wrong"\n')
    subprocess.run(
        ["git", "add", "."], cwd=agent_dir, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@test", "commit", "-m", "break agent"],
        cwd=agent_dir, capture_output=True, check=True,
    )

    # Evaluate again — should get score 0.0 now
    result_v2 = await run_evaluation(
        project_path=agent_dir,
        dataset=str(dataset_path),
        split="test",
        task="math",
        task_project=task_dir,
        task_module="my_eval_tasks.vero_tasks",
        sample_ids=[0],
        timeout=120,
        vero_home=vero_home,
    )
    assert result_v2.sample_results[0].score == 0.0
