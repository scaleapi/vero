"""Integration test for vero.harbor.serve — assemble the sidecar/verifier from a
ServeConfig and run a real (deterministic, no-LLM) Mode-A eval + finalize.

Reuses the external-task project pattern: a trivial agent + a separate task project,
scored deterministically. Validates that `build_components` produces a working engine,
and that a real eval flows into verifier selection + scoring.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from vero.core.dataset.store import resolve_and_save_dataset
from vero.evaluation.engine import EvalRequest
from vero.harbor.serve import ServeConfig, build_components


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=path, capture_output=True, check=True, text=True,
    ).stdout.strip()


def _create_agent(root: Path) -> tuple[Path, str]:
    d = root / "my-agent"
    (d / "src" / "my_agent").mkdir(parents=True)
    (d / "pyproject.toml").write_text(textwrap.dedent("""\
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
    (d / "src" / "my_agent" / "__init__.py").write_text('def solve(q): return "42"\n')
    _git(d, "init")
    _git(d, "add", ".")
    _git(d, "commit", "-m", "init")
    return d, _git(d, "rev-parse", "HEAD")


def _create_task_project(root: Path, vero_path: Path) -> Path:
    d = root / "my-eval-tasks"
    vt = d / "src" / "my_eval_tasks" / "vero_tasks"
    vt.mkdir(parents=True)
    (d / "pyproject.toml").write_text(textwrap.dedent(f"""\
        [project]
        name = "my-eval-tasks"
        version = "0.1.0"
        requires-python = ">=3.11"
        dependencies = ["scale-vero[optimize]"]
        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"
        [tool.hatch.build.targets.wheel]
        packages = ["src/my_eval_tasks"]
        [tool.uv.sources]
        scale-vero = {{ path = "{vero_path}", editable = true }}
    """))
    (vt / "__init__.py").write_text("from my_eval_tasks.vero_tasks import math_task  # noqa\n")
    (vt / "math_task.py").write_text(textwrap.dedent("""\
        from my_agent import solve
        from vero.core.db.result import TaskOutput, TaskResult
        from vero.core.evaluation import EvaluationParameters
        from vero.core.task import create_task
        math_task = create_task("math")
        @math_task.inference()
        async def run_inference(task, evaluation_parameters):
            return TaskOutput(output=solve(task["question"]))
        @math_task.evaluation()
        async def evaluate(task, output, evaluation_parameters):
            return TaskResult(output=output.output, score=1.0 if output.output == task["expected"] else 0.0)
    """))
    subprocess.run(["uv", "sync"], cwd=d, capture_output=True, check=True)
    return d


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    from vero.core.constants import PACKAGE_DIR
    from datasets import Dataset, DatasetDict

    vh = tmp_path / "vero_home"
    (vh / "sessions").mkdir(parents=True)
    (vh / "datasets").mkdir(parents=True)
    monkeypatch.setenv("VERO_HOME_DIR", str(vh))

    agent_dir, head = _create_agent(tmp_path)
    task_dir = _create_task_project(tmp_path, PACKAGE_DIR)
    ds = DatasetDict({"test": Dataset.from_dict(
        {"question": ["6*7?", "2+2?"], "expected": ["42", "4"]})})
    ds_path = tmp_path / "ds"
    ds.save_to_disk(str(ds_path))
    dataset_id = resolve_and_save_dataset(str(ds_path), vh / "sessions", vh / "datasets", "sess")
    return agent_dir, head, task_dir, dataset_id, tmp_path


def _serve_config(agent_dir, head, task_dir, dataset_id, tmp) -> ServeConfig:
    return ServeConfig(
        repo_path=str(agent_dir),
        agent_repo_path=str(agent_dir),
        session_id="sess",
        dataset_id=dataset_id,
        split_accesses=[{"split": "test", "access": "non_viewable"}],
        budgets=[{"split": "test", "dataset_id": dataset_id, "total_run_budget": 5}],
        task="math",
        task_project=str(task_dir),
        task_module="my_eval_tasks.vero_tasks",
        reward_mode="auto_best",
        selection_split="test",
        targets=[{"task": "math", "dataset_id": dataset_id, "split": "test", "reward_key": "reward", "sample_ids": [0]}],
        agent_volume=str(tmp / "agent_vol"),
        admin_volume=str(tmp / "admin_vol"),
        admin_token_path=str(tmp / "admin_vol" / "token"),
        timeout=300,
    )


@pytest.mark.asyncio
async def test_serve_assembles_and_evaluates_and_finalizes(fixture):
    agent_dir, head, task_dir, dataset_id, tmp = fixture
    config = _serve_config(agent_dir, head, task_dir, dataset_id, tmp)

    sidecar, verifier, token = await build_components(config)
    assert token and (tmp / "admin_vol" / "token").read_text() == token

    # real eval (no LLM): sample 0 expects "42", agent solves -> "42" -> score 1.0
    exp = await sidecar.engine.evaluate(
        EvalRequest(dataset_id=dataset_id, split="test", commit=head, sample_ids=[0])
    )
    assert exp.result.sample_results[0].score == 1.0

    # verifier selects the (only) candidate on "test" and scores it on the test target
    rewards = await verifier.finalize()
    assert rewards["reward"] == 1.0
