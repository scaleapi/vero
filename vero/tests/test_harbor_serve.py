"""Integration test for vero.harbor.serve — assemble the sidecar/verifier from a
ServeConfig and run a real (deterministic, no-LLM) Mode-A eval + finalize.

Reuses the external-task project pattern: a trivial agent + a separate task project,
scored deterministically. Validates that `build_components` produces a working engine,
and that a real eval flows into verifier selection + scoring.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from vero.core.dataset.store import resolve_and_save_dataset
from vero.evaluation.engine import EvalRequest
from vero.harbor.serve import (
    ServeConfigA,
    ServeConfigB,
    _load_or_build_ledger,
    build_components,
)


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


def _serve_config(agent_dir, head, task_dir, dataset_id, tmp) -> ServeConfigA:
    return ServeConfigA(
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
    rewards = (await verifier.finalize())["rewards"]
    assert rewards["reward"] == 1.0


def _create_cheating_agent(root: Path) -> tuple[Path, str]:
    """An agent whose committed repo ALSO ships a vero_tasks scorer that returns 1.0
    for everything, and whose solve() returns a wrong answer. If the verifier ran
    the agent's scorer, reward would be 1.0; with the sidecar-baked task project it
    must reflect the real (0.0) score."""
    d = root / "cheating-agent"
    vt = d / "src" / "my_agent" / "vero_tasks"
    vt.mkdir(parents=True)
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
    (d / "src" / "my_agent" / "__init__.py").write_text('def solve(q): return "WRONG"\n')
    # Adversarial scorer baked into the AGENT repo: always 1.0.
    (vt / "__init__.py").write_text("from my_agent.vero_tasks import math_task  # noqa\n")
    (vt / "math_task.py").write_text(textwrap.dedent("""\
        from vero.core.db.result import TaskOutput, TaskResult
        from vero.core.task import create_task
        math_task = create_task("math")
        @math_task.inference()
        async def run_inference(task, evaluation_parameters):
            return TaskOutput(output="WRONG")
        @math_task.evaluation()
        async def evaluate(task, output, evaluation_parameters):
            return TaskResult(output=output.output, score=1.0)  # always passes
    """))
    _git(d, "init")
    _git(d, "add", ".")
    _git(d, "commit", "-m", "init")
    return d, _git(d, "rev-parse", "HEAD")


@pytest.mark.asyncio
async def test_finalize_does_not_run_agent_supplied_scorer(fixture):
    # Reuse the trusted task project from the fixture; swap in a cheating agent
    # whose committed repo carries a 1.0 scorer and a wrong solve().
    _, _, task_dir, dataset_id, tmp = fixture
    agent_dir, head = _create_cheating_agent(tmp)
    config = _serve_config(agent_dir, head, task_dir, dataset_id, tmp)

    sidecar, verifier, _ = await build_components(config)
    # Real eval: agent answers "WRONG" for sample 0 (expects "42"); trusted scorer -> 0.0
    exp = await sidecar.engine.evaluate(
        EvalRequest(dataset_id=dataset_id, split="test", commit=head, sample_ids=[0])
    )
    assert exp.result.sample_results[0].score == 0.0
    # Finalize must reflect the TRUSTED score, not the agent's 1.0 scorer.
    rewards = (await verifier.finalize())["rewards"]
    assert rewards["reward"] == 0.0


@pytest.mark.asyncio
async def test_ledger_reloads_spent_budget_across_restart(fixture):
    agent_dir, head, task_dir, dataset_id, tmp = fixture
    config = _serve_config(agent_dir, head, task_dir, dataset_id, tmp)

    # First boot: spend one run on the test split.
    sidecar, _, _ = await build_components(config)
    before = sidecar.engine.budget.get(dataset_id, "test").remaining_run_budget
    await sidecar.engine.evaluate(
        EvalRequest(dataset_id=dataset_id, split="test", commit=head, sample_ids=[0])
    )
    after = sidecar.engine.budget.get(dataset_id, "test").remaining_run_budget
    assert after == before - 1

    # Restart: rebuild from the SAME config + admin_volume; spent budget must persist.
    sidecar2, _, _ = await build_components(config)
    reloaded = sidecar2.engine.budget.get(dataset_id, "test").remaining_run_budget
    assert reloaded == after, "sidecar restart must not refill spent budget"


class TestLedgerFailClosed:
    """A persisted ledger that exists but cannot be read fails CLOSED: spend
    that cannot be reconstructed is treated as fully spent. The old fallback
    (configured budgets) refunded the agent everything already spent, so any
    crash that corrupted the flush minted budget."""

    _CFGS = [{
        "split": "validation", "dataset_id": "ds",
        "total_run_budget": 5, "total_sample_budget": 50,
    }]

    def test_missing_file_boots_configured(self, tmp_path):
        led = _load_or_build_ledger(self._CFGS, tmp_path / "ledger.json")
        b = led.get("ds", "validation")
        assert b.remaining_run_budget == 5
        assert b.remaining_sample_budget == 50

    def test_unparseable_file_fails_closed_and_keeps_evidence(self, tmp_path):
        p = tmp_path / "ledger.json"
        p.write_text("{definitely not json")
        led = _load_or_build_ledger(self._CFGS, p)
        b = led.get("ds", "validation")
        assert b.remaining_run_budget == 0
        assert b.remaining_sample_budget == 0
        # the unreadable original survives for the operator to inspect
        assert p.with_suffix(".corrupt").read_text() == "{definitely not json"

    def test_malformed_entries_fail_closed(self, tmp_path):
        p = tmp_path / "ledger.json"
        p.write_text(json.dumps([{"no_split_key": 1}]))
        led = _load_or_build_ledger(self._CFGS, p)
        assert led.get("ds", "validation").remaining_run_budget == 0


@pytest.mark.asyncio
async def test_feedback_levers_reach_harbor_runner(fixture):
    # Lever 1 pass-through: ServeConfigB -> build_components -> HarborRunner kwargs
    # (mirrors how score_baseline reaches the Verifier). Built as a Mode-B config
    # directly: the levers are Mode-B-only under the type split.
    agent_dir, head, task_dir, dataset_id, tmp = fixture
    config = ServeConfigB(
        mode="B",
        repo_path=str(agent_dir),
        agent_repo_path=str(agent_dir),
        session_id="sess",
        dataset_id=dataset_id,
        split_accesses=[{"split": "test", "access": "non_viewable"}],
        budgets=[{"split": "test", "dataset_id": dataset_id, "total_run_budget": 5}],
        harbor={"task_source": "org/x", "agent_import_path": "p:C"},
        feedback_transcripts=True,
        feedback_max_bytes=512,
        expose_attempt_detail=True,
        reward_mode="auto_best",
        selection_split="test",
        agent_volume=str(tmp / "agent_vol"),
        admin_volume=str(tmp / "admin_vol"),
        admin_token_path=str(tmp / "admin_vol" / "token"),
        timeout=300,
    )
    sidecar, _, _ = await build_components(config)
    runner = sidecar.engine.evaluator.eval_strategy
    assert runner.feedback_transcripts is True
    assert runner.feedback_max_bytes == 512
    assert runner.expose_attempt_detail is True


def test_mode_mismatch_fields_rejected():
    # PR #20's runtime warnings are superseded by the Mode-A / Mode-B type split:
    # a wrong-mode field is now a load-time ValidationError, not a no-op. Mode B
    # has no sample_timeout, and Mode A has no feedback levers / harbor.
    from pydantic import ValidationError

    base = dict(
        repo_path="/r", agent_repo_path="/a", session_id="s", dataset_id="ds",
        split_accesses=[], budgets=[], agent_volume="/v", admin_volume="/adm",
        admin_token_path="/t",
    )
    # Mode B rejects sample_timeout.
    with pytest.raises(ValidationError):
        ServeConfigB(mode="B", harbor=None, sample_timeout=1200, **base)
    # Mode A rejects harbor / feedback levers.
    with pytest.raises(ValidationError):
        ServeConfigA(harbor={"task_source": "org/x"}, **base)
    with pytest.raises(ValidationError):
        ServeConfigA(feedback_transcripts=True, **base)
    # The valid arrangements still construct.
    ServeConfigA(sample_timeout=900, task_project="/tp", **base)
    ServeConfigB(mode="B", harbor={"task_source": "org/x"}, **base)
