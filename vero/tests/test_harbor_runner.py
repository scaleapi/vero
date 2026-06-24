"""Tests for vero.harbor.runner.HarborRunner — command build, collation, resume."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from vero.core.db.candidate import Candidate
from vero.core.db.dataset import DatasetSample, DatasetSubset
from vero.core.db.result import SampleResult
from vero.core.db.run import ExperimentRun
from vero.core.evaluation import EvaluationParameters
from vero.core.sessions import (
    get_vero_home_dir,
    load_all_sample_results,
    save_sample_result,
)
from vero.harbor.config import HarborConfig
from vero.harbor.runner import HarborRunner


def _runner(reward_key=None, task_source="org/ds@1"):
    return HarborRunner(
        HarborConfig(
            task_source=task_source,
            agent_import_path="pkg.mod:Agent",
            model="anthropic/x",
            environment="modal",
            reward_key=reward_key,
        )
    )


def _params():
    return EvaluationParameters(
        run=ExperimentRun(
            candidate=Candidate(commit="c1", repo_name="r"),
            dataset_subset=DatasetSubset(split="test", dataset_id="ds", sample_ids=[0, 1]),
        ),
        session_id="s",
    )


def _write_trial(jobs_dir: Path, trial: str, task_name: str, rewards: dict):
    # Real harbor layout: <jobs>/<timestamp>/<trial>/result.json, plus a job-level
    # <jobs>/<timestamp>/result.json summary (no task_name) that collation must skip.
    run = jobs_dir / "2026-01-01__00-00-00"
    d = run / trial
    d.mkdir(parents=True, exist_ok=True)
    (run / "result.json").write_text(json.dumps({"job": "summary"}))  # job-level, no task_name
    (d / "result.json").write_text(
        json.dumps({"task_name": task_name, "trial_name": trial, "verifier_result": {"rewards": rewards}})
    )


class TestBuildCommand:
    def test_registry_source_and_flags(self):
        cmd = _runner()._build_command("/wt", _params(), ["t0", "t1"], Path("/jobs"))
        assert cmd[:5] == ["uv", "run", "--project", "/wt", "harbor"]
        assert "-d" in cmd and "org/ds@1" in cmd
        assert "--agent-import-path" in cmd and "pkg.mod:Agent" in cmd
        assert cmd.count("-i") == 2 and "t0" in cmd and "t1" in cmd
        assert "-m" in cmd and "-e" in cmd and "--jobs-dir" in cmd

    def test_local_source(self, tmp_path):
        cmd = _runner(task_source=str(tmp_path))._build_command("/wt", _params(), ["t0"], Path("/jobs"))
        assert "-p" in cmd and str(tmp_path) in cmd
        assert "-d" not in cmd


class TestExtractReward:
    def test_priority_pass_then_reward_then_mean(self):
        r = _runner()
        assert r._extract_reward({"pass": 1.0, "reward": 0.0}) == 1.0
        assert r._extract_reward({"reward": 0.7}) == 0.7
        assert r._extract_reward({"a": 0.2, "b": 0.4}) == pytest.approx(0.3)

    def test_reward_key_override(self):
        assert _runner(reward_key="acc")._extract_reward({"acc": 0.9, "pass": 0.0}) == 0.9


class TestCollate:
    @pytest.mark.asyncio
    async def test_produces_results_and_marks_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VERO_HOME_DIR", str(tmp_path / "vh"))
        runner = _runner()
        params = _params()
        result_dir = tmp_path / "result"
        jobs = result_dir / "jobs"
        _write_trial(jobs, "trial0", "t0", {"pass": 1.0, "extra": 0.5})
        # no trial for t1

        monkeypatch.setattr(runner, "_task_names_for", lambda p: [(0, "t0"), (1, "t1")])
        runner._run_harbor = AsyncMock()  # fixtures already present; don't shell out

        ws = MagicMock(project_path="/wt")
        await runner.produce_sample_results(workspace=ws, params=params, result_dir=result_dir)

        results = load_all_sample_results(get_vero_home_dir() / "sessions", "s", params.result_id)
        assert results[0].score == 1.0
        assert results[0].metrics["extra"] == 0.5
        assert results[1].error is not None  # missing trial -> error sample

    @pytest.mark.asyncio
    async def test_resume_only_runs_pending(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VERO_HOME_DIR", str(tmp_path / "vh"))
        runner = _runner()
        params = _params()
        result_dir = tmp_path / "result"

        # sample 0 already done
        save_sample_result(
            get_vero_home_dir() / "sessions", "s", params.result_id, sample_id=0,
            result=SampleResult(
                dataset_sample=DatasetSample(sample_id=0, split="test", dataset_id="ds"),
                score=1.0, commit="c1", result_id=params.result_id,
            ),
        )
        _write_trial(result_dir / "jobs", "trial1", "t1", {"pass": 0.0})
        monkeypatch.setattr(runner, "_task_names_for", lambda p: [(0, "t0"), (1, "t1")])
        runner._run_harbor = AsyncMock()

        ws = MagicMock(project_path="/wt")
        await runner.produce_sample_results(workspace=ws, params=params, result_dir=result_dir)

        # only the pending task name was passed to harbor
        assert runner._run_harbor.await_args.args[2] == ["t1"]
