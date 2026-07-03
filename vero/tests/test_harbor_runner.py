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


class TestReviewFixes:
    def test_emits_attempts_and_retries(self):
        runner = HarborRunner(
            HarborConfig(
                task_source="org/ds@1",
                agent_import_path="pkg.mod:Agent",
                model="anthropic/x",
                environment="modal",
                n_attempts=3,
                max_retries=5,
            )
        )
        cmd = runner._build_command("/wt", _params(), ["t0"], Path("/jobs"))
        assert "--n-attempts" in cmd and cmd[cmd.index("--n-attempts") + 1] == "3"
        assert "--max-retries" in cmd and cmd[cmd.index("--max-retries") + 1] == "5"

    def test_passing_trial_wins_over_later_failing_retry(self, tmp_path):
        runner = _runner()
        jobs = tmp_path / "jobs"
        run = jobs / "2026-01-01__00-00-00"
        # passing trial, earlier finished_at
        good = run / "trial0"
        good.mkdir(parents=True)
        (good / "result.json").write_text(json.dumps({
            "task_name": "t0", "trial_name": "trial0", "finished_at": "2026-01-01T00:01:00",
            "verifier_result": {"rewards": {"pass": 1.0}},
        }))
        # failing retry, later finished_at + exception_info; written second (newer mtime)
        bad = run / "trial1"
        bad.mkdir(parents=True)
        (bad / "result.json").write_text(json.dumps({
            "task_name": "t0", "trial_name": "trial1", "finished_at": "2026-01-01T00:09:00",
            "exception_info": {"exception_type": "RuntimeError", "exception_message": "boom",
                               "exception_traceback": ""},
            "verifier_result": None,
        }))
        trials = runner._load_trials(jobs)
        assert trials["t0"]["trial_name"] == "trial0"
        assert (trials["t0"]["verifier_result"] or {}).get("rewards") == {"pass": 1.0}

    def test_latest_attempt_wins_when_both_clean(self, tmp_path):
        runner = _runner()
        jobs = tmp_path / "jobs"
        run = jobs / "2026-01-01__00-00-00"
        early = run / "a"
        early.mkdir(parents=True)
        (early / "result.json").write_text(json.dumps({
            "task_name": "t0", "trial_name": "early", "finished_at": "2026-01-01T00:01:00",
            "verifier_result": {"rewards": {"pass": 0.0}},
        }))
        late = run / "b"
        late.mkdir(parents=True)
        (late / "result.json").write_text(json.dumps({
            "task_name": "t0", "trial_name": "late", "finished_at": "2026-01-01T00:05:00",
            "verifier_result": {"rewards": {"pass": 1.0}},
        }))
        trials = runner._load_trials(jobs)
        assert trials["t0"]["trial_name"] == "late"

    @pytest.mark.asyncio
    async def test_resume_reruns_persisted_error_sample(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VERO_HOME_DIR", str(tmp_path / "vh"))
        runner = _runner()
        params = _params()
        result_dir = tmp_path / "result"
        # sample 0 previously errored (transient failure)
        save_sample_result(
            get_vero_home_dir() / "sessions", "s", params.result_id, sample_id=0,
            result=SampleResult(
                dataset_sample=DatasetSample(sample_id=0, split="test", dataset_id="ds"),
                error="transient harbor failure", commit="c1", result_id=params.result_id,
            ),
        )
        # a good trial for t0 now exists
        _write_trial(result_dir / "jobs", "trial0", "t0", {"pass": 1.0})
        monkeypatch.setattr(runner, "_task_names_for", lambda p: [(0, "t0")])
        runner._run_harbor = AsyncMock()
        ws = MagicMock(project_path="/wt")
        await runner.produce_sample_results(workspace=ws, params=params, result_dir=result_dir)
        # error sample was treated as pending (re-run) and re-collated to a score
        assert runner._run_harbor.await_args.args[2] == ["t0"]
        results = load_all_sample_results(get_vero_home_dir() / "sessions", "s", params.result_id)
        assert results[0].error is None
        assert results[0].score == 1.0


class TestCollateMismatchGuard:
    """_collate must not silently score 0.0 when the nested run's trials exist
    but match none of the requested task names (keying mismatch), or when the
    run produced no trials at all. Found live: a partition with bare TB2 names
    (vs harbor's canonical 'terminal-bench/<name>') zeroed a whole trial,
    anchors included, indistinguishable from total agent failure.
    """

    def test_zero_name_matches_raises(self, tmp_path):
        runner = _runner()
        jobs = tmp_path / "jobs"
        run = jobs / "2026-01-01__00-00-00"
        t = run / "trial0"
        t.mkdir(parents=True)
        (t / "result.json").write_text(json.dumps({
            "task_name": "terminal-bench/foo", "trial_name": "trial0",
            "verifier_result": {"rewards": {"reward": 1.0}},
        }))
        with pytest.raises(RuntimeError, match="none match the requested"):
            runner._collate(jobs, [(0, "foo")], _params(), ran=["foo"])

    def test_no_trials_at_all_raises(self, tmp_path):
        runner = _runner()
        jobs = tmp_path / "jobs"
        jobs.mkdir()
        with pytest.raises(RuntimeError, match="no trial results"):
            runner._collate(jobs, [(0, "foo")], _params(), ran=["foo"])

    def test_partial_match_still_collates(self, tmp_path, monkeypatch):
        # One of two tasks matched: not a keying mismatch; the missing task is
        # recorded as an error sample (existing behavior).
        runner = _runner()
        jobs = tmp_path / "jobs"
        run = jobs / "2026-01-01__00-00-00"
        t = run / "trial0"
        t.mkdir(parents=True)
        (t / "result.json").write_text(json.dumps({
            "task_name": "t0", "trial_name": "trial0",
            "verifier_result": {"rewards": {"reward": 1.0}},
        }))
        saved = []
        monkeypatch.setattr(
            "vero.harbor.runner.save_sample_result",
            lambda *a, **k: saved.append(k.get("sample_id")),
        )
        runner._collate(jobs, [(0, "t0"), (1, "missing")], _params(), ran=["t0", "missing"])
        assert saved == [0, 1]

    def test_resume_with_nothing_ran_skips_guard(self, tmp_path, monkeypatch):
        # All samples already done (resume): no `ran` tasks, empty jobs dir is fine.
        runner = _runner()
        monkeypatch.setattr(HarborRunner, "_is_done", lambda self, p, s: True)
        runner._collate(tmp_path / "jobs", [(0, "t0")], _params(), ran=[])


class TestMeanAttemptAggregation:
    """aggregate_attempts='mean': average the reward across every SCORED
    attempt, dirty or clean (de-noising; estimates per-attempt pass
    probability). Harbor scores timed-out attempts 0.0 while also recording
    the exception; those must count, or the mean forgives slow candidates.
    Default 'best' keeps the existing latest-clean behavior, which inflates
    toward pass@k.
    """

    def _write(self, run, trial, task, rewards=None, exc=False):
        d = run / trial
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "task_name": task, "trial_name": trial,
            "finished_at": f"2026-01-01T00:0{len(trial) % 10}:00",
            "verifier_result": {"rewards": rewards} if rewards else None,
            "exception_info": {"exception_type": "X", "exception_message": "",
                               "exception_traceback": ""} if exc else None,
        }))

    def test_mean_averages_clean_attempts(self, tmp_path):
        runner = HarborRunner(HarborConfig(
            task_source="org/ds", agent_import_path="p:m",
            n_attempts=2, aggregate_attempts="mean",
        ))
        jobs = tmp_path / "jobs"; run = jobs / "2026-01-01__00-00-00"
        self._write(run, "t0a", "t0", rewards={"reward": 1.0})
        self._write(run, "t0b", "t0", rewards={"reward": 0.0})
        groups = runner._trial_groups(jobs)
        r = runner._sample_result(groups["t0"][0], 0, "t0", _params(), attempts=groups["t0"])
        assert r.score == 0.5
        assert r.metrics["n_scored"] == 2.0

    def test_mean_excludes_attempts_without_rewards(self, tmp_path):
        # An attempt that died before the verifier scored it carries no
        # measurement; it is excluded (but still counted in n_attempts).
        runner = HarborRunner(HarborConfig(
            task_source="org/ds", agent_import_path="p:m",
            n_attempts=2, aggregate_attempts="mean",
        ))
        jobs = tmp_path / "jobs"; run = jobs / "2026-01-01__00-00-00"
        self._write(run, "t0a", "t0", rewards={"reward": 1.0})
        self._write(run, "t0bad", "t0", exc=True)
        groups = runner._trial_groups(jobs)
        r = runner._sample_result(groups["t0"][0], 0, "t0", _params(), attempts=groups["t0"])
        assert r.score == 1.0
        assert r.metrics["n_scored"] == 1.0
        assert r.metrics["n_attempts"] == 2.0

    def test_mean_counts_scored_exception_attempts(self, tmp_path):
        # The live-GAIA shape: harbor records AgentTimeoutError but still runs
        # the verifier, so the attempt has BOTH exception_info and a scored 0.0.
        # [1.0 clean, 0.0 timeout, 0.0 timeout] must score 1/3, not 1.0.
        runner = HarborRunner(HarborConfig(
            task_source="org/ds", agent_import_path="p:m",
            n_attempts=3, aggregate_attempts="mean",
        ))
        jobs = tmp_path / "jobs"; run = jobs / "2026-01-01__00-00-00"
        self._write(run, "t0a", "t0", rewards={"reward": 1.0})
        self._write(run, "t0b", "t0", rewards={"reward": 0.0}, exc=True)
        self._write(run, "t0c", "t0", rewards={"reward": 0.0}, exc=True)
        groups = runner._trial_groups(jobs)
        r = runner._sample_result(groups["t0"][0], 0, "t0", _params(), attempts=groups["t0"])
        assert r.score == pytest.approx(1 / 3)
        assert r.metrics["n_scored"] == 3.0
        assert r.metrics["n_clean"] == 1.0

    def test_mean_over_all_dirty_attempts(self, tmp_path):
        # Every attempt timed out but was scored (the all-timeouts live shape):
        # the mean path must still apply, not the single-best-trial fallback.
        runner = HarborRunner(HarborConfig(
            task_source="org/ds", agent_import_path="p:m",
            n_attempts=2, aggregate_attempts="mean",
        ))
        jobs = tmp_path / "jobs"; run = jobs / "2026-01-01__00-00-00"
        self._write(run, "t0a", "t0", rewards={"reward": 1.0}, exc=True)
        self._write(run, "t0b", "t0", rewards={"reward": 0.0}, exc=True)
        groups = runner._trial_groups(jobs)
        r = runner._sample_result(groups["t0"][0], 0, "t0", _params(), attempts=groups["t0"])
        assert r.score == 0.5
        assert r.metrics["n_clean"] == 0.0
        assert r.output["aggregate"] == "mean"

    def test_default_best_unchanged(self, tmp_path):
        # No attempts passed (default 'best' config): single-trial path intact.
        runner = _runner()
        r = runner._sample_result(
            {"task_name": "t0", "trial_name": "x",
             "verifier_result": {"rewards": {"reward": 1.0}}},
            0, "t0", _params(),
        )
        assert r.score == 1.0
