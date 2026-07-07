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
from vero.utils import SubprocessTimeoutError
from vero.utils.asyncio import SubprocessResult


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


def _write_trial(
    jobs_dir: Path,
    trial: str,
    task_name: str,
    rewards: dict | None,
    *,
    pane: str | None = None,
    trajectory: str | None = None,
    finished_at: str | None = None,
    exception_type: str | None = None,
):
    # Real harbor layout: <jobs>/<timestamp>/<trial>/result.json, plus a job-level
    # <jobs>/<timestamp>/result.json summary (no task_name) that collation must skip.
    # Transcripts (when present) live at <trial>/agent/terminus_2.pane and
    # <trial>/agent/trajectory.json, next to result.json.
    run = jobs_dir / "2026-01-01__00-00-00"
    d = run / trial
    d.mkdir(parents=True, exist_ok=True)
    (run / "result.json").write_text(json.dumps({"job": "summary"}))  # job-level, no task_name
    data = {
        "task_name": task_name,
        "trial_name": trial,
        "verifier_result": {"rewards": rewards} if rewards is not None else None,
    }
    if finished_at is not None:
        data["finished_at"] = finished_at
    if exception_type is not None:
        data["exception_info"] = {
            "exception_type": exception_type,
            "exception_message": "",
            "exception_traceback": "",
        }
    (d / "result.json").write_text(json.dumps(data))
    if pane is not None:
        (d / "agent").mkdir(exist_ok=True)
        (d / "agent" / "terminus_2.pane").write_text(pane)
    if trajectory is not None:
        (d / "agent").mkdir(exist_ok=True)
        (d / "agent" / "trajectory.json").write_text(trajectory)


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
    def test_priority_pass_then_reward_then_sole_key(self):
        r = _runner()
        assert r._extract_reward({"pass": 1.0, "reward": 0.0}) == 1.0
        assert r._extract_reward({"reward": 0.7}) == 0.7
        assert r._extract_reward({"accuracy": 0.9}) == 0.9  # sole key: unambiguous

    def test_several_unknown_keys_refused_not_averaged(self):
        # Averaging arbitrary keys would let a candidate inflate its score by
        # emitting easy auxiliary metrics beside the real one.
        assert _runner()._extract_reward({"a": 0.2, "b": 0.4}) is None

    def test_reward_key_override(self):
        assert _runner(reward_key="acc")._extract_reward({"acc": 0.9, "pass": 0.0}) == 0.9

    def test_configured_key_is_strict_no_fallback(self):
        # A configured reward_key missing from the dict is unscorable (None),
        # never a silent substitution of 'pass'/'reward'.
        assert _runner(reward_key="acc")._extract_reward({"pass": 1.0}) is None


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
        monkeypatch.setenv("VERO_HOME_DIR", str(tmp_path / "vh"))  # _is_done must not read the real home
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
    """aggregate_attempts='mean': average the reward across every attempt
    that RAN (de-noising; estimates per-attempt pass probability). Harbor
    scores timed-out attempts 0.0 while also recording the exception; those
    must count, or the mean forgives slow candidates. Attempts that died
    BEFORE scoring count 0.0 too (n_dead in metrics), or dying early becomes
    a scoring exploit. Default 'best' picks the single highest-scoring clean
    trial (pass@k-like).
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

    def test_mean_zero_fills_attempts_without_rewards(self, tmp_path):
        # An attempt that died before the verifier scored it is a real, failed
        # attempt and counts 0.0. Excluding it would estimate
        # P(pass | attempt survived), which rewards dying early on hard tasks:
        # a live optimizer won selection on exactly that artifact.
        runner = HarborRunner(HarborConfig(
            task_source="org/ds", agent_import_path="p:m",
            n_attempts=2, aggregate_attempts="mean",
        ))
        jobs = tmp_path / "jobs"; run = jobs / "2026-01-01__00-00-00"
        self._write(run, "t0a", "t0", rewards={"reward": 1.0})
        self._write(run, "t0bad", "t0", exc=True)
        groups = runner._trial_groups(jobs)
        r = runner._sample_result(groups["t0"][0], 0, "t0", _params(), attempts=groups["t0"])
        assert r.score == 0.5
        assert r.metrics["n_scored"] == 1.0
        assert r.metrics["n_dead"] == 1.0
        assert r.metrics["n_attempts"] == 2.0

    def test_mean_all_attempts_dead_errors_not_zero(self, tmp_path):
        # Every attempt died before scoring: that is an outage to investigate,
        # not a silent 0.0 measurement; the sample must surface as an error.
        runner = HarborRunner(HarborConfig(
            task_source="org/ds", agent_import_path="p:m",
            n_attempts=2, aggregate_attempts="mean",
        ))
        jobs = tmp_path / "jobs"; run = jobs / "2026-01-01__00-00-00"
        self._write(run, "t0a", "t0", exc=True)
        self._write(run, "t0b", "t0", exc=True)
        groups = runner._trial_groups(jobs)
        r = runner._sample_result(groups["t0"][0], 0, "t0", _params(), attempts=groups["t0"])
        assert r.error is not None
        assert r.score is None

    def test_best_rank_is_monotone_in_reward(self, tmp_path):
        # 'best' must never let a later clean 0.0 clobber an earlier clean 1.0:
        # the reward is part of the rank, recency only breaks ties.
        runner = _runner()
        jobs = tmp_path / "jobs"; run = jobs / "2026-01-01__00-00-00"
        # _write derives finished_at from len(trial)%10: "t0a" -> 00:03,
        # "t0badlate" -> 00:09, so the 0.0 trial genuinely finishes LATER.
        self._write(run, "t0a", "t0", rewards={"reward": 1.0})
        self._write(run, "t0badlate", "t0", rewards={"reward": 0.0})
        trials = runner._load_trials(jobs)
        assert (trials["t0"]["verifier_result"]["rewards"]["reward"]) == 1.0

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


class TestAggregateAttemptsValidation:
    """A mistyped aggregate_attempts value must fail loudly at construction:
    only the exact string 'mean' activates de-noising, so 'Mean'/'avg' would
    otherwise silently run inflated best-of-k."""

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="aggregate_attempts"):
            HarborConfig(
                task_source="org/ds", agent_import_path="p:m",
                aggregate_attempts="Mean",
            )

    def test_valid_values_accepted(self):
        for value in ("best", "mean"):
            cfg = HarborConfig(
                task_source="org/ds", agent_import_path="p:m",
                aggregate_attempts=value,
            )
            assert cfg.aggregate_attempts == value


class TestTimeoutSalvage:
    """A nested `harbor run` cut off by the vero-side timeout must salvage the
    trials that completed instead of erroring the whole (already-debited)
    eval with nothing."""

    @pytest.mark.asyncio
    async def test_timeout_collates_partials(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VERO_HOME_DIR", str(tmp_path / "vh"))
        runner = _runner()
        params = _params()
        result_dir = tmp_path / "result"
        # t0 finished before the cutoff; t1 did not.
        _write_trial(result_dir / "jobs", "trial0", "t0", {"pass": 1.0})
        monkeypatch.setattr(runner, "_task_names_for", lambda p: [(0, "t0"), (1, "t1")])

        async def _timeout(*args, **kwargs):
            raise SubprocessTimeoutError(SubprocessResult(
                args=["harbor", "run"], returncode=None, stdout="", stderr="",
                timed_out=True,
            ))

        monkeypatch.setattr("vero.harbor.runner.run_subprocess_with_tee", _timeout)

        ws = MagicMock(project_path="/wt")
        await runner.produce_sample_results(workspace=ws, params=params, result_dir=result_dir)

        results = load_all_sample_results(get_vero_home_dir() / "sessions", "s", params.result_id)
        assert results[0].score == 1.0
        assert results[1].error is not None  # cut-off task -> error sample

    @pytest.mark.asyncio
    async def test_timeout_with_zero_trials_still_raises(self, tmp_path, monkeypatch):
        # Nothing completed before the cutoff: the collate guard must still
        # fail loudly rather than record an all-zero/all-error experiment.
        monkeypatch.setenv("VERO_HOME_DIR", str(tmp_path / "vh"))
        runner = _runner()
        params = _params()
        monkeypatch.setattr(runner, "_task_names_for", lambda p: [(0, "t0"), (1, "t1")])

        async def _timeout(*args, **kwargs):
            raise SubprocessTimeoutError(SubprocessResult(
                args=["harbor", "run"], returncode=None, stdout="", stderr="",
                timed_out=True,
            ))

        monkeypatch.setattr("vero.harbor.runner.run_subprocess_with_tee", _timeout)
        ws = MagicMock(project_path="/wt")
        with pytest.raises(RuntimeError, match="no trial results"):
            await runner.produce_sample_results(
                workspace=ws, params=params, result_dir=tmp_path / "result"
            )

    def test_partial_k_mean_warns(self, tmp_path, caplog):
        # 2 of 3 configured attempts scored: the mean must not shrink k silently.
        runner = HarborRunner(HarborConfig(
            task_source="org/ds", agent_import_path="p:m",
            n_attempts=3, aggregate_attempts="mean",
        ))
        jobs = tmp_path / "jobs"; run = jobs / "2026-01-01__00-00-00"
        w = TestMeanAttemptAggregation()
        w._write(run, "t0a", "t0", rewards={"reward": 1.0})
        w._write(run, "t0b", "t0", rewards={"reward": 0.0})
        groups = runner._trial_groups(jobs)
        with caplog.at_level("WARNING", logger="vero.harbor.runner"):
            r = runner._sample_result(groups["t0"][0], 0, "t0", _params(), attempts=groups["t0"])
        assert r.metrics["n_scored"] == 2.0
        assert any(
            "2 attempt(s) of 3 configured (2 scored, 0 dead" in m
            for m in caplog.messages
        )


def _fb_runner(**kwargs):
    return HarborRunner(
        HarborConfig(task_source="org/ds@1", agent_import_path="pkg.mod:Agent"),
        feedback_transcripts=True,
        **kwargs,
    )


class TestTranscriptFeedback:
    """Lever 1 (feedback_transcripts): a FAILED sample (reward 0) carries the
    tail of its trial transcript in SampleResult.feedback. Population rules are
    tested here; the hidden-split gate (per-sample files are viewable-only) is
    the sidecar's and is covered in test_harbor_server."""

    def _result(self, runner, jobs, task="t0"):
        trials = runner._load_trials(jobs)
        groups = runner._trial_groups(jobs)
        return runner._sample_result(
            trials.get(task), 0, task, _params(), attempts=groups.get(task)
        )

    @pytest.mark.asyncio
    async def test_failed_carries_pane_tail_passed_does_not(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VERO_HOME_DIR", str(tmp_path / "vh"))
        runner = _fb_runner()
        params = _params()
        result_dir = tmp_path / "result"
        _write_trial(result_dir / "jobs", "trial0", "t0", {"reward": 0.0}, pane="failing tail")
        _write_trial(result_dir / "jobs", "trial1", "t1", {"reward": 1.0}, pane="passing tail")
        monkeypatch.setattr(runner, "_task_names_for", lambda p: [(0, "t0"), (1, "t1")])
        runner._run_harbor = AsyncMock()
        ws = MagicMock(project_path="/wt")
        await runner.produce_sample_results(workspace=ws, params=params, result_dir=result_dir)
        results = load_all_sample_results(get_vero_home_dir() / "sessions", "s", params.result_id)
        assert results[0].score == 0.0
        assert results[0].feedback == "failing tail"
        assert results[1].score == 1.0
        assert results[1].feedback is None  # passed samples carry no feedback

    def test_flag_off_leaves_feedback_unset(self, tmp_path):
        runner = _runner()  # default: feedback_transcripts=False
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "trial0", "t0", {"reward": 0.0}, pane="failing tail")
        assert self._result(runner, jobs).feedback is None

    def test_byte_cap_keeps_last_bytes_only(self, tmp_path):
        runner = _fb_runner(feedback_max_bytes=16)
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "trial0", "t0", {"reward": 0.0}, pane="A" * 100 + "TAIL-OF-THE-PANE")
        r = self._result(runner, jobs)
        assert r.feedback == "TAIL-OF-THE-PANE"
        assert len(r.feedback.encode()) <= 16

    def test_falls_back_to_trajectory_when_pane_missing(self, tmp_path):
        runner = _fb_runner()
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "trial0", "t0", {"reward": 0.0}, trajectory='{"steps": []}')
        assert self._result(runner, jobs).feedback == '{"steps": []}'

    def test_missing_transcripts_omitted_silently(self, tmp_path):
        runner = _fb_runner()
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "trial0", "t0", {"reward": 0.0})  # no pane, no trajectory
        r = self._result(runner, jobs)
        assert r.score == 0.0
        assert r.feedback is None

    def test_first_failed_attempt_transcript_used(self, tmp_path):
        # Two failed attempts: the FIRST one's transcript (by finished_at) is
        # attached, deterministically, regardless of rglob order.
        runner = HarborRunner(
            HarborConfig(
                task_source="org/ds@1", agent_import_path="pkg.mod:Agent",
                n_attempts=2, aggregate_attempts="mean",
            ),
            feedback_transcripts=True,
        )
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "zz-early", "t0", {"reward": 0.0}, pane="first attempt",
                     finished_at="2026-01-01T00:01:00")
        _write_trial(jobs, "aa-late", "t0", {"reward": 0.0}, pane="second attempt",
                     finished_at="2026-01-01T00:09:00")
        r = self._result(runner, jobs)
        assert r.score == 0.0
        assert r.feedback == "first attempt"

    def test_partially_passing_mean_sample_gets_no_feedback(self, tmp_path):
        # Failed means reward 0; a mean of [1.0, 0.0] is not a failed sample.
        runner = HarborRunner(
            HarborConfig(
                task_source="org/ds@1", agent_import_path="pkg.mod:Agent",
                n_attempts=2, aggregate_attempts="mean",
            ),
            feedback_transcripts=True,
        )
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "a", "t0", {"reward": 1.0}, pane="p1",
                     finished_at="2026-01-01T00:01:00")
        _write_trial(jobs, "b", "t0", {"reward": 0.0}, pane="p2",
                     finished_at="2026-01-01T00:02:00")
        r = self._result(runner, jobs)
        assert r.score == 0.5
        assert r.feedback is None


class TestTranscriptFeedbackEdgeCases:
    """Byte-cap boundary + feedback_max_bytes<=0 + path-confinement + next-attempt
    fallback. The tail must be exactly capped (never over), never unbounded when
    the cap is 0, must not crash on a multibyte char straddling the boundary, must
    refuse a symlinked / escaping transcript, and must try the next failed attempt
    when the first has no transcript."""

    def _result(self, runner, jobs, task="t0"):
        trials = runner._load_trials(jobs)
        groups = runner._trial_groups(jobs)
        return runner._sample_result(
            trials.get(task), 0, task, _params(), attempts=groups.get(task)
        )

    def test_exact_length_at_cap_returns_full(self, tmp_path):
        # A transcript exactly cap bytes long is returned whole (not truncated).
        pane = "B" * 16
        runner = _fb_runner(feedback_max_bytes=16)
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "trial0", "t0", {"reward": 0.0}, pane=pane)
        r = self._result(runner, jobs)
        assert r.feedback == pane
        assert len(r.feedback.encode()) == 16

    def test_one_byte_over_cap_truncates_to_cap(self, tmp_path):
        # 17 bytes with a 16-byte cap keeps only the last 16 bytes.
        runner = _fb_runner(feedback_max_bytes=16)
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "trial0", "t0", {"reward": 0.0}, pane="X" + "Y" * 16)
        r = self._result(runner, jobs)
        assert r.feedback == "Y" * 16
        assert len(r.feedback.encode()) == 16

    def test_multibyte_char_straddling_cap_does_not_crash(self, tmp_path):
        # A 3-byte U+2603 (snowman) straddles the cap boundary. The slice cuts
        # mid-character; errors="replace" must render it without crashing.
        runner = _fb_runner(feedback_max_bytes=4)
        jobs = tmp_path / "jobs"
        # 6 bytes: 'AAA' + a 3-byte char -> last 4 bytes cut the char mid-sequence
        _write_trial(jobs, "trial0", "t0", {"reward": 0.0}, pane="AAA☃")
        r = self._result(runner, jobs)  # must not raise
        assert r.feedback is not None
        assert len(r.feedback.encode()) <= 8  # replacement chars may re-expand slightly

    def test_zero_cap_emits_no_feedback(self, tmp_path):
        # feedback_max_bytes=0 means "no feedback", NOT the whole transcript.
        runner = _fb_runner(feedback_max_bytes=0)
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "trial0", "t0", {"reward": 0.0}, pane="should not leak")
        r = self._result(runner, jobs)
        assert r.feedback is None

    def test_negative_cap_emits_no_feedback(self, tmp_path):
        runner = _fb_runner(feedback_max_bytes=-5)
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "trial0", "t0", {"reward": 0.0}, pane="should not leak")
        r = self._result(runner, jobs)
        assert r.feedback is None

    def test_symlinked_transcript_is_refused(self, tmp_path):
        # A symlinked transcript file must be skipped silently (field omitted).
        runner = _fb_runner()
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "trial0", "t0", {"reward": 0.0})  # no real transcript
        # place a secret outside the trial dir and symlink the pane path to it
        secret = tmp_path / "secret.txt"
        secret.write_text("SECRET-OUTSIDE")
        trial_dir = jobs / "2026-01-01__00-00-00" / "trial0"
        (trial_dir / "agent").mkdir(exist_ok=True)
        (trial_dir / "agent" / "terminus_2.pane").symlink_to(secret)
        r = self._result(runner, jobs)
        assert r.feedback is None  # symlink refused, nothing leaked

    def test_escaping_transcript_is_refused(self, tmp_path):
        # A trajectory that is a symlink to a file outside the trial dir is also
        # refused (path resolves outside trial_root).
        runner = _fb_runner()
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "trial0", "t0", {"reward": 0.0})
        outside = tmp_path / "outside.json"
        outside.write_text('{"leak": true}')
        trial_dir = jobs / "2026-01-01__00-00-00" / "trial0"
        (trial_dir / "agent").mkdir(exist_ok=True)
        (trial_dir / "agent" / "trajectory.json").symlink_to(outside)
        r = self._result(runner, jobs)
        assert r.feedback is None

    def test_next_failed_attempt_used_when_first_has_no_transcript(self, tmp_path):
        # First failed attempt records no transcript; the second failed attempt's
        # transcript is used instead of giving up.
        runner = HarborRunner(
            HarborConfig(
                task_source="org/ds@1", agent_import_path="pkg.mod:Agent",
                n_attempts=2, aggregate_attempts="mean",
            ),
            feedback_transcripts=True,
        )
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "zz-early", "t0", {"reward": 0.0},  # no pane/trajectory
                     finished_at="2026-01-01T00:01:00")
        _write_trial(jobs, "aa-late", "t0", {"reward": 0.0}, pane="second tail",
                     finished_at="2026-01-01T00:09:00")
        r = self._result(runner, jobs)
        assert r.score == 0.0
        assert r.feedback == "second tail"


class TestAttemptSortOrder:
    """Attempts missing finished_at must sort LAST, not first: an empty-string
    timestamp would sort ahead of every real ISO timestamp and mislabel a
    timestamp-less attempt as the "first" (which feedback keys off)."""

    def test_missing_finished_at_sorts_last(self, tmp_path):
        runner = _runner()
        jobs = tmp_path / "jobs"
        # one attempt with a real timestamp, one with none
        _write_trial(jobs, "with-ts", "t0", {"reward": 0.0},
                     finished_at="2026-01-01T00:05:00")
        _write_trial(jobs, "no-ts", "t0", {"reward": 0.0})  # finished_at absent
        groups = runner._trial_groups(jobs)
        names = [a.get("trial_name") for a in groups["t0"]]
        assert names == ["with-ts", "no-ts"]  # timestamped first, missing last

    def test_feedback_uses_timestamped_attempt_over_timeless_one(self, tmp_path):
        runner = _fb_runner()
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "no-ts", "t0", {"reward": 0.0}, pane="timeless tail")
        _write_trial(jobs, "with-ts", "t0", {"reward": 0.0}, pane="timestamped tail",
                     finished_at="2026-01-01T00:05:00")
        trials = runner._load_trials(jobs)
        groups = runner._trial_groups(jobs)
        r = runner._sample_result(trials.get("t0"), 0, "t0", _params(), attempts=groups.get("t0"))
        assert r.feedback == "timestamped tail"


class TestAttemptDetail:
    """Lever 3 (expose_attempt_detail): sample output carries an `attempts`
    list, one {reward, exception} entry per attempt. Population rules here;
    the viewable-only exposure gate is the sidecar's (test_harbor_server)."""

    def _result(self, runner, jobs, task="t0"):
        trials = runner._load_trials(jobs)
        groups = runner._trial_groups(jobs)
        return runner._sample_result(
            trials.get(task), 0, task, _params(), attempts=groups.get(task)
        )

    def test_one_entry_per_attempt_with_exception_names(self, tmp_path):
        runner = HarborRunner(
            HarborConfig(
                task_source="org/ds@1", agent_import_path="pkg.mod:Agent",
                n_attempts=3, aggregate_attempts="mean",
            ),
            expose_attempt_detail=True,
        )
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "a", "t0", {"reward": 1.0}, finished_at="2026-01-01T00:01:00")
        _write_trial(jobs, "b", "t0", {"reward": 0.0}, finished_at="2026-01-01T00:02:00",
                     exception_type="AgentTimeoutError")
        _write_trial(jobs, "c", "t0", None, finished_at="2026-01-01T00:03:00",
                     exception_type="RuntimeError")
        r = self._result(runner, jobs)
        assert r.output["attempts"] == [
            {"reward": 1.0, "exception": None},
            {"reward": 0.0, "exception": "AgentTimeoutError"},
            {"reward": None, "exception": "RuntimeError"},
        ]

    @pytest.mark.asyncio
    async def test_best_mode_collates_attempts_end_to_end(self, tmp_path, monkeypatch):
        # 'best' aggregation does not need the attempt groups for scoring, so
        # this pins that _collate still loads them when the lever asks for it.
        monkeypatch.setenv("VERO_HOME_DIR", str(tmp_path / "vh"))
        runner = HarborRunner(
            HarborConfig(task_source="org/ds@1", agent_import_path="pkg.mod:Agent"),
            expose_attempt_detail=True,
        )
        params = _params()
        result_dir = tmp_path / "result"
        _write_trial(result_dir / "jobs", "trial0", "t0", {"reward": 1.0})
        monkeypatch.setattr(runner, "_task_names_for", lambda p: [(0, "t0")])
        runner._run_harbor = AsyncMock()
        ws = MagicMock(project_path="/wt")
        await runner.produce_sample_results(workspace=ws, params=params, result_dir=result_dir)
        results = load_all_sample_results(get_vero_home_dir() / "sessions", "s", params.result_id)
        assert results[0].score == 1.0  # best-mode scoring untouched
        assert results[0].output["attempts"] == [{"reward": 1.0, "exception": None}]

    def test_flag_off_leaves_output_without_attempts(self, tmp_path):
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "a", "t0", {"reward": 1.0}, finished_at="2026-01-01T00:01:00")
        _write_trial(jobs, "b", "t0", {"reward": 0.0}, finished_at="2026-01-01T00:02:00")
        best = self._result(_runner(), jobs)
        assert "attempts" not in best.output
        mean_runner = HarborRunner(HarborConfig(
            task_source="org/ds@1", agent_import_path="pkg.mod:Agent",
            n_attempts=2, aggregate_attempts="mean",
        ))
        mean = self._result(mean_runner, jobs)
        assert "attempts" not in mean.output
