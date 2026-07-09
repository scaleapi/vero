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
    exception_message: str = "",
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
            "exception_message": exception_message,
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

    def test_harbor_requirement_layers_trusted_cli(self):
        # The overlay must come before the `harbor` executable name so uv
        # resolves the CLI from the trusted spec, not the candidate's lockfile.
        runner = HarborRunner(
            HarborConfig(
                task_source="org/ds@1",
                agent_import_path="pkg.mod:Agent",
                harbor_requirement="harbor==0.1.17",
            )
        )
        cmd = runner._build_command("/wt", _params(), ["t0"], Path("/jobs"))
        assert cmd[:6] == ["uv", "run", "--project", "/wt", "--with", "harbor==0.1.17"]
        assert cmd[6] == "harbor"

    def test_no_harbor_requirement_keeps_candidate_env(self):
        cmd = _runner()._build_command("/wt", _params(), ["t0"], Path("/jobs"))
        assert "--with" not in cmd

    def test_model_override_beats_configured_model(self):
        # Transfer targets: a per-eval executor override (via task_params)
        # wins over the task's configured model.
        params = _params()
        params.task_params = {"harbor_model_override": "openai/gpt-4o"}
        cmd = _runner()._build_command("/wt", params, ["t0"], Path("/jobs"))
        m = cmd[cmd.index("-m") + 1]
        assert m == "openai/gpt-4o"

    def test_no_override_uses_configured_model(self):
        cmd = _runner()._build_command("/wt", _params(), ["t0"], Path("/jobs"))
        assert cmd[cmd.index("-m") + 1] == "anthropic/x"


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

    def test_mean_records_dead_exception_types(self, tmp_path):
        # n_dead alone hides WHY attempts died, and cause matters: rate-limit
        # deaths are infra noise, crashes point at the candidate, and deaths
        # cluster hard by cause (E1: 110/129 UnicodeDecodeErrors sat on two
        # tasks). Every zero-filled attempt gets its exception type counted.
        runner = HarborRunner(HarborConfig(
            task_source="org/ds", agent_import_path="p:m",
            n_attempts=3, aggregate_attempts="mean",
        ))
        jobs = tmp_path / "jobs"; run = jobs / "2026-01-01__00-00-00"
        self._write(run, "t0a", "t0", rewards={"reward": 1.0})
        self._write(run, "t0bad", "t0", exc=True)  # exception_type "X"
        self._write(run, "t0gone", "t0")  # no rewards AND no exception recorded
        groups = runner._trial_groups(jobs)
        r = runner._sample_result(groups["t0"][0], 0, "t0", _params(), attempts=groups["t0"])
        assert r.output["dead_exception_types"] == {"X": 1, "no_rewards_recorded": 1}
        # all-clean samples carry no key at all
        self._write(run, "t1a", "t1", rewards={"reward": 1.0})
        groups = runner._trial_groups(jobs)
        r1 = runner._sample_result(groups["t1"][0], 1, "t1", _params(), attempts=groups["t1"])
        assert "dead_exception_types" not in r1.output

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

    def test_empty_pane_falls_through_to_trajectory(self, tmp_path):
        # An empty transcript file carries nothing and must not surface as ""
        # feedback; the empty pane falls through to the trajectory.
        runner = _fb_runner()
        jobs = tmp_path / "jobs"
        _write_trial(
            jobs, "trial0", "t0", {"reward": 0.0}, pane="", trajectory='{"steps": [1]}'
        )
        assert self._result(runner, jobs).feedback == '{"steps": [1]}'

    def test_all_empty_transcripts_yield_no_feedback(self, tmp_path):
        runner = _fb_runner()
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "trial0", "t0", {"reward": 0.0}, pane="", trajectory="")
        assert self._result(runner, jobs).feedback is None

    def test_no_rewards_error_sample_carries_crash_transcript(self, tmp_path):
        # A candidate edit that crashes the agent before scoring lands in the
        # no-verifier-rewards error branch; the transcript is the only way the
        # optimizer can see the crash it caused.
        runner = _fb_runner()
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "trial0", "t0", None, pane="crash tail")
        r = self._result(runner, jobs)
        assert r.error is not None
        assert r.feedback == "crash tail"

    def test_no_rewards_error_names_dead_exception_types(self, tmp_path):
        # The error string must carry WHY the attempts died: it is the one
        # field that flows to the DB, the per-sample files, and the verifier's
        # target_errors, and it separates a deterministic candidate crash
        # (measured live: 72/72 UnsupportedParamsError on an off-model
        # executor) from an infra outage.
        runner = _fb_runner()
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "trial0", "t0", None, exception_type="UnsupportedParamsError")
        _write_trial(jobs, "trial1", "t0", None, exception_type="UnsupportedParamsError")
        r = self._result(runner, jobs)
        assert r.error is not None
        assert "UnsupportedParamsError x2" in r.error

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


class TestMeanRewardKeyMismatch:
    def test_mean_zero_fills_reward_key_mismatch(self, tmp_path):
        # An attempt whose rewards LACK the configured key is unscorable on the
        # configured metric and counts 0.0 in the mean (n_dead), exactly like
        # dying pre-verifier: falling back to another key would score attempts
        # within one mean on different metrics.
        runner = HarborRunner(HarborConfig(
            task_source="org/ds", agent_import_path="p:m",
            n_attempts=2, aggregate_attempts="mean", reward_key="acc",
        ))
        jobs = tmp_path / "jobs"; run = jobs / "2026-01-01__00-00-00"
        w = TestMeanAttemptAggregation()
        w._write(run, "t0a", "t0", rewards={"acc": 1.0})
        w._write(run, "t0b", "t0", rewards={"other": 1.0})
        groups = runner._trial_groups(jobs)
        r = runner._sample_result(groups["t0"][0], 0, "t0", _params(), attempts=groups["t0"])
        assert r.score == 0.5
        assert r.metrics["n_dead"] == 1.0
        assert r.metrics["n_scored"] == 1.0


class TestInfraResilience:
    """Dead-attempt classification + bounded within-eval infra retry.

    Classification is diagnostic and retry-gating only: every dead attempt
    still scores 0.0 regardless of class (excusing infra-labeled deaths from
    the score would let a candidate raise fake ConnectionErrors on hard tasks).
    The retry is OFF BY DEFAULT and exists for trusted-candidate evaluations:
    against an adversarial optimizer it is a re-roll lever, since the
    qualifying predicate derives from exceptions raised in candidate code. It
    re-measures only samples whose EVERY attempt died of a transient infra
    cause, in a fresh sibling jobs dir, and stamps an audit marker on
    recovered samples so the discarded round stays visible.
    """

    def _flow_runner(self, monkeypatch, tmp_path, rounds_by_call, **cfg):
        """A runner whose _run_harbor writes fixture trials per call:
        rounds_by_call[i] is a list of (trial, task, rewards, exc_type, exc_msg)
        written into whatever jobs dir call i receives."""
        monkeypatch.setenv("VERO_HOME_DIR", str(tmp_path / "vh"))
        runner = HarborRunner(HarborConfig(
            task_source="org/ds@1", agent_import_path="pkg.mod:Agent",
            **cfg,
        ))
        calls = []

        async def _fake_run(project_path, params, task_names, jobs_dir):
            i = len(calls)
            calls.append((list(task_names), Path(jobs_dir)))
            for trial, task, rewards, exc_type, exc_msg in rounds_by_call[i]:
                _write_trial(Path(jobs_dir), trial, task, rewards,
                             exception_type=exc_type, exception_message=exc_msg)

        monkeypatch.setattr(runner, "_run_harbor", _fake_run)
        monkeypatch.setattr(runner, "_task_names_for", lambda p: [(0, "t0"), (1, "t1")])
        sleeps = []
        import asyncio as _asyncio
        real_sleep = _asyncio.sleep

        async def _fast_sleep(d, *a, **k):
            sleeps.append(d)
            await real_sleep(0)

        monkeypatch.setattr("asyncio.sleep", _fast_sleep)
        return runner, calls, sleeps

    def test_config_rejects_zero_delay_when_retries_enabled(self):
        # A zero delay silently nullifies the backoff and an instant retry
        # re-enters the same outage; misconfiguration must fail loudly.
        with pytest.raises(ValueError, match="infra_retry_delay_s"):
            HarborConfig(task_source="org/ds@1", agent_import_path="p:m",
                         infra_retry_rounds=1, infra_retry_delay_s=0.0)
        # with retries off the delay is never used, so 0 is not an error
        HarborConfig(task_source="org/ds@1", agent_import_path="p:m",
                     infra_retry_rounds=0, infra_retry_delay_s=0.0)

    def test_config_rejects_negative_rounds(self):
        with pytest.raises(ValueError, match="infra_retry_rounds"):
            HarborConfig(task_source="org/ds@1", agent_import_path="p:m",
                         infra_retry_rounds=-1)

    def test_infra_deaths_labeled_and_counted_but_still_zero_filled(self, tmp_path):
        # One scored attempt, one infra death, one candidate crash: the mean
        # zero-fills BOTH deaths (classification must never move the score),
        # and the labels + n_dead_infra let an analyst see which zeros
        # measured the plumbing.
        runner = HarborRunner(HarborConfig(
            task_source="org/ds@1", agent_import_path="pkg.mod:Agent",
            n_attempts=3, aggregate_attempts="mean",
        ))
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "a", "t0", {"reward": 1.0})
        _write_trial(jobs, "b", "t0", None, exception_type="ConnectionError")
        _write_trial(jobs, "c", "t0", None, exception_type="UnsupportedParamsError")
        groups = runner._trial_groups(jobs)
        r = runner._sample_result(groups["t0"][0], 0, "t0", _params(), attempts=groups["t0"])
        assert r.score == pytest.approx(1.0 / 3)
        assert r.output["dead_exception_types"] == {
            "ConnectionError[infra]": 1,
            "UnsupportedParamsError": 1,
        }
        assert r.metrics["n_dead"] == 2.0
        assert r.metrics["n_dead_infra"] == 1.0

    def test_forged_infra_suffix_is_neutralized(self, tmp_path):
        # The suffix contract is load-bearing and the exception type name is
        # candidate-authored: a class literally named "XError[infra]" must not
        # classify as infra. Brackets are neutralized before labeling.
        runner = HarborRunner(HarborConfig(
            task_source="org/ds@1", agent_import_path="pkg.mod:Agent",
            n_attempts=2, aggregate_attempts="mean",
        ))
        jobs = tmp_path / "jobs"
        _write_trial(jobs, "a", "t0", {"reward": 1.0})
        _write_trial(jobs, "b", "t0", None, exception_type="MadeUpError[infra]")
        groups = runner._trial_groups(jobs)
        r = runner._sample_result(groups["t0"][0], 0, "t0", _params(), attempts=groups["t0"])
        assert r.output["dead_exception_types"] == {"MadeUpError(infra)": 1}
        assert r.metrics["n_dead_infra"] == 0.0

    def test_key_budget_exhaustion_labeled_and_alarmed(self, tmp_path, caplog):
        # litellm reports a spent key budget as a BadRequestError; only the
        # message identifies it. It must be labeled infra (those zeros likely
        # measure an outage) and alarmed at ERROR (every later call fails
        # identically). The alarm text hedges: the signature is read from
        # candidate-process exceptions and needs corroboration.
        import logging as _logging

        runner = HarborRunner(HarborConfig(
            task_source="org/ds@1", agent_import_path="pkg.mod:Agent",
            n_attempts=2, aggregate_attempts="mean",
        ))
        jobs = tmp_path / "jobs"
        msg = "litellm.BadRequestError: Budget has been exceeded! Current cost: 102.47, Max budget: 99.0"
        _write_trial(jobs, "a", "t0", None, exception_type="BadRequestError", exception_message=msg)
        _write_trial(jobs, "b", "t0", None, exception_type="BadRequestError", exception_message=msg)
        groups = runner._trial_groups(jobs)
        with caplog.at_level(_logging.ERROR, logger="vero.harbor.runner"):
            r = runner._sample_result(groups["t0"][0], 0, "t0", _params(), attempts=groups["t0"])
        assert r.error is not None
        assert "BadRequestError[infra:llm-key-budget] x2" in r.error
        assert r.output["dead_exception_types"] == {"BadRequestError[infra:llm-key-budget]": 2}
        assert any("spend budget as exhausted" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_retry_is_off_by_default(self, tmp_path, monkeypatch):
        # The re-roll lever must be opt-in: with a default config, an
        # infra-outaged sample books its cause-rich error and nothing re-runs.
        runner, calls, sleeps = self._flow_runner(
            monkeypatch, tmp_path,
            rounds_by_call=[
                [("a", "t0", {"reward": 1.0}, None, ""),
                 ("b", "t1", None, "ConnectionError", "")],
            ],
        )
        assert runner.config.infra_retry_rounds == 0
        params = _params()
        await runner.produce_sample_results(
            workspace=MagicMock(project_path="/wt"), params=params, result_dir=tmp_path / "res"
        )
        assert len(calls) == 1
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_retry_reruns_only_the_outaged_sample(self, tmp_path, monkeypatch):
        # t0 scores; t1 loses every attempt to ConnectionError. The retry round
        # re-runs t1 ALONE in a fresh SIBLING jobs dir after a backoff, the
        # fresh measurement replaces the recorded outage, and the booked output
        # carries the audit marker naming the discarded dead attempts.
        runner, calls, sleeps = self._flow_runner(
            monkeypatch, tmp_path,
            rounds_by_call=[
                [("a", "t0", {"reward": 1.0}, None, ""),
                 ("b", "t1", None, "ConnectionError", ""),
                 ("c", "t1", None, "ConnectionError", "")],
                [("r1", "t1", {"reward": 0.5}, None, "")],
            ],
            infra_retry_rounds=1,
        )
        params = _params()
        await runner.produce_sample_results(
            workspace=MagicMock(project_path="/wt"), params=params, result_dir=tmp_path / "res"
        )
        results = load_all_sample_results(get_vero_home_dir() / "sessions", "s", params.result_id)
        assert results[0].score == 1.0
        assert results[1].error is None and results[1].score == 0.5
        assert results[1].output["infra_retry"] == {
            "recovered_round": 1,
            "discarded_rounds": [{"ConnectionError[infra]": 2}],
        }
        assert "infra_retry" not in (results[0].output or {})
        assert len(calls) == 2
        assert calls[1][0] == ["t1"]
        # sibling of jobs/, never nested inside it (resume collations rglob
        # the whole jobs dir and would pool this round's dead attempts)
        assert calls[1][1].name == "jobs-infra-retry-1"
        assert calls[1][1].parent == calls[0][1].parent
        assert sleeps == [30.0]

    @pytest.mark.asyncio
    async def test_candidate_crash_and_spent_key_never_retry(self, tmp_path, monkeypatch):
        # A crash is a result; a spent key cannot recover by waiting. Neither
        # triggers a retry round even with retries enabled.
        budget_msg = "Budget has been exceeded! Current cost: 102.47, Max budget: 99.0"
        runner, calls, sleeps = self._flow_runner(
            monkeypatch, tmp_path,
            rounds_by_call=[
                [("a", "t0", None, "UnsupportedParamsError", ""),
                 ("b", "t1", None, "BadRequestError", budget_msg)],
            ],
            infra_retry_rounds=1,
        )
        params = _params()
        await runner.produce_sample_results(
            workspace=MagicMock(project_path="/wt"), params=params, result_dir=tmp_path / "res"
        )
        results = load_all_sample_results(get_vero_home_dir() / "sessions", "s", params.result_id)
        assert results[0].error is not None and results[1].error is not None
        assert len(calls) == 1
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_mixed_cause_fully_dead_sample_never_retries(self, tmp_path, monkeypatch):
        # EVERY dead cause must be transient-infra (all, not any): a
        # deterministically-crashing candidate must not qualify for a re-roll
        # by mixing one fake ConnectionError into its crashes.
        runner, calls, sleeps = self._flow_runner(
            monkeypatch, tmp_path,
            rounds_by_call=[
                [("a", "t0", {"reward": 1.0}, None, ""),
                 ("b", "t1", None, "ConnectionError", ""),
                 ("c", "t1", None, "UnsupportedParamsError", "")],
            ],
            n_attempts=2, aggregate_attempts="mean", infra_retry_rounds=1,
        )
        params = _params()
        await runner.produce_sample_results(
            workspace=MagicMock(project_path="/wt"), params=params, result_dir=tmp_path / "res"
        )
        results = load_all_sample_results(get_vero_home_dir() / "sessions", "s", params.result_id)
        assert results[1].error is not None
        assert len(calls) == 1
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_persistent_outage_keeps_the_cause_rich_error(self, tmp_path, monkeypatch):
        # The retry round produces nothing (outage persists): the durable
        # record must keep the original infra-labeled error, not degrade to
        # "no Harbor trial result" or raise out of the eval.
        runner, calls, sleeps = self._flow_runner(
            monkeypatch, tmp_path,
            rounds_by_call=[
                [("a", "t0", {"reward": 1.0}, None, ""),
                 ("b", "t1", None, "ConnectionError", "")],
                [],
            ],
            infra_retry_rounds=1,
        )
        params = _params()
        await runner.produce_sample_results(
            workspace=MagicMock(project_path="/wt"), params=params, result_dir=tmp_path / "res"
        )
        results = load_all_sample_results(get_vero_home_dir() / "sessions", "s", params.result_id)
        assert results[1].error is not None
        assert "ConnectionError[infra]" in results[1].error
        assert results[1].output["dead_exception_types"] == {"ConnectionError[infra]": 1}
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_partially_scored_sample_is_a_measurement_not_an_outage(self, tmp_path, monkeypatch):
        # One attempt scored, one died of infra: the sample is a (noisy)
        # measurement. Its infra death stays zero-filled in the mean and it is
        # NOT retried; anything else would let infra flakiness re-roll scores.
        runner, calls, sleeps = self._flow_runner(
            monkeypatch, tmp_path,
            rounds_by_call=[
                [("a", "t0", {"reward": 1.0}, None, ""),
                 ("b", "t0", None, "ConnectionError", ""),
                 ("c", "t1", {"reward": 1.0}, None, "")],
            ],
            n_attempts=2, aggregate_attempts="mean", infra_retry_rounds=1,
        )
        params = _params()
        await runner.produce_sample_results(
            workspace=MagicMock(project_path="/wt"), params=params, result_dir=tmp_path / "res"
        )
        results = load_all_sample_results(get_vero_home_dir() / "sessions", "s", params.result_id)
        assert results[0].score == 0.5
        assert results[0].metrics["n_dead_infra"] == 1.0
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_retry_round_mean_is_not_diluted_by_dead_rounds(self, tmp_path, monkeypatch):
        # The fresh round is collated from its own sibling dir alone: the two
        # dead attempts of round 0 must not zero-dilute the fresh mean
        # (1.0, not 0.5), and the audit marker records what was discarded.
        runner, calls, sleeps = self._flow_runner(
            monkeypatch, tmp_path,
            rounds_by_call=[
                [("a", "t0", {"reward": 1.0}, None, ""),
                 ("b", "t1", None, "ConnectionError", ""),
                 ("c", "t1", None, "ConnectionError", "")],
                [("r1", "t1", {"reward": 1.0}, None, ""),
                 ("r2", "t1", {"reward": 1.0}, None, "")],
            ],
            n_attempts=2, aggregate_attempts="mean", infra_retry_rounds=1,
        )
        params = _params()
        await runner.produce_sample_results(
            workspace=MagicMock(project_path="/wt"), params=params, result_dir=tmp_path / "res"
        )
        results = load_all_sample_results(get_vero_home_dir() / "sessions", "s", params.result_id)
        assert results[1].score == 1.0
        assert results[1].metrics["n_scored"] == 2.0
        assert results[1].metrics["n_dead"] == 0.0
        assert results[1].output["infra_retry"]["discarded_rounds"] == [
            {"ConnectionError[infra]": 2}
        ]

    @pytest.mark.asyncio
    async def test_multi_round_backoff_and_partial_recovery(self, tmp_path, monkeypatch):
        # Two rounds: round 1 recovers t0 and leaves t1 outaged; round 2
        # retries ONLY the survivor, after a LONGER (linear) backoff, in its
        # own fresh sibling dir.
        runner, calls, sleeps = self._flow_runner(
            monkeypatch, tmp_path,
            rounds_by_call=[
                [("a", "t0", None, "ConnectionError", ""),
                 ("b", "t1", None, "TimeoutError", "")],
                [("r1a", "t0", {"reward": 1.0}, None, ""),
                 ("r1b", "t1", None, "TimeoutError", "")],
                [("r2b", "t1", {"reward": 0.5}, None, "")],
            ],
            infra_retry_rounds=2,
        )
        params = _params()
        await runner.produce_sample_results(
            workspace=MagicMock(project_path="/wt"), params=params, result_dir=tmp_path / "res"
        )
        results = load_all_sample_results(get_vero_home_dir() / "sessions", "s", params.result_id)
        assert results[0].score == 1.0
        assert results[0].output["infra_retry"] == {
            "recovered_round": 1,
            "discarded_rounds": [{"ConnectionError[infra]": 1}],
        }
        # t1 burned TWO rounds before recovering; the audit marker must list
        # both, not just the round immediately before recovery.
        assert results[1].score == 0.5
        assert results[1].output["infra_retry"] == {
            "recovered_round": 2,
            "discarded_rounds": [{"TimeoutError[infra]": 1}, {"TimeoutError[infra]": 1}],
        }
        assert [c[0] for c in calls] == [["t0", "t1"], ["t0", "t1"], ["t1"]]
        assert [c[1].name for c in calls[1:]] == ["jobs-infra-retry-1", "jobs-infra-retry-2"]
        assert sleeps == [30.0, 60.0]
