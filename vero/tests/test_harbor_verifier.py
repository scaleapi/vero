"""Tests for vero.harbor.verifier.Verifier — selection + multi-target scoring."""

import json
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from vero.harbor.verifier import NoCandidateError, VerificationTarget, Verifier


def _engine(scores_by_call):
    engine = MagicMock()
    engine.evaluate_admin = AsyncMock(
        side_effect=[MagicMock(result=MagicMock(score=MagicMock(return_value=s))) for s in scores_by_call]
    )
    return engine


class TestSubmitSelection:
    @pytest.mark.asyncio
    async def test_finalize_submit_scores_nominated_commit(self, tmp_path):
        (tmp_path / "submission.json").write_text(json.dumps({"commit": "deadbeef"}))
        engine = _engine([0.8])
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="submit",
            targets=[VerificationTarget(task="t", dataset_id="ds1", split="test", reward_key="reward")],
        )
        rewards = (await v.finalize())["rewards"]
        assert rewards == {"reward": 0.8}
        assert engine.evaluate_admin.await_args.kwargs["commit"] == "deadbeef"
        assert engine.evaluate_admin.await_args.kwargs["split"] == "test"

    @pytest.mark.asyncio
    async def test_finalize_submit_no_submission_floors_rewards(self, tmp_path):
        # "The agent never submitted" is an outcome, not an infrastructure
        # failure: finalize floors every target instead of raising, so the
        # outer harness records reward 0.0 rather than a missing-reward error.
        engine = _engine([])
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="submit",
            targets=[
                VerificationTarget(task="t", dataset_id="ds1", split="test", reward_key="reward"),
                VerificationTarget(task="t2", dataset_id="ds2", split="test", reward_key="held_out"),
            ],
        )
        rewards = (await v.finalize())["rewards"]
        assert rewards == {"reward": 0.0, "held_out": 0.0}
        engine.evaluate_admin.assert_not_awaited()


class TestMultiTarget:
    @pytest.mark.asyncio
    async def test_finalize_emits_multiple_reward_keys(self, tmp_path):
        (tmp_path / "submission.json").write_text(json.dumps({"commit": "c1"}))
        engine = _engine([0.9, 0.4])
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="submit",
            targets=[
                VerificationTarget(task="t", dataset_id="ds1", split="test", reward_key="in_domain"),
                VerificationTarget(task="t2", dataset_id="ds2", split="test", reward_key="held_out"),
            ],
        )
        rewards = (await v.finalize())["rewards"]
        assert rewards == {"in_domain": 0.9, "held_out": 0.4}
        assert engine.evaluate_admin.await_count == 2


class TestAutoBestSelection:
    @pytest.mark.asyncio
    async def test_auto_best_reranks_by_admin_score(self, tmp_path):
        # 'hi' has the best RECORDED score (agent-influenced) but the admin re-score
        # rates 'lo' higher; selection must follow the admin score.
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = pd.DataFrame(
            {
                "dataset_subset_split": ["validation", "validation", "train"],
                "dataset_subset_dataset_id": ["ds1", "ds1", "ds1"],
                "candidate_commit": ["lo", "hi", "ignored"],
                "mean_score": [0.5, 0.9, 1.0],
                "candidate_created_at": [1, 2, 3],
            }
        )
        admin_scores = {"hi": 0.1, "lo": 0.95}

        async def _admin(*, task, dataset_id, split, commit, sample_ids=None):
            return MagicMock(
                result=MagicMock(score=MagicMock(return_value=admin_scores.get(commit, 0.99)))
            )

        engine.evaluate_admin = AsyncMock(side_effect=_admin)
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="validation",
            selection_task="math",
            targets=[VerificationTarget(task="math", dataset_id="ds1", split="test", reward_key="reward")],
        )
        rewards = (await v.finalize())["rewards"]
        # the final (target) eval is on the WINNER 'lo', chosen by admin re-score
        assert engine.evaluate_admin.await_args.kwargs["commit"] == "lo"
        assert engine.evaluate_admin.await_args.kwargs["split"] == "test"
        # reward is the winner 'lo' scored on the target split -> its admin score
        assert rewards["reward"] == 0.95

    @pytest.mark.asyncio
    async def test_auto_best_excludes_baseline_from_ranking(self, tmp_path):
        # base_commit is excluded from the candidate ranking pool. Floor off here so
        # the test isolates ranking-exclusion (the floor is covered separately below).
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = pd.DataFrame(
            {
                "dataset_subset_split": ["validation", "validation"],
                "dataset_subset_dataset_id": ["ds1", "ds1"],
                "candidate_commit": ["base", "agent"],
                "mean_score": [0.99, 0.6],
                "candidate_created_at": [1, 2],
            }
        )

        async def _admin(*, task, dataset_id, split, commit, sample_ids=None):
            return MagicMock(result=MagicMock(score=MagicMock(return_value=0.7)))

        engine.evaluate_admin = AsyncMock(side_effect=_admin)
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="validation",
            base_commit="base",
            selection_task="math",
            auto_best_baseline_floor=False,
            targets=[VerificationTarget(task="math", dataset_id="ds1", split="test", reward_key="reward")],
        )
        await v.finalize()
        # baseline 'base' was never re-scored; 'agent' is selected + target-scored
        rescored_commits = [c.kwargs["commit"] for c in engine.evaluate_admin.await_args_list]
        assert "base" not in rescored_commits
        assert engine.evaluate_admin.await_args.kwargs["commit"] == "agent"


class TestSubsetEvalShortlistFilter:
    """auto_best ranks the shortlist on FULL-split evals only. A subset eval
    (num_samples / sample_ids) records a mean over a few samples, so a lucky
    1-sample eval can inflate a candidate's recorded score and steal a shortlist
    slot from a genuinely better full-split candidate. Full-split evals have
    dataset_subset_sample_ids = None; subset evals carry a list and are ignored
    for ranking (the admin re-score still runs on the full split)."""

    @pytest.mark.asyncio
    async def test_lucky_subset_eval_does_not_outrank_full_split(self, tmp_path):
        # 'lucky' has a high 1-sample eval (recorded 0.95) but a low full-split
        # eval (0.30). 'solid' has a higher full-split eval (0.70). With
        # rescore_top_k=1 only one commit is shortlisted; ranking on full-split
        # evals must shortlist 'solid', not 'lucky'.
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = pd.DataFrame(
            {
                "dataset_subset_split": ["validation", "validation", "validation"],
                "dataset_subset_dataset_id": ["ds1", "ds1", "ds1"],
                "dataset_subset_sample_ids": [[0], None, None],  # lucky subset, then full splits
                "candidate_commit": ["lucky", "lucky", "solid"],
                "mean_score": [0.95, 0.30, 0.70],
                "candidate_created_at": [3, 1, 2],
            }
        )

        async def _admin(*, task, dataset_id, split, commit, sample_ids=None):
            # admin re-score agrees the full-split ranking is right
            score = {"solid": 0.7, "lucky": 0.3}.get(commit, 0.5)
            return MagicMock(result=MagicMock(score=MagicMock(return_value=score)))

        engine.evaluate_admin = AsyncMock(side_effect=_admin)
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="validation",
            selection_task="math",
            rescore_top_k=1,
            auto_best_baseline_floor=False,
            targets=[VerificationTarget(task="math", dataset_id="ds1", split="test", reward_key="reward")],
        )
        await v.finalize()
        # 'solid' is selected (target-scored); 'lucky' never entered the shortlist
        rescored = [c.kwargs["commit"] for c in engine.evaluate_admin.await_args_list]
        assert "lucky" not in rescored
        assert engine.evaluate_admin.await_args.kwargs["commit"] == "solid"

    @pytest.mark.asyncio
    async def test_all_subset_evals_still_rankable(self, tmp_path):
        # When EVERY eval is a subset, there is no full-split candidate to
        # protect, so the subset evals are the only ranking signal and are kept
        # (a legitimate all-subset workflow must still select a candidate). The
        # admin re-score on the full split remains the trust anchor.
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = pd.DataFrame(
            {
                "dataset_subset_split": ["validation", "validation"],
                "dataset_subset_dataset_id": ["ds1", "ds1"],
                "dataset_subset_sample_ids": [[0], [0, 1]],
                "candidate_commit": ["a", "b"],
                "mean_score": [0.9, 0.8],
                "candidate_created_at": [1, 2],
            }
        )

        async def _admin(*, task, dataset_id, split, commit, sample_ids=None):
            return MagicMock(result=MagicMock(score=MagicMock(return_value=0.5)))

        engine.evaluate_admin = AsyncMock(side_effect=_admin)
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="validation",
            selection_task="math",
            auto_best_baseline_floor=False,
            targets=[VerificationTarget(task="math", dataset_id="ds1", split="test", reward_key="reward")],
        )
        rewards = (await v.finalize())["rewards"]
        # a candidate was selected and target-scored (not floored)
        assert rewards == {"reward": 0.5}
        engine.evaluate_admin.assert_awaited()


class TestAutoBestBaselineFloor:
    """auto_best never ships a candidate that fails to beat the baseline.

    auto_best excludes base_commit from the candidate pool, so without a floor it
    selects the least-bad candidate even when every candidate regressed (observed
    live: a weak inner model, every candidate below baseline, shipped a -0.10
    regression despite the free baseline being available). The floor reverts to the
    seed instead.
    """

    def _df(self):
        return pd.DataFrame(
            {
                "dataset_subset_split": ["train", "train"],
                "dataset_subset_dataset_id": ["ds1", "ds1"],
                "candidate_commit": ["base", "agent"],
                "mean_score": [0.3, 0.9],  # agent inflated its own recorded score
                "candidate_created_at": [1, 2],
            }
        )

    @pytest.mark.asyncio
    async def test_reverts_to_base_when_no_candidate_beats_baseline(self, tmp_path):
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = self._df()

        # agent admin-scores 0.2 on the selection split; base admin-scores 0.3;
        # the reverted base scores 0.35 on the target split (distinct values so the
        # assertions can tell the target eval apart from the floor comparison).
        async def _admin(*, task, dataset_id, split, commit, sample_ids=None):
            if commit == "base":
                score = 0.35 if split == "validation" else 0.3
            else:
                score = 0.2
            return MagicMock(result=MagicMock(score=MagicMock(return_value=score)))

        engine.evaluate_admin = AsyncMock(side_effect=_admin)
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="train",
            base_commit="base",
            selection_task="math",
            targets=[VerificationTarget(task="math", dataset_id="ds1", split="validation", reward_key="reward")],
        )
        result = await v.finalize()
        # winner reverted to base -> the emitted reward is the SEED's target-split
        # score, not the regressed candidate's
        assert result["rewards"] == {"reward": 0.35}
        rescored = [c.kwargs["commit"] for c in engine.evaluate_admin.await_args_list]
        assert "base" in rescored  # base was admin-scored for the floor comparison
        # the final call is the target eval of the reverted commit (validation split),
        # not the floor comparison (train split)
        assert engine.evaluate_admin.await_args.kwargs["commit"] == "base"
        assert engine.evaluate_admin.await_args.kwargs["split"] == "validation"

    @pytest.mark.asyncio
    async def test_exact_tie_reverts_to_base(self, tmp_path):
        # The floor uses '<=': a statistical tie reverts. If the optimizer cannot
        # show an improvement, shipping the seed is the safe outcome. Pins the
        # boundary so a refactor to '<' regresses loudly.
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = self._df()

        async def _admin(*, task, dataset_id, split, commit, sample_ids=None):
            return MagicMock(result=MagicMock(score=MagicMock(return_value=0.3)))  # all equal

        engine.evaluate_admin = AsyncMock(side_effect=_admin)
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="train",
            base_commit="base",
            selection_task="math",
            targets=[VerificationTarget(task="math", dataset_id="ds1", split="validation", reward_key="reward")],
        )
        await v.finalize()
        assert engine.evaluate_admin.await_args.kwargs["commit"] == "base"

    @pytest.mark.asyncio
    async def test_floor_noop_without_base_commit(self, tmp_path):
        # floor on (default) but base_commit=None: the floor must silently no-op,
        # never issuing an eval with commit=None, and the best candidate ships.
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = pd.DataFrame(
            {
                "dataset_subset_split": ["train"],
                "dataset_subset_dataset_id": ["ds1"],
                "candidate_commit": ["agent"],
                "mean_score": [0.9],
                "candidate_created_at": [1],
            }
        )

        async def _admin(*, task, dataset_id, split, commit, sample_ids=None):
            return MagicMock(result=MagicMock(score=MagicMock(return_value=0.5)))

        engine.evaluate_admin = AsyncMock(side_effect=_admin)
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="train",
            selection_task="math",
            targets=[VerificationTarget(task="math", dataset_id="ds1", split="validation", reward_key="reward")],
        )
        await v.finalize()
        commits = [c.kwargs["commit"] for c in engine.evaluate_admin.await_args_list]
        assert None not in commits
        assert engine.evaluate_admin.await_args.kwargs["commit"] == "agent"

    @pytest.mark.asyncio
    async def test_keeps_candidate_that_beats_baseline(self, tmp_path):
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = self._df()

        async def _admin(*, task, dataset_id, split, commit, sample_ids=None):
            score = 0.3 if commit == "base" else 0.6  # agent genuinely improves
            return MagicMock(result=MagicMock(score=MagicMock(return_value=score)))

        engine.evaluate_admin = AsyncMock(side_effect=_admin)
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="train",
            base_commit="base",
            selection_task="math",
            targets=[VerificationTarget(task="math", dataset_id="ds1", split="validation", reward_key="reward")],
        )
        await v.finalize()
        # 'agent' beats base -> it is selected and target-scored
        assert engine.evaluate_admin.await_args.kwargs["commit"] == "agent"

    @pytest.mark.asyncio
    async def test_floor_off_ships_least_bad_candidate(self, tmp_path):
        # With the floor disabled, the old behavior stands: the best candidate is
        # shipped even if it did not beat the baseline (base is never scored).
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = self._df()

        async def _admin(*, task, dataset_id, split, commit, sample_ids=None):
            return MagicMock(result=MagicMock(score=MagicMock(return_value=0.2)))

        engine.evaluate_admin = AsyncMock(side_effect=_admin)
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="train",
            base_commit="base",
            selection_task="math",
            auto_best_baseline_floor=False,
            targets=[VerificationTarget(task="math", dataset_id="ds1", split="validation", reward_key="reward")],
        )
        await v.finalize()
        rescored = [c.kwargs["commit"] for c in engine.evaluate_admin.await_args_list]
        assert "base" not in rescored
        assert engine.evaluate_admin.await_args.kwargs["commit"] == "agent"


class TestNoCandidateFallback:
    """finalize() floors rewards when the optimizer produced no candidate.

    Found live: with a small budget, an optimizer that spends every eval on
    the seeded baseline leaves an empty candidate pool (the baseline is
    excluded from auto_best selection), and finalize used to 409 -> the outer
    Harbor trial died with RewardFileNotFoundError instead of scoring 0.0.
    """

    @pytest.mark.asyncio
    async def test_auto_best_baseline_only_floors_rewards(self, tmp_path):
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = pd.DataFrame(
            {
                "dataset_subset_split": ["train", "train"],
                "dataset_subset_dataset_id": ["ds1", "ds1"],
                "candidate_commit": ["base", "base"],
                "mean_score": [0.0, 0.2],
                "candidate_created_at": [1, 2],
            }
        )
        engine.evaluate_admin = AsyncMock()
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="train",
            base_commit="base",
            targets=[VerificationTarget(task=None, dataset_id="ds1", split="validation", reward_key="accuracy")],
        )
        rewards = (await v.finalize())["rewards"]
        assert rewards == {"accuracy": 0.0}
        # no candidate -> nothing re-scored, no target eval spent
        engine.evaluate_admin.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auto_best_no_experiments_floors_rewards(self, tmp_path):
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = pd.DataFrame()
        engine.evaluate_admin = AsyncMock()
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="train",
            targets=[VerificationTarget(task=None, dataset_id="ds1", split="validation", reward_key="accuracy")],
        )
        rewards = (await v.finalize())["rewards"]
        assert rewards == {"accuracy": 0.0}
        engine.evaluate_admin.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auto_best_missing_db_still_raises(self, tmp_path):
        # A missing experiment DB is sidecar misconfiguration, not an agent
        # outcome: it must surface as an error, not silently zero the trial.
        engine = MagicMock()
        engine.db = None
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="train",
            targets=[VerificationTarget(task=None, dataset_id="ds1", split="validation", reward_key="accuracy")],
        )
        with pytest.raises(RuntimeError, match="no experiment database"):
            await v.finalize()

    @pytest.mark.asyncio
    async def test_candidates_present_keeps_normal_selection(self, tmp_path):
        # Regression guard: the fallback must not swallow the normal path. Floor off
        # so this isolates candidate selection (the floor is covered separately).
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = pd.DataFrame(
            {
                "dataset_subset_split": ["train", "train"],
                "dataset_subset_dataset_id": ["ds1", "ds1"],
                "candidate_commit": ["base", "agent"],
                "mean_score": [0.9, 0.1],
                "candidate_created_at": [1, 2],
            }
        )

        async def _admin(*, task, dataset_id, split, commit, sample_ids=None):
            return MagicMock(result=MagicMock(score=MagicMock(return_value=0.5)))

        engine.evaluate_admin = AsyncMock(side_effect=_admin)
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="train",
            base_commit="base",
            auto_best_baseline_floor=False,
            targets=[VerificationTarget(task=None, dataset_id="ds1", split="validation", reward_key="accuracy")],
        )
        rewards = (await v.finalize())["rewards"]
        assert rewards == {"accuracy": 0.5}
        assert engine.evaluate_admin.await_args.kwargs["commit"] == "agent"


class TestBaselineAtFinalize:
    """score_baseline=True: finalize also admin-scores the untouched baseline
    and persists it to admin_volume/baseline.json, so regressions are visible
    (observed live: optimization took a weak model from 0.3 to 0.2 and nothing
    surfaced it). reward.json keys are unaffected.
    """

    @pytest.mark.asyncio
    async def test_baseline_scored_and_persisted(self, tmp_path):
        (tmp_path / "submission.json").write_text(json.dumps({"commit": "cand"}))
        engine = _engine([0.2, 0.3])  # candidate target eval, then baseline eval
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="submit",
            base_commit="base",
            score_baseline=True,
            targets=[VerificationTarget(task=None, dataset_id="ds", split="validation", reward_key="accuracy")],
        )
        result = await v.finalize()
        assert result["rewards"] == {"accuracy": 0.2}  # reward.json content unchanged
        # the baseline outcome is surfaced in the finalize response (durable channel:
        # echoed to the trial stdout, which survives teardown; the admin volume does not)
        assert result["baseline"]["scores"] == {"accuracy": 0.3}
        data = json.loads((tmp_path / "baseline.json").read_text())
        assert data == {"accuracy": 0.3}
        # second admin eval was the baseline commit
        assert engine.evaluate_admin.await_args_list[-1].kwargs["commit"] == "base"

    @pytest.mark.asyncio
    async def test_default_off_no_extra_evals(self, tmp_path):
        (tmp_path / "submission.json").write_text(json.dumps({"commit": "cand"}))
        engine = _engine([0.9])
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="submit",
            base_commit="base",
            targets=[VerificationTarget(task=None, dataset_id="ds", split="validation", reward_key="accuracy")],
        )
        rewards = (await v.finalize())["rewards"]
        assert rewards == {"accuracy": 0.9}
        assert engine.evaluate_admin.await_count == 1
        assert not (tmp_path / "baseline.json").exists()

    @pytest.mark.asyncio
    async def test_baseline_failure_retries_then_reports_error_without_failing_trial(self, tmp_path):
        # The baseline eval fails on every attempt (2 by default). The trial reward
        # must survive, AND the failure must be surfaced in the finalize response
        # (not silently swallowed): a live trial once lost its baseline check because
        # the only record was a log line that died with the container at teardown.
        (tmp_path / "submission.json").write_text(json.dumps({"commit": "cand"}))
        engine = MagicMock()
        engine.evaluate_admin = AsyncMock(
            side_effect=[MagicMock(result=MagicMock(score=MagicMock(return_value=0.7))),
                         RuntimeError("modal down"),   # baseline attempt 1
                         RuntimeError("modal down")]   # baseline attempt 2 (retry)
        )
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="submit",
            base_commit="base",
            score_baseline=True,
            targets=[VerificationTarget(task=None, dataset_id="ds", split="validation", reward_key="accuracy")],
        )
        result = await v.finalize()
        assert result["rewards"] == {"accuracy": 0.7}  # trial reward survives baseline failure
        assert result["baseline"]["error_type"] == "RuntimeError"
        assert result["baseline"]["attempts"] == 2  # tried twice before reporting
        # 1 target eval + 2 baseline attempts
        assert engine.evaluate_admin.await_count == 3
        assert not (tmp_path / "baseline.json").exists()  # nothing persisted on failure

    @pytest.mark.asyncio
    async def test_baseline_transient_failure_recovers_on_retry(self, tmp_path):
        # A single transient blip on the baseline eval must not drop the check: the
        # retry succeeds and the baseline score is reported normally.
        (tmp_path / "submission.json").write_text(json.dumps({"commit": "cand"}))
        engine = MagicMock()
        engine.evaluate_admin = AsyncMock(
            side_effect=[MagicMock(result=MagicMock(score=MagicMock(return_value=0.7))),  # target
                         RuntimeError("transient"),                                        # baseline attempt 1
                         MagicMock(result=MagicMock(score=MagicMock(return_value=0.5)))]   # baseline attempt 2 ok
        )
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="submit",
            base_commit="base",
            score_baseline=True,
            targets=[VerificationTarget(task=None, dataset_id="ds", split="validation", reward_key="accuracy")],
        )
        result = await v.finalize()
        assert result["rewards"] == {"accuracy": 0.7}
        assert result["baseline"]["scores"] == {"accuracy": 0.5}
        assert result["baseline"]["attempts"] == 2

    @pytest.mark.asyncio
    async def test_missing_base_commit_warns(self, tmp_path, caplog):
        # score_baseline=True with no base_commit must not be a silent no-op.
        (tmp_path / "submission.json").write_text(json.dumps({"commit": "cand"}))
        engine = _engine([0.9])
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="submit",
            base_commit=None,
            score_baseline=True,
            targets=[VerificationTarget(task=None, dataset_id="ds", split="validation", reward_key="accuracy")],
        )
        with caplog.at_level("WARNING", logger="vero.harbor.verifier"):
            rewards = (await v.finalize())["rewards"]
        assert rewards == {"accuracy": 0.9}
        assert not (tmp_path / "baseline.json").exists()
        assert any("base_commit is not set" in m for m in caplog.messages)


class TestSelectionIntegrity:
    """PR: finalize idempotency, pooled shortlisting, retrying reward-critical
    evals, and the fail-safe floor. Each behavior traces to a live incident:
    optimizers farmed max-over-rows selection with re-measure commits, and a
    disk-full trial shipped no reward.json because finalize was single-shot."""

    def _submit(self, tmp_path, commit="cand"):
        (tmp_path / "submission.json").write_text(json.dumps({"commit": commit}))

    @pytest.mark.asyncio
    async def test_finalize_is_idempotent(self, tmp_path):
        # A retried finalize must replay the first result verbatim, not
        # recompute (a re-run would re-rank against a DB polluted by the first
        # finalize's own admin evals and could crown a different champion).
        self._submit(tmp_path)
        engine = _engine([0.8, 0.1])  # a recompute would consume the 0.1
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="submit",
            targets=[VerificationTarget(task="t", dataset_id="ds", split="test", reward_key="reward")],
        )
        first = await v.finalize()
        second = await v.finalize()
        assert first == second
        assert engine.evaluate_admin.await_count == 1

    @pytest.mark.asyncio
    async def test_same_commit_re_measures_pool_to_mean(self, tmp_path):
        # Re-measuring a commit must reduce variance, not mint lottery draws:
        # A's [0.9, 0.1] pools to 0.5 and loses the only shortlist slot to
        # B's 0.6. Max-over-rows would have shortlisted A on the lucky 0.9.
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = pd.DataFrame(
            {
                "dataset_subset_split": ["validation"] * 3,
                "dataset_subset_dataset_id": ["ds1"] * 3,
                "candidate_commit": ["A", "A", "B"],
                "mean_score": [0.9, 0.1, 0.6],
                "candidate_created_at": [1, 2, 3],
            }
        )
        engine.evaluate_admin = AsyncMock(
            side_effect=lambda **kw: MagicMock(
                result=MagicMock(score=MagicMock(return_value=0.7))
            )
        )
        engine.evaluator.workspace.tree_hash = AsyncMock(side_effect=lambda ref: ref + "-tree")
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="validation",
            selection_task="math",
            rescore_top_k=1,
            auto_best_baseline_floor=False,
            targets=[VerificationTarget(task="math", dataset_id="ds1", split="test", reward_key="reward")],
        )
        await v.finalize()
        rescored = [c.kwargs["commit"] for c in engine.evaluate_admin.await_args_list]
        assert "A" not in rescored
        assert "B" in rescored

    @pytest.mark.asyncio
    async def test_identical_trees_pool_and_do_not_stuff_shortlist(self, tmp_path):
        # A1/A2 are the same content (same git tree) recommitted: they must
        # collapse into ONE candidate group (pooled 0.85), so B still gets a
        # top-2 shortlist slot and the group is re-scored exactly once.
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = pd.DataFrame(
            {
                "dataset_subset_split": ["validation"] * 3,
                "dataset_subset_dataset_id": ["ds1"] * 3,
                "candidate_commit": ["A1", "A2", "B"],
                "mean_score": [0.9, 0.8, 0.55],
                "candidate_created_at": [1, 2, 3],
            }
        )
        engine.evaluate_admin = AsyncMock(
            side_effect=lambda **kw: MagicMock(
                result=MagicMock(score=MagicMock(return_value=0.7))
            )
        )
        trees = {"A1": "T", "A2": "T", "B": "TB"}
        engine.evaluator.workspace.tree_hash = AsyncMock(side_effect=lambda ref: trees[ref])
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="validation",
            selection_task="math",
            rescore_top_k=2,
            auto_best_baseline_floor=False,
            targets=[VerificationTarget(task="math", dataset_id="ds1", split="test", reward_key="reward")],
        )
        await v.finalize()
        # count only selection-split re-scores (the winner also gets a final
        # eval on the TARGET split, which is not a shortlist draw)
        rescored = [
            c.kwargs["commit"]
            for c in engine.evaluate_admin.await_args_list
            if c.kwargs["split"] == "validation"
        ]
        assert "B" in rescored
        assert len([c for c in rescored if c in ("A1", "A2")]) == 1

    @pytest.mark.asyncio
    async def test_floor_fail_safe_reverts_on_unmeasurable_baseline(self, tmp_path):
        # If the floor's baseline eval cannot be measured, shipping the
        # candidate would re-open the shipped-regression hole: revert to seed.
        engine = MagicMock()
        engine.db.get_experiments_df.return_value = pd.DataFrame(
            {
                "dataset_subset_split": ["validation"],
                "dataset_subset_dataset_id": ["ds1"],
                "candidate_commit": ["cand"],
                "mean_score": [0.9],
                "candidate_created_at": [1],
            }
        )

        async def _admin(*, task, dataset_id, split, commit, sample_ids=None):
            if commit == "base" and split == "validation":
                raise RuntimeError("nested run crashed")
            return MagicMock(result=MagicMock(score=MagicMock(return_value=0.9)))

        engine.evaluate_admin = AsyncMock(side_effect=_admin)
        engine.evaluator.workspace.tree_hash = AsyncMock(side_effect=lambda ref: ref + "-tree")
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="validation",
            selection_task="math",
            base_commit="base",
            baseline_score_attempts=2,
            targets=[VerificationTarget(task="math", dataset_id="ds1", split="test", reward_key="reward")],
        )
        await v.finalize()
        # the final (target) eval ran on the SEED, not the unverified candidate
        assert engine.evaluate_admin.await_args.kwargs["commit"] == "base"
        assert engine.evaluate_admin.await_args.kwargs["split"] == "test"

    @pytest.mark.asyncio
    async def test_target_eval_retries_then_floors_with_error_marker(self, tmp_path):
        # A persistently failing target eval floors the reward (reward.json
        # must still ship) and records the outage in the wrapper so a floored
        # reward can never masquerade as a measured 0.0.
        self._submit(tmp_path)
        engine = MagicMock()
        engine.evaluate_admin = AsyncMock(side_effect=RuntimeError("boom"))
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="submit",
            baseline_score_attempts=2,
            targets=[VerificationTarget(task="t", dataset_id="ds", split="test", reward_key="reward")],
        )
        result = await v.finalize()
        assert result["rewards"] == {"reward": 0.0}
        assert "reward" in result["target_errors"]
        assert engine.evaluate_admin.await_count == 2  # retried before flooring

    @pytest.mark.asyncio
    async def test_target_eval_transient_failure_recovers(self, tmp_path):
        self._submit(tmp_path)
        engine = MagicMock()
        healthy = MagicMock(result=MagicMock(score=MagicMock(return_value=0.8)))
        engine.evaluate_admin = AsyncMock(side_effect=[RuntimeError("blip"), healthy])
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="submit",
            baseline_score_attempts=2,
            targets=[VerificationTarget(task="t", dataset_id="ds", split="test", reward_key="reward")],
        )
        result = await v.finalize()
        assert result["rewards"] == {"reward": 0.8}
        assert "target_errors" not in result

    @pytest.mark.asyncio
    async def test_all_error_eval_is_retried_not_zeroed(self, tmp_path):
        # An eval in which no sample scored (score(fill_score=None) is None)
        # is an outage: retry it, never let it quietly become a measured 0.0.
        self._submit(tmp_path)
        engine = MagicMock()
        all_error = MagicMock(
            result=MagicMock(
                score=MagicMock(side_effect=lambda fill_score=0.0: None if fill_score is None else 0.0)
            )
        )
        healthy = MagicMock(result=MagicMock(score=MagicMock(return_value=0.7)))
        engine.evaluate_admin = AsyncMock(side_effect=[all_error, healthy])
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="submit",
            baseline_score_attempts=2,
            targets=[VerificationTarget(task="t", dataset_id="ds", split="test", reward_key="reward")],
        )
        result = await v.finalize()
        assert result["rewards"] == {"reward": 0.7}
        assert engine.evaluate_admin.await_count == 2
