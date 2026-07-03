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
        rewards = await v.finalize()
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
        rewards = await v.finalize()
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
        rewards = await v.finalize()
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
        rewards = await v.finalize()
        # the final (target) eval is on the WINNER 'lo', chosen by admin re-score
        assert engine.evaluate_admin.await_args.kwargs["commit"] == "lo"
        assert engine.evaluate_admin.await_args.kwargs["split"] == "test"
        # reward is the winner 'lo' scored on the target split -> its admin score
        assert rewards["reward"] == 0.95

    @pytest.mark.asyncio
    async def test_auto_best_excludes_baseline_after_rescore(self, tmp_path):
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
            targets=[VerificationTarget(task="math", dataset_id="ds1", split="test", reward_key="reward")],
        )
        await v.finalize()
        # baseline 'base' was never re-scored; 'agent' is selected + target-scored
        rescored_commits = [c.kwargs["commit"] for c in engine.evaluate_admin.await_args_list]
        assert "base" not in rescored_commits
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
        rewards = await v.finalize()
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
        rewards = await v.finalize()
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
        # Regression guard: the fallback must not swallow the normal path.
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
            targets=[VerificationTarget(task=None, dataset_id="ds1", split="validation", reward_key="accuracy")],
        )
        rewards = await v.finalize()
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
        rewards = await v.finalize()
        assert rewards == {"accuracy": 0.2}  # reward.json content unchanged
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
        rewards = await v.finalize()
        assert rewards == {"accuracy": 0.9}
        assert engine.evaluate_admin.await_count == 1
        assert not (tmp_path / "baseline.json").exists()

    @pytest.mark.asyncio
    async def test_baseline_failure_never_fails_trial(self, tmp_path):
        (tmp_path / "submission.json").write_text(json.dumps({"commit": "cand"}))
        engine = MagicMock()
        engine.evaluate_admin = AsyncMock(
            side_effect=[MagicMock(result=MagicMock(score=MagicMock(return_value=0.7))),
                         RuntimeError("modal down")]
        )
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="submit",
            base_commit="base",
            score_baseline=True,
            targets=[VerificationTarget(task=None, dataset_id="ds", split="validation", reward_key="accuracy")],
        )
        rewards = await v.finalize()
        assert rewards == {"accuracy": 0.7}  # trial reward survives baseline failure
