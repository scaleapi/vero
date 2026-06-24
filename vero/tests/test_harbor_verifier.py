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
    async def test_finalize_submit_no_submission_raises(self, tmp_path):
        v = Verifier(
            engine=_engine([]),
            admin_volume=tmp_path,
            reward_mode="submit",
            targets=[VerificationTarget(task="t", dataset_id="ds1", split="test", reward_key="reward")],
        )
        with pytest.raises(NoCandidateError):
            await v.finalize()


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
    async def test_finalize_auto_best_picks_top_validation_score(self, tmp_path):
        engine = _engine([0.95])
        engine.db.get_experiments_df.return_value = pd.DataFrame(
            {
                "dataset_subset_split": ["validation", "validation", "train"],
                "candidate_commit": ["lo", "hi", "ignored"],
                "mean_score": [0.5, 0.9, 1.0],
                "candidate_created_at": [1, 2, 3],
            }
        )
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="validation",
            targets=[VerificationTarget(task="t", dataset_id="ds1", split="test", reward_key="reward")],
        )
        rewards = await v.finalize()
        assert rewards == {"reward": 0.95}
        # selected the highest validation score ("hi"), not the train row
        assert engine.evaluate_admin.await_args.kwargs["commit"] == "hi"

    @pytest.mark.asyncio
    async def test_finalize_auto_best_excludes_baseline(self, tmp_path):
        engine = _engine([0.7])
        engine.db.get_experiments_df.return_value = pd.DataFrame(
            {
                "dataset_subset_split": ["validation", "validation"],
                "candidate_commit": ["base", "agent"],
                "mean_score": [0.99, 0.6],
                "candidate_created_at": [1, 2],
            }
        )
        v = Verifier(
            engine=engine,
            admin_volume=tmp_path,
            reward_mode="auto_best",
            selection_split="validation",
            base_commit="base",
            targets=[VerificationTarget(task="t", dataset_id="ds1", split="test", reward_key="reward")],
        )
        await v.finalize()
        # baseline excluded even though it scored higher
        assert engine.evaluate_admin.await_args.kwargs["commit"] == "agent"
