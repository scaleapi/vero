"""Tests for vero.harbor.server.EvaluationSidecar — handlers, tier-routing, submit."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from vero.core.budget import BudgetLedger, SplitBudget
from vero.core.dataset.base import SplitAccess
from vero.core.db.candidate import Candidate
from vero.core.db.dataset import DatasetSample, DatasetSubset
from vero.core.db.database import Experiment
from vero.core.db.result import (
    ExperimentResult,
    ExperimentResultStatus,
    SampleResult,
)
from vero.core.db.run import ExperimentRun
from vero.harbor.server import EvaluationSidecar, SubmitDisabledError
from vero.evaluation.engine import EvalRequest


def _experiment(split: str, commit: str = "abcdef123456") -> Experiment:
    run = ExperimentRun(
        candidate=Candidate(commit=commit, repo_name="r"),
        dataset_subset=DatasetSubset(split=split, dataset_id="ds1"),
    )
    sample_results = {
        i: SampleResult(
            dataset_sample=DatasetSample(sample_id=i, split=split, dataset_id="ds1"),
            score=float(i % 2),
            feedback=f"Expected: secret-{i}",  # label-bearing: must NOT reach agent on partial
        )
        for i in range(3)
    }
    return Experiment(
        run=run,
        result=ExperimentResult(
            run_id=run.id, status=ExperimentResultStatus.SUCCESS, sample_results=sample_results
        ),
    )


def _sidecar(tmp_path, *, split, submit_enabled=False):
    engine = MagicMock()
    engine.evaluate = AsyncMock(return_value=_experiment(split))
    engine.budget = BudgetLedger(
        [SplitBudget(split=split, dataset_id="ds1", total_run_budget=5, total_sample_budget=100)]
    )
    sidecar = EvaluationSidecar(
        engine=engine,
        split_accesses=[SplitAccess.non_viewable("validation"), SplitAccess.no_access("test")],
        agent_repo_path=tmp_path / "agent_repo",
        agent_volume=tmp_path / "agent_vol",
        admin_volume=tmp_path / "admin_vol",
        submit_enabled=submit_enabled,
    )
    # Stub the git transfer (integration-tested separately); pin the sha.
    sidecar._transfer_commit = AsyncMock(return_value="abcdef123456")
    return sidecar


class TestRouting:
    @pytest.mark.asyncio
    async def test_visible_split_writes_full_per_sample(self, tmp_path):
        sidecar = _sidecar(tmp_path, split="train")  # train defaults to viewable
        summary = await sidecar.evaluate(EvalRequest(dataset_id="ds1", split="train"))

        dest = tmp_path / "agent_vol" / "results" / "train__abcdef123456"
        assert (dest / "summary.json").exists()
        assert {(dest / f"{i}.json").exists() for i in range(3)} == {True}
        assert summary.result_path == str(dest)
        assert summary.n_samples == 3

    @pytest.mark.asyncio
    async def test_partial_split_writes_summary_only_no_labels(self, tmp_path):
        sidecar = _sidecar(tmp_path, split="validation")  # non_viewable -> partial
        summary = await sidecar.evaluate(EvalRequest(dataset_id="ds1", split="validation"))

        dest = tmp_path / "agent_vol" / "results" / "validation__abcdef123456"
        assert (dest / "summary.json").exists()
        # NO per-sample files -> the label-bearing feedback never reaches the agent
        assert not list(dest.glob("[0-9]*.json"))
        blob = (dest / "summary.json").read_text()
        assert "secret-" not in blob
        assert summary.result_path == str(dest)

    @pytest.mark.asyncio
    async def test_admin_eval_writes_nothing_to_agent_volume(self, tmp_path):
        sidecar = _sidecar(tmp_path, split="test")  # no_access; admin only
        summary = await sidecar.evaluate(
            EvalRequest(dataset_id="ds1", split="test"), admin=True
        )
        assert not (tmp_path / "agent_vol").exists() or not list(
            (tmp_path / "agent_vol").rglob("*.json")
        )
        assert summary.result_path is None
        # admin call bypasses metering
        assert summary.budget_remaining is None


class TestSubmit:
    @pytest.mark.asyncio
    async def test_submit_records_nomination(self, tmp_path):
        sidecar = _sidecar(tmp_path, split="train", submit_enabled=True)
        out = await sidecar.submit(commit="deadbeef")
        rec = json.loads((tmp_path / "admin_vol" / "submission.json").read_text())
        assert rec["commit"] == "abcdef123456"  # the transferred sha
        assert out["submitted_commit"] == "abcdef123456"

    @pytest.mark.asyncio
    async def test_submit_disabled_raises(self, tmp_path):
        sidecar = _sidecar(tmp_path, split="train", submit_enabled=False)
        with pytest.raises(SubmitDisabledError):
            await sidecar.submit(commit="x")


class TestStatus:
    def test_status_reports_submit_and_splits(self, tmp_path):
        sidecar = _sidecar(tmp_path, split="train", submit_enabled=True)
        status = sidecar.status()
        assert status.submit_enabled is True
        assert status.splits[0]["split"] == "train"
        assert status.splits[0]["remaining_run_budget"] == 5
