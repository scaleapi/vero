"""Tests for the Evaluator strategy seam (vero.evaluation.strategy)."""

import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from vero.core.db.candidate import Candidate
from vero.core.db.dataset import DatasetSample, DatasetSubset
from vero.core.db.result import SampleResult
from vero.core.db.run import ExperimentRun
from vero.core.evaluation import EvaluationParameters
from vero.core.sessions import get_vero_home_dir, save_sample_result
from vero.evaluation.evaluator import Evaluator


def _mock_workspace():
    ws = MagicMock()
    ws.name = "repo"
    ws.is_dirty = AsyncMock(return_value=False)

    @contextlib.asynccontextmanager
    async def _at(commit):
        yield

    ws.at = _at
    return ws


@pytest.mark.asyncio
async def test_injected_strategy_produces_results(tmp_path, monkeypatch):
    monkeypatch.setenv("VERO_HOME_DIR", str(tmp_path / "vero_home"))

    called = {}

    class FakeStrategy:
        async def produce_sample_results(self, *, workspace, params, result_dir):
            called["yes"] = True
            save_sample_result(
                get_vero_home_dir() / "sessions",
                params.session_id,
                params.result_id,
                sample_id=0,
                result=SampleResult(
                    dataset_sample=DatasetSample(sample_id=0, split="test", dataset_id="ds"),
                    score=1.0,
                    commit=params.run.candidate.commit,
                    result_id=params.result_id,
                ),
            )

    evaluator = Evaluator(_mock_workspace(), session_id="s", eval_strategy=FakeStrategy())
    params = EvaluationParameters(
        run=ExperimentRun(
            candidate=Candidate(commit="c1", repo_name="repo"),
            dataset_subset=DatasetSubset(split="test", dataset_id="ds", sample_ids=[0]),
        ),
        session_id="s",
    )

    result = await evaluator.run(params, use_copy=False)

    assert called.get("yes") is True
    assert result.sample_results[0].score == 1.0
    # Mode-A staging path was NOT taken (strategy branch); sandbox untouched
    evaluator.workspace.sandbox.upload.assert_not_called()
