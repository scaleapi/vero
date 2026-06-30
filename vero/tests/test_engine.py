"""Tests for EvaluationEngine (vero.evaluation.engine) — the shared evaluation core."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from vero.core.budget import BudgetLedger, SplitBudget
from vero.core.dataset import DatasetInfo
from vero.exceptions import ExperimentBudgetExceeded, InvalidSplitError
from vero.evaluation.engine import EvalRequest, EvaluationEngine

_DATASET_INFO = DatasetInfo(
    id="ds1", splits={"dev": 100, "test": 50}, features={"dev": [], "test": []}
)


def _make_service(budgets=None, monkeypatch=None):
    evaluator = MagicMock()
    evaluator.evaluate = AsyncMock(return_value="EXPERIMENT")  # sentinel
    svc = EvaluationEngine(
        evaluator=evaluator,
        budget=BudgetLedger(
            budgets
            or [SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=100, total_run_budget=3)]
        ),
        default_task="main",
        session_id="s1",
    )
    if monkeypatch is not None:
        monkeypatch.setattr(svc, "_get_dataset_info", lambda dataset_id: _DATASET_INFO)
    return svc


class TestResolveSamples:
    def test_rejects_both_sample_ids_and_num_samples(self, monkeypatch):
        svc = _make_service(monkeypatch=monkeypatch)
        with pytest.raises(ValueError, match="both sample_ids and num_samples"):
            svc.resolve_samples(EvalRequest(dataset_id="ds1", split="dev", sample_ids=[0], num_samples=1))

    def test_num_samples_first_n(self, monkeypatch):
        svc = _make_service(monkeypatch=monkeypatch)
        ids, n = svc.resolve_samples(EvalRequest(dataset_id="ds1", split="dev", num_samples=5))
        assert ids == [0, 1, 2, 3, 4] and n == 5

    def test_num_samples_full_split_is_none(self, monkeypatch):
        svc = _make_service(monkeypatch=monkeypatch)
        ids, n = svc.resolve_samples(EvalRequest(dataset_id="ds1", split="dev", num_samples=100))
        assert ids is None and n == 100  # None == whole split

    def test_sample_ids_out_of_range_raises(self, monkeypatch):
        svc = _make_service(monkeypatch=monkeypatch)
        with pytest.raises(ValueError, match="outside the range"):
            svc.resolve_samples(EvalRequest(dataset_id="ds1", split="dev", sample_ids=[0, 999]))


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_evaluate_meters_and_runs(self, monkeypatch):
        svc = _make_service(monkeypatch=monkeypatch)
        exp = await svc.evaluate(EvalRequest(dataset_id="ds1", split="dev", commit="c1", num_samples=10))

        assert exp == "EXPERIMENT"
        svc.evaluator.evaluate.assert_awaited_once()
        kwargs = svc.evaluator.evaluate.await_args.kwargs
        assert kwargs["commit"] == "c1" and kwargs["split"] == "dev" and kwargs["task"] == "main"
        assert kwargs["sample_ids"] == list(range(10))
        # budget metered
        assert svc.status()[("dev", "ds1")].remaining_run_budget == 2
        assert svc.status()[("dev", "ds1")].remaining_sample_budget == 90

    @pytest.mark.asyncio
    async def test_evaluate_budget_exhausted_does_not_run(self, monkeypatch):
        # 50-sample budget; num_samples=60 caps to 60 (< split size 100) and exceeds it
        svc = _make_service(
            budgets=[SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=50)],
            monkeypatch=monkeypatch,
        )
        with pytest.raises(ExperimentBudgetExceeded):
            await svc.evaluate(EvalRequest(dataset_id="ds1", split="dev", commit="c1", num_samples=60))
        svc.evaluator.evaluate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_evaluate_unknown_split_rejected(self, monkeypatch):
        svc = _make_service(monkeypatch=monkeypatch)
        with pytest.raises(InvalidSplitError):
            await svc.evaluate(EvalRequest(dataset_id="ds1", split="test", commit="c1", num_samples=10))

    @pytest.mark.asyncio
    async def test_admin_bypasses_budget(self, monkeypatch):
        svc = _make_service(monkeypatch=monkeypatch)
        # 'test' isn't in the agent budget map, but admin may evaluate it
        await svc.evaluate(
            EvalRequest(dataset_id="ds1", split="test", commit="c1", num_samples=10), admin=True
        )
        svc.evaluator.evaluate.assert_awaited_once()
        # nothing metered
        assert svc.status()[("dev", "ds1")].remaining_run_budget == 3


class TestNoAccessGate:
    @pytest.mark.asyncio
    async def test_evaluate_no_access_split_rejected_before_ledger(self, monkeypatch):
        from vero.core.dataset import SplitAccess

        # Pathological-but-instructive: a no_access split that DOES have a ledger
        # entry. Without the explicit tier gate the implicit ledger check would
        # let it through. With the gate it must be rejected before reserve().
        svc = _make_service(
            budgets=[
                SplitBudget(split="test", dataset_id="ds1", total_sample_budget=100, total_run_budget=3)
            ],
            monkeypatch=monkeypatch,
        )
        svc.split_accesses = [SplitAccess.no_access("test")]
        with pytest.raises(InvalidSplitError):
            await svc.evaluate(
                EvalRequest(dataset_id="ds1", split="test", commit="c1", num_samples=10)
            )
        svc.evaluator.evaluate.assert_not_awaited()
        # nothing metered: tier gate fired before the ledger
        assert svc.status()[("test", "ds1")].remaining_run_budget == 3
        assert svc.status()[("test", "ds1")].remaining_sample_budget == 100

    @pytest.mark.asyncio
    async def test_evaluate_unlisted_split_defaults_no_access(self, monkeypatch):
        from vero.core.dataset import SplitAccess

        svc = _make_service(
            budgets=[
                SplitBudget(split="test", dataset_id="ds1", total_sample_budget=100, total_run_budget=3)
            ],
            monkeypatch=monkeypatch,
        )
        # 'test' is not listed in split_accesses at all -> fail closed to no_access
        svc.split_accesses = [SplitAccess.viewable("dev")]
        with pytest.raises(InvalidSplitError):
            await svc.evaluate(
                EvalRequest(dataset_id="ds1", split="test", commit="c1", num_samples=10)
            )
        svc.evaluator.evaluate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_admin_bypasses_no_access_gate(self, monkeypatch):
        from vero.core.dataset import SplitAccess

        svc = _make_service(monkeypatch=monkeypatch)
        svc.split_accesses = [SplitAccess.no_access("test")]
        await svc.evaluate(
            EvalRequest(dataset_id="ds1", split="test", commit="c1", num_samples=10),
            admin=True,
        )
        svc.evaluator.evaluate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_viewable_split_still_evaluable(self, monkeypatch):
        from vero.core.dataset import SplitAccess

        svc = _make_service(monkeypatch=monkeypatch)  # 'dev' budget present
        svc.split_accesses = [SplitAccess.non_viewable("dev")]
        await svc.evaluate(
            EvalRequest(dataset_id="ds1", split="dev", commit="c1", num_samples=10)
        )
        svc.evaluator.evaluate.assert_awaited_once()
        assert svc.status()[("dev", "ds1")].remaining_run_budget == 2
