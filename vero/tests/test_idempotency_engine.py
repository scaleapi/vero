"""Budget reservations must not leak when the evaluator fails unexpectedly.

The engine refunds a reservation on cancellation (typed and raw), on a recorded
execution failure, and on an infrastructure diagnostic, but a bare exception from
the evaluator used to escape every one of those handlers with the reservation
still charged. Nothing else notices, so the run continues against a permanently
short budget and the evaluations at the end of a long search are starved.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from vero.candidate import Candidate
from vero.evaluation import (
    BackendProvenance,
    BackendRegistry,
    BudgetLedger,
    EvaluationBudget,
    EvaluationCost,
    EvaluationDatabase,
    EvaluationEngine,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStore,
    Evaluator,
    allow_all_evaluations,
)


class NeverCheckedOutRepository:
    """Candidate repository stand-in: these tests fail before any checkout."""

    family = "stub"

    def checkout(self, candidate, *, sandbox, name=None):
        raise AssertionError("the running-manifest write must fail before checkout")


class StubBackend:
    """Minimal backend: only resolve_cost is reached before the failure."""

    @property
    def provenance(self) -> BackendProvenance:
        return BackendProvenance(name="stub", version="1", config_digest="0" * 64)

    async def resolve_cost(self, evaluation_set: EvaluationSet) -> EvaluationCost:
        return EvaluationCost(runs=1, cases=4)

    async def evaluate(self, *, context, request) -> EvaluationReport:
        raise AssertionError("the running-manifest write must fail before evaluate")


def request() -> EvaluationRequest:
    return EvaluationRequest(
        candidate=Candidate(
            id="id:candidate",
            version="candidate",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        evaluation_set=EvaluationSet(name="performance"),
    )


def engine_with_budget(tmp_path: Path, ledger: BudgetLedger) -> EvaluationEngine:
    return EvaluationEngine(
        evaluator=Evaluator(
            candidate_repository=NeverCheckedOutRepository(),
            sandbox=None,
            session_dir=tmp_path / "sessions" / "session",
        ),
        backends=BackendRegistry({"default": StubBackend()}),
        database=EvaluationDatabase(id="session"),
        database_path=tmp_path / "database.json",
        budget_ledger=ledger,
        authorization_resolver=allow_all_evaluations,
    )


def ledger_with_two_runs(tmp_path: Path) -> BudgetLedger:
    evaluation_set = request().evaluation_set
    ledger = BudgetLedger(
        [
            EvaluationBudget(
                backend_id="default",
                evaluation_set_key=evaluation_set.budget_key("default"),
                total_runs=2,
                total_cases=8,
            )
        ],
        path=tmp_path / "budgets.json",
    )
    ledger.save()
    return ledger


def fail_running_manifest(monkeypatch) -> None:
    """Break the evaluator's pre-try running-manifest write.

    This is the concrete unexpected failure: write_running happens before the
    evaluator's own try block, so an OSError from it is never converted into
    EvaluationCancelledError or EvaluationExecutionError and arrives at the
    engine as a bare exception.
    """

    def boom(self, **kwargs):
        raise OSError("running manifest write failed")

    monkeypatch.setattr(EvaluationStore, "write_running", boom)


@pytest.mark.asyncio
async def test_unexpected_evaluator_failure_refunds_the_reservation(
    tmp_path: Path, monkeypatch
):
    ledger = ledger_with_two_runs(tmp_path)
    evaluation_set = request().evaluation_set
    engine = engine_with_budget(tmp_path, ledger)
    fail_running_manifest(monkeypatch)

    # The bare exception still reaches the caller unchanged: the refund is the
    # only new behaviour, so retry classification upstream is untouched.
    with pytest.raises(OSError, match="running manifest write failed"):
        await engine.evaluate_record(backend_id="default", request=request())

    restored = ledger.get("default", evaluation_set)
    assert restored is not None
    assert restored.remaining_runs == 2
    assert restored.remaining_cases == 8
    durable = BudgetLedger.load(tmp_path / "budgets.json").get(
        "default", evaluation_set
    )
    assert durable is not None
    assert durable.remaining_runs == 2
    assert durable.remaining_cases == 8


@pytest.mark.asyncio
async def test_unexpected_failure_refund_failure_is_chained_not_substituted(
    tmp_path: Path, monkeypatch
):
    # A refund that fails on its own must not replace the error that actually
    # stopped the evaluation, the same contract the cancellation and
    # infrastructure handlers already hold themselves to.
    ledger = ledger_with_two_runs(tmp_path)
    engine = engine_with_budget(tmp_path, ledger)
    fail_running_manifest(monkeypatch)

    async def failing_refund(*args, **kwargs):
        raise RuntimeError("durable refund write failed")

    monkeypatch.setattr(ledger, "refund", failing_refund)

    with pytest.raises(OSError, match="running manifest write failed") as raised:
        await engine.evaluate_record(backend_id="default", request=request())
    assert isinstance(raised.value.__cause__, RuntimeError)
