"""Authorization, budget, backend, persistence, and disclosure boundary."""

from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path
from typing import Awaitable, Callable

from vero.evaluation.backend import BackendRegistry
from vero.evaluation.budget import BudgetLedger
from vero.evaluation.evaluator import Evaluator
from vero.evaluation.exceptions import EvaluationDeniedError, EvaluationExecutionError
from vero.evaluation.models import (
    EvaluationAcknowledgement,
    EvaluationAuthorization,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationSummary,
    ObjectiveSpec,
)
from vero.evaluation.objective import project_evaluation
from vero.evaluation.persistence import EvaluationDatabase, EvaluationStore

logger = logging.getLogger(__name__)

AuthorizationResolver = Callable[
    [str, EvaluationRequest],
    EvaluationAuthorization | Awaitable[EvaluationAuthorization],
]


class EvaluationEngine:
    """The only runtime path from an evaluation request to a stored record."""

    def __init__(
        self,
        *,
        evaluator: Evaluator,
        backends: BackendRegistry,
        database: EvaluationDatabase,
        database_path: Path | None = None,
        budget_ledger: BudgetLedger | None = None,
        authorization_resolver: AuthorizationResolver | None = None,
    ):
        self.evaluator = evaluator
        self.backends = backends
        self.database = database
        self.database_path = database_path
        self.budget_ledger = budget_ledger
        self.authorization_resolver = authorization_resolver
        self.listeners: list[Callable[[EvaluationRecord], object]] = []
        self._record_lock = asyncio.Lock()

    async def _authorization(
        self,
        backend_id: str,
        request: EvaluationRequest,
        supplied: EvaluationAuthorization | None,
    ) -> EvaluationAuthorization:
        if supplied is not None:
            return supplied
        if self.authorization_resolver is None:
            return EvaluationAuthorization(may_evaluate=True)
        resolved = self.authorization_resolver(backend_id, request)
        if inspect.isawaitable(resolved):
            resolved = await resolved
        return resolved

    async def _evaluate_record(
        self,
        *,
        backend_id: str,
        request: EvaluationRequest,
        objective_spec: ObjectiveSpec | None,
        authorization: EvaluationAuthorization | None,
    ) -> tuple[EvaluationRecord, EvaluationAuthorization]:
        backend = self.backends.resolve(backend_id)
        decision = await self._authorization(backend_id, request, authorization)
        if not decision.may_evaluate:
            raise EvaluationDeniedError(decision.reason or "evaluation is not authorized")

        validate_request = getattr(backend, "validate_request", None)
        if callable(validate_request):
            validate_request(request)
        cost = await backend.resolve_cost(request.evaluation_set)
        if decision.meter_budget and self.budget_ledger is not None:
            await self.budget_ledger.reserve(
                backend_id,
                request.evaluation_set,
                cost,
            )

        try:
            record = await self.evaluator.evaluate(
                backend_id=backend_id,
                backend=backend,
                request=request,
                objective_spec=objective_spec,
            )
        except EvaluationExecutionError as error:
            failure = EvaluationStore(
                self.evaluator.evaluations_dir / error.evaluation_id
            ).load()
            await self._record(failure)
            raise
        await self._record(record)
        return record, decision

    async def _record(self, record: EvaluationRecord) -> None:
        """Index, persist, and publish a completed evaluation exactly once."""
        async with self._record_lock:
            self.database.add_evaluation(record)
            if self.database_path is not None:
                self.database.save_to_file(self.database_path)
        for listener in self.listeners:
            try:
                result = listener(record)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Evaluation listener failed for %s", record.id)

    async def evaluate_record(
        self,
        *,
        backend_id: str,
        request: EvaluationRequest,
        objective_spec: ObjectiveSpec | None = None,
        authorization: EvaluationAuthorization | None = None,
    ) -> EvaluationRecord:
        record, _ = await self._evaluate_record(
            backend_id=backend_id,
            request=request,
            objective_spec=objective_spec,
            authorization=authorization,
        )
        return record

    async def evaluate(
        self,
        *,
        backend_id: str,
        request: EvaluationRequest,
        objective_spec: ObjectiveSpec | None = None,
        authorization: EvaluationAuthorization | None = None,
    ) -> EvaluationRecord | EvaluationSummary | EvaluationAcknowledgement:
        record, decision = await self._evaluate_record(
            backend_id=backend_id,
            request=request,
            objective_spec=objective_spec,
            authorization=authorization,
        )
        return project_evaluation(record, decision.disclosure)
