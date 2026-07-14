"""Authorization, budget, registry, database, and disclosure policy boundary."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Awaitable, Callable

from vero.evaluation.backend import BackendRegistry
from vero.evaluation.budget import BudgetLedger
from vero.evaluation.evaluator import Evaluator
from vero.evaluation.exceptions import EvaluationDeniedError
from vero.evaluation.models import (
    DisclosureLevel,
    EvaluationAcknowledgement,
    EvaluationAuthorization,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationSummary,
    ObjectiveSpec,
)
from vero.evaluation.objective import project_evaluation
from vero.evaluation.persistence import EvaluationDatabase

logger = logging.getLogger(__name__)

AuthorizationResolver = Callable[
    [str, EvaluationRequest],
    EvaluationAuthorization | Awaitable[EvaluationAuthorization],
]


class EvaluationEngine:
    """Evaluate only through approved backends and trusted policy decisions."""

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

    async def _resolve_authorization(
        self,
        backend_id: str,
        request: EvaluationRequest,
        authorization: EvaluationAuthorization | None,
    ) -> EvaluationAuthorization:
        if authorization is not None:
            return authorization
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
        objective_spec: ObjectiveSpec | None = None,
        authorization: EvaluationAuthorization | None = None,
    ) -> tuple[EvaluationRecord, EvaluationAuthorization]:
        backend = self.backends.resolve(backend_id)
        decision = await self._resolve_authorization(
            backend_id,
            request,
            authorization,
        )
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

        record = await self.evaluator.evaluate(
            backend_id=backend_id,
            backend=backend,
            request=request,
            objective_spec=objective_spec,
        )
        self.database.add_evaluation(record)
        if self.database_path is not None:
            self.database.save_to_file(self.database_path)
        for listener in self.listeners:
            try:
                result = listener(record)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Evaluation callback failed for %s", record.id)
        return record, decision

    async def evaluate_record(
        self,
        *,
        backend_id: str,
        request: EvaluationRequest,
        objective_spec: ObjectiveSpec | None = None,
        authorization: EvaluationAuthorization | None = None,
    ) -> EvaluationRecord:
        """Run an evaluation and return its full trusted record."""
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
