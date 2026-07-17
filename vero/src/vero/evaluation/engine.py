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
from vero.evaluation.exceptions import (
    EvaluationCancelledError,
    EvaluationDeniedError,
    EvaluationExecutionError,
    EvaluationInfrastructureError,
    EvaluationRequestError,
)
from vero.evaluation.models import (
    AgentSelectionMode,
    EvaluationAcknowledgement,
    EvaluationAuthorization,
    EvaluationPrincipal,
    EvaluationPlan,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationSummary,
    ObjectiveSpec,
)
from vero.evaluation.objective import project_evaluation
from vero.evaluation.persistence import EvaluationDatabase, EvaluationStore

logger = logging.getLogger(__name__)

AuthorizationResolver = Callable[
    [EvaluationPrincipal, str, EvaluationRequest],
    EvaluationAuthorization | Awaitable[EvaluationAuthorization],
]


def allow_all_evaluations(
    _principal: EvaluationPrincipal,
    _backend_id: str,
    _request: EvaluationRequest,
) -> EvaluationAuthorization:
    """Explicit resolver for trusted runtimes without an evaluation boundary."""

    return EvaluationAuthorization(
        may_evaluate=True,
        may_view=True,
        expose_case_resources=True,
    )


def authorize_evaluation_plan(plan: EvaluationPlan) -> AuthorizationResolver:
    """Build the canonical principal-aware resolver for an evaluation plan."""

    def resolve(
        principal: EvaluationPrincipal,
        _backend_id: str,
        request: EvaluationRequest,
    ) -> EvaluationAuthorization:
        definition = plan.for_evaluation_set(request.evaluation_set)
        if definition is None:
            return EvaluationAuthorization(
                may_evaluate=False,
                may_view=False,
                reason="evaluation set is not present in the session plan",
            )
        access = definition.access
        if principal == EvaluationPrincipal.ADMIN:
            return EvaluationAuthorization(
                may_evaluate=True,
                may_view=False,
                meter_budget=False,
                disclosure=access.disclosure,
            )
        if principal == EvaluationPrincipal.SYSTEM:
            return EvaluationAuthorization(
                may_evaluate=True,
                may_view=access.agent_visible,
                meter_budget=definition.system_budget is not None,
                disclosure=access.disclosure,
                expose_case_resources=False,
            )
        if (
            access.agent_selection == AgentSelectionMode.FIXED
            and request.evaluation_set.selection
            != definition.evaluation_set.selection
        ):
            return EvaluationAuthorization(
                may_evaluate=False,
                may_view=access.agent_visible,
                reason="evaluation set requires its fixed case selection",
            )
        return EvaluationAuthorization(
            may_evaluate=access.agent_can_evaluate,
            may_view=access.agent_visible,
            meter_budget=definition.agent_budget is not None,
            disclosure=access.disclosure,
            expose_case_resources=access.expose_case_resources,
            reason=(
                None
                if access.agent_can_evaluate
                else "evaluation set is not agent-evaluable"
            ),
        )

    return resolve


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

    async def authorize(
        self,
        backend_id: str,
        request: EvaluationRequest,
        principal: EvaluationPrincipal = EvaluationPrincipal.AGENT,
        supplied: EvaluationAuthorization | None = None,
    ) -> EvaluationAuthorization:
        """Resolve the trusted access decision without executing an evaluation."""

        if supplied is not None:
            return supplied
        if self.authorization_resolver is None:
            return EvaluationAuthorization(
                may_evaluate=False,
                reason="evaluation authorization was not configured",
            )
        resolved = self.authorization_resolver(principal, backend_id, request)
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
        principal: EvaluationPrincipal,
    ) -> tuple[EvaluationRecord, EvaluationAuthorization]:
        backend = self.backends.resolve(backend_id)
        decision = await self.authorize(
            backend_id,
            request,
            principal,
            authorization,
        )
        if not decision.may_evaluate:
            raise EvaluationDeniedError(
                decision.reason or "evaluation is not authorized"
            )

        validate_request = getattr(backend, "validate_request", None)
        if callable(validate_request):
            try:
                validate_request(request)
            except ValueError as error:
                raise EvaluationRequestError(str(error)) from error
        try:
            cost = await backend.resolve_cost(request.evaluation_set)
        except ValueError as error:
            raise EvaluationRequestError(str(error)) from error
        charged = decision.meter_budget and self.budget_ledger is not None
        if charged:
            await self.budget_ledger.reserve(
                backend_id,
                request.evaluation_set,
                cost,
                principal,
            )

        try:
            record = await self.evaluator.evaluate(
                backend_id=backend_id,
                backend=backend,
                request=request,
                objective_spec=objective_spec,
                principal=principal,
            )
        except EvaluationCancelledError as error:
            cancelled = EvaluationStore(
                self.evaluator.evaluations_dir / error.evaluation_id
            ).load()
            await asyncio.shield(self._record(cancelled))
            if charged:
                await asyncio.shield(
                    self.budget_ledger.refund(
                        backend_id,
                        request.evaluation_set,
                        cost,
                        principal,
                    )
                )
            raise
        except EvaluationExecutionError as error:
            failure = EvaluationStore(
                self.evaluator.evaluations_dir / error.evaluation_id
            ).load()
            await self._record(failure)
            if charged:
                await self.budget_ledger.refund(
                    backend_id,
                    request.evaluation_set,
                    cost,
                    principal,
                )
            raise
        await self._record(record)
        infrastructure = next(
            (
                diagnostic
                for diagnostic in record.report.diagnostics
                if diagnostic.code == "infrastructure_failure"
            ),
            None,
        )
        if infrastructure is not None:
            if charged:
                await self.budget_ledger.refund(
                    backend_id,
                    request.evaluation_set,
                    cost,
                    principal,
                )
            raise EvaluationInfrastructureError(record.id, infrastructure.message)
        return record, decision

    async def _record(self, record: EvaluationRecord) -> None:
        """Index, persist, and publish a completed evaluation exactly once."""
        async with self._record_lock:
            self.database.add_evaluation(record)
            if self.database_path is not None:
                await asyncio.to_thread(
                    self.database.save_to_file,
                    self.database_path,
                )
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
        principal: EvaluationPrincipal = EvaluationPrincipal.AGENT,
    ) -> EvaluationRecord:
        record, _ = await self._evaluate_record(
            backend_id=backend_id,
            request=request,
            objective_spec=objective_spec,
            authorization=authorization,
            principal=principal,
        )
        return record

    async def evaluate(
        self,
        *,
        backend_id: str,
        request: EvaluationRequest,
        objective_spec: ObjectiveSpec | None = None,
        authorization: EvaluationAuthorization | None = None,
        principal: EvaluationPrincipal = EvaluationPrincipal.AGENT,
    ) -> EvaluationRecord | EvaluationSummary | EvaluationAcknowledgement:
        record, decision = await self._evaluate_record(
            backend_id=backend_id,
            request=request,
            objective_spec=objective_spec,
            authorization=authorization,
            principal=principal,
        )
        return project_evaluation(record, decision.disclosure)
