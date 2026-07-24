"""Authorization, budget, backend, persistence, and disclosure boundary."""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

from vero.evaluation.backend import BackendRegistry
from vero.evaluation.budget import BudgetLedger
from vero.evaluation.error_taxonomy import TERMINATING_DIAGNOSTIC_CODES
from vero.evaluation.evaluator import Evaluator
from vero.evaluation.exceptions import (
    EvaluationCancelledError,
    EvaluationDeniedError,
    EvaluationExecutionError,
    EvaluationInfrastructureError,
    EvaluationRequestError,
    EvaluationTerminatedError,
)
from vero.evaluation.models import (
    AgentSelectionMode,
    EvaluationAcknowledgement,
    EvaluationAuthorization,
    EvaluationPlan,
    EvaluationPrincipal,
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
            and request.evaluation_set.selection != definition.evaluation_set.selection
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
        self._agent_evaluations_open = True
        self._active_agent_evaluations: dict[object, asyncio.Task[object]] = {}
        self._agent_evaluations_idle = asyncio.Event()
        self._agent_evaluations_idle.set()
        self._agent_evaluation_scope_depth: ContextVar[int] = ContextVar(
            f"vero_agent_evaluation_scope_{id(self)}",
            default=0,
        )

    def _begin_evaluation(
        self,
        principal: EvaluationPrincipal,
    ) -> object | None:
        """Atomically admit and track an agent evaluation on this event loop."""

        if principal != EvaluationPrincipal.AGENT:
            return None
        if not self._agent_evaluations_open:
            raise EvaluationDeniedError("evaluation finalization has started")
        task = asyncio.current_task()
        if task is None:  # pragma: no cover - async entry points always have a task
            raise RuntimeError("evaluation requires an active asyncio task")
        token = object()
        self._active_agent_evaluations[token] = task
        self._agent_evaluations_idle.clear()
        return token

    def _finish_evaluation(self, token: object | None) -> None:
        if token is None:
            return
        self._active_agent_evaluations.pop(token, None)
        if not self._active_agent_evaluations:
            self._agent_evaluations_idle.set()

    @asynccontextmanager
    async def agent_evaluation_scope(self) -> AsyncIterator[None]:
        """Track a complete agent request, including candidate import and disclosure."""

        depth = self._agent_evaluation_scope_depth.get()
        if depth:
            nested = self._agent_evaluation_scope_depth.set(depth + 1)
            try:
                yield
            finally:
                self._agent_evaluation_scope_depth.reset(nested)
            return

        token = self._begin_evaluation(EvaluationPrincipal.AGENT)
        outer = self._agent_evaluation_scope_depth.set(1)
        try:
            yield
        finally:
            self._agent_evaluation_scope_depth.reset(outer)
            self._finish_evaluation(token)

    async def quiesce_agent_evaluations(
        self,
        *,
        timeout_seconds: float,
        cancellation_grace_seconds: float = 30.0,
    ) -> int:
        """Close agent admission and drain requests accepted before finalization.

        The admission close and active-task snapshot contain no suspension point,
        so a new agent evaluation cannot slip between them. If the bounded wait
        expires, the remaining requests are cancelled; the normal evaluator
        cancellation path persists terminal records and refunds their budgets.
        Admin and system evaluations remain available to the verifier.
        """

        if timeout_seconds <= 0:
            raise ValueError("evaluation drain timeout must be positive")
        if cancellation_grace_seconds <= 0:
            raise ValueError("evaluation cancellation grace must be positive")

        self._agent_evaluations_open = False
        admitted = len(self._active_agent_evaluations)
        if not admitted:
            return 0

        try:
            async with asyncio.timeout(timeout_seconds):
                await self._agent_evaluations_idle.wait()
            return admitted
        except TimeoutError:
            pending = set(self._active_agent_evaluations.values())
            logger.warning(
                "Cancelling %d agent evaluation(s) after a %.1fs finalization drain",
                len(pending),
                timeout_seconds,
            )
            for task in pending:
                task.cancel()
            try:
                async with asyncio.timeout(cancellation_grace_seconds):
                    await asyncio.gather(*pending, return_exceptions=True)
                    await self._agent_evaluations_idle.wait()
            except TimeoutError:
                logger.error(
                    "%d agent evaluation(s) did not stop within the %.1fs "
                    "cancellation grace",
                    len(self._active_agent_evaluations),
                    cancellation_grace_seconds,
                )
            return admitted

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
        token = (
            None
            if principal == EvaluationPrincipal.AGENT
            and self._agent_evaluation_scope_depth.get()
            else self._begin_evaluation(principal)
        )
        try:
            return await self._execute_record(
                backend_id=backend_id,
                request=request,
                objective_spec=objective_spec,
                authorization=authorization,
                principal=principal,
            )
        finally:
            self._finish_evaluation(token)

    async def _execute_record(
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
        terminating = next(
            (
                diagnostic
                for diagnostic in record.report.diagnostics
                if diagnostic.code in TERMINATING_DIAGNOSTIC_CODES
            ),
            None,
        )
        if terminating is not None:
            # A terminating condition (inference-budget exhaustion or auth
            # failure) will not heal: do not refund and do not retry.
            raise EvaluationTerminatedError(record.id, terminating.message)
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
