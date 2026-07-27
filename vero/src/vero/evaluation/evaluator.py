"""Program-neutral evaluator lifecycle."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from vero.candidate import Candidate
from vero.candidate_repository import CandidateRepository
from vero.evaluation.backends.base import EvaluationBackend, EvaluationContext
from vero.evaluation.exceptions import (
    EvaluationCancelledError,
    EvaluationExecutionError,
)
from vero.evaluation.models import (
    CaseStatus,
    DiagnosticSeverity,
    EvaluationDiagnostic,
    EvaluationPrincipal,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationStatus,
    ObjectiveSpec,
)
from vero.evaluation.scoring.objective import evaluate_objective
from vero.evaluation.store.persistence import EvaluationStore
from vero.sandbox import Sandbox
from vero.workspace import Workspace


class Evaluator:
    """Run one backend against a clean candidate snapshot and persist it."""

    def __init__(
        self,
        *,
        candidate_repository: CandidateRepository,
        sandbox: Sandbox,
        session_dir: Path,
        session_id: str | None = None,
    ):
        self.candidate_repository = candidate_repository
        self.sandbox = sandbox
        self.session_dir = session_dir
        self.session_id = session_id or session_dir.name

    @property
    def evaluations_dir(self) -> Path:
        return self.session_dir / "evaluations"

    @asynccontextmanager
    async def _candidate_workspace(
        self,
        candidate: Candidate,
    ) -> AsyncIterator[Workspace]:
        async with self.candidate_repository.checkout(
            candidate,
            sandbox=self.sandbox,
            name=f"vero-evaluation-{candidate.id}",
        ) as workspace:
            yield workspace

    async def _persist_failure(
        self,
        *,
        store: EvaluationStore,
        evaluation_id: str,
        backend_id: str,
        backend: EvaluationBackend,
        request: EvaluationRequest,
        objective_spec: ObjectiveSpec | None,
        principal: EvaluationPrincipal,
        created_at: datetime,
        code: str,
        message: str,
        status: EvaluationStatus = EvaluationStatus.FAILED,
    ) -> EvaluationRecord:
        report = EvaluationReport(
            status=status,
            diagnostics=[
                EvaluationDiagnostic(
                    code=code,
                    message=message,
                    severity=DiagnosticSeverity.ERROR,
                    phase="evaluation",
                )
            ],
        )
        objective = (
            evaluate_objective(report, objective_spec)
            if objective_spec is not None
            else None
        )
        record = EvaluationRecord(
            id=evaluation_id,
            request=request,
            report=report,
            backend_id=backend_id,
            backend=backend.provenance,
            principal=principal,
            objective_spec=objective_spec,
            objective=objective,
            created_at=created_at,
            completed_at=datetime.now(UTC),
        )
        await store.save(record)
        return record

    async def evaluate(
        self,
        *,
        backend_id: str,
        backend: EvaluationBackend,
        request: EvaluationRequest,
        objective_spec: ObjectiveSpec | None = None,
        principal: EvaluationPrincipal = EvaluationPrincipal.SYSTEM,
    ) -> EvaluationRecord:
        evaluation_id = str(uuid4())
        created_at = datetime.now(UTC)
        result_dir = self.evaluations_dir / evaluation_id
        store = EvaluationStore(result_dir)
        result_dir.mkdir(parents=True, exist_ok=False)
        store.artifact_dir.mkdir(parents=True, exist_ok=True)
        store.write_running(
            evaluation_id=evaluation_id,
            request=request,
            backend_id=backend_id,
            backend=backend.provenance,
            objective_spec=objective_spec,
            created_at=created_at,
        )

        try:
            async with self._candidate_workspace(
                request.candidate,
            ) as candidate_workspace:
                actual_version = await candidate_workspace.current_version()
                if actual_version != request.candidate.version:
                    raise ValueError(
                        f"candidate workspace is at {actual_version!r}, expected "
                        f"{request.candidate.version!r}"
                    )
                if await candidate_workspace.is_dirty():
                    raise ValueError(
                        "candidate workspace must be clean before evaluation"
                    )
                context = EvaluationContext(
                    workspace=candidate_workspace,
                    session_id=self.session_id,
                    evaluation_id=evaluation_id,
                    result_dir=result_dir,
                    artifact_dir=store.artifact_dir,
                    case_store=store.cases,
                    finalization=principal == EvaluationPrincipal.ADMIN,
                )
                async with asyncio.timeout(request.limits.timeout_seconds):
                    raw_report = await backend.evaluate(
                        context=context,
                        request=request,
                    )
                report = EvaluationReport.model_validate(raw_report)
                report = self._apply_error_rate_threshold(
                    report,
                    request.limits.error_rate_threshold,
                )

            objective = (
                evaluate_objective(report, objective_spec)
                if objective_spec is not None
                else None
            )
            record = EvaluationRecord(
                id=evaluation_id,
                request=request,
                report=report,
                backend_id=backend_id,
                backend=backend.provenance,
                principal=principal,
                objective_spec=objective_spec,
                objective=objective,
                created_at=created_at,
                completed_at=datetime.now(UTC),
            )
            await store.save(record)
            return record
        except asyncio.CancelledError as error:
            message = "evaluation was cancelled"
            await asyncio.shield(
                self._persist_failure(
                    store=store,
                    evaluation_id=evaluation_id,
                    backend_id=backend_id,
                    backend=backend,
                    request=request,
                    objective_spec=objective_spec,
                    principal=principal,
                    created_at=created_at,
                    code="evaluation_cancelled",
                    message=message,
                    status=EvaluationStatus.CANCELLED,
                )
            )
            raise EvaluationCancelledError(evaluation_id, message) from error
        # The two handlers below shield their persist for the same reason the
        # cancellation handler above does. asyncio.timeout absorbs its own
        # internal cancellation and re-raises as TimeoutError, but an *external*
        # cancel -- quiesce_agent_evaluations draining agent evaluations at
        # finalization -- can still be pending, and the first await inside an
        # unshielded _persist_failure would deliver it, losing the failure record
        # and unwinding as a raw CancelledError instead of the typed error.
        # (EvaluationEngine refunds on that raw path too, belt and braces.)
        except TimeoutError as error:
            message = f"evaluation exceeded {request.limits.timeout_seconds} seconds"
            await asyncio.shield(
                self._persist_failure(
                    store=store,
                    evaluation_id=evaluation_id,
                    backend_id=backend_id,
                    backend=backend,
                    request=request,
                    objective_spec=objective_spec,
                    principal=principal,
                    created_at=created_at,
                    code="evaluation_timeout",
                    message=message,
                )
            )
            raise EvaluationExecutionError(evaluation_id, message) from error
        except Exception as error:
            message = str(error) or type(error).__name__
            await asyncio.shield(
                self._persist_failure(
                    store=store,
                    evaluation_id=evaluation_id,
                    backend_id=backend_id,
                    backend=backend,
                    request=request,
                    objective_spec=objective_spec,
                    principal=principal,
                    created_at=created_at,
                    code="backend_error",
                    message=message,
                )
            )
            raise EvaluationExecutionError(evaluation_id, message) from error

    @staticmethod
    def _apply_error_rate_threshold(
        report: EvaluationReport,
        threshold: float | None,
    ) -> EvaluationReport:
        """Mark a report INVALID when too many cases were lost to infrastructure.

        Only infrastructure cases (``CaseStatus.ERROR``) count: a legitimate
        agent failure is now an informative ``SUCCESS`` at the failure value, so
        it does not push a candidate over the threshold. Crossing the threshold
        means the aggregate is unreliable, not that the candidate is bad, so the
        report becomes ``INVALID`` rather than ``FAILED``."""

        if threshold is None or report.status != EvaluationStatus.SUCCESS:
            return report
        considered = [
            case
            for case in report.cases
            if case.status in (CaseStatus.SUCCESS, CaseStatus.ERROR)
        ]
        if considered:
            error_rate = sum(
                case.status == CaseStatus.ERROR for case in considered
            ) / len(considered)
        else:
            error_rate = report.metrics.get("error_rate")
        if error_rate is None or error_rate < threshold:
            return report
        diagnostic = EvaluationDiagnostic(
            code="infrastructure_invalidity_threshold_exceeded",
            message=(
                f"infrastructure-error rate {error_rate:.6g} reached the configured "
                f"threshold {threshold:.6g}; the aggregate score is unreliable"
            ),
            severity=DiagnosticSeverity.ERROR,
            phase="evaluation",
        )
        return report.model_copy(
            update={
                "status": EvaluationStatus.INVALID,
                "metrics": {**report.metrics, "error_rate": error_rate},
                "diagnostics": [*report.diagnostics, diagnostic],
            }
        )
