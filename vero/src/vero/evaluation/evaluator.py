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
from vero.evaluation.backend import EvaluationBackend, EvaluationContext
from vero.evaluation.exceptions import (
    EvaluationCancelledError,
    EvaluationExecutionError,
)
from vero.evaluation.models import (
    DiagnosticSeverity,
    EvaluationDiagnostic,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationStatus,
    ObjectiveSpec,
)
from vero.evaluation.objective import evaluate_objective
from vero.evaluation.persistence import EvaluationStore
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
                )
                async with asyncio.timeout(request.limits.timeout_seconds):
                    raw_report = await backend.evaluate(
                        context=context,
                        request=request,
                    )
                report = EvaluationReport.model_validate(raw_report)

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
                    created_at=created_at,
                    code="evaluation_cancelled",
                    message=message,
                    status=EvaluationStatus.CANCELLED,
                )
            )
            raise EvaluationCancelledError(evaluation_id, message) from error
        except TimeoutError as error:
            message = f"evaluation exceeded {request.limits.timeout_seconds} seconds"
            await self._persist_failure(
                store=store,
                evaluation_id=evaluation_id,
                backend_id=backend_id,
                backend=backend,
                request=request,
                objective_spec=objective_spec,
                created_at=created_at,
                code="evaluation_timeout",
                message=message,
            )
            raise EvaluationExecutionError(evaluation_id, message) from error
        except Exception as error:
            message = str(error) or type(error).__name__
            await self._persist_failure(
                store=store,
                evaluation_id=evaluation_id,
                backend_id=backend_id,
                backend=backend,
                request=request,
                objective_spec=objective_spec,
                created_at=created_at,
                code="backend_error",
                message=message,
            )
            raise EvaluationExecutionError(evaluation_id, message) from error
