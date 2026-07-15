"""Program-neutral evaluator lifecycle."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from vero.evaluation.backend import EvaluationBackend, EvaluationContext
from vero.evaluation.exceptions import EvaluationExecutionError
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
from vero.workspace import Workspace


class Evaluator:
    """Run one backend against a clean candidate version and persist its record."""

    def __init__(
        self,
        *,
        workspace: Workspace,
        sessions_dir: Path,
        session_id: str,
        use_copy: bool = True,
    ):
        self.workspace = workspace
        self.sessions_dir = sessions_dir
        self.session_id = session_id
        self.use_copy = use_copy

    @property
    def evaluations_dir(self) -> Path:
        """Canonical records, stored under the stable historical directory name."""
        return self.sessions_dir / self.session_id / "experiments"

    @property
    def experiments_dir(self) -> Path:
        """Deprecated alias for callers written during schema-v2 development."""
        return self.evaluations_dir

    @asynccontextmanager
    async def _candidate_workspace(
        self,
        candidate_commit: str,
        use_copy: bool,
    ) -> AsyncIterator[Workspace]:
        if use_copy:
            async with self.workspace.temp_copy(
                from_version=candidate_commit
            ) as candidate_workspace:
                yield candidate_workspace
            return

        if await self.workspace.is_dirty():
            raise ValueError("direct evaluation requires a clean workspace")
        async with self.workspace.at(candidate_commit):
            yield self.workspace

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
    ) -> EvaluationRecord:
        report = EvaluationReport(
            status=EvaluationStatus.FAILED,
            diagnostics=[
                EvaluationDiagnostic(
                    code=code,
                    message=message,
                    severity=DiagnosticSeverity.ERROR,
                    phase="evaluation",
                )
            ],
            error=message,
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
            completed_at=datetime.now(),
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
        use_copy: bool | None = None,
    ) -> EvaluationRecord:
        evaluation_id = str(uuid4())
        created_at = datetime.now()
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
                request.candidate.commit,
                self.use_copy if use_copy is None else use_copy,
            ) as candidate_workspace:
                actual_version = await candidate_workspace.current_version()
                if actual_version != request.candidate.commit:
                    raise ValueError(
                        f"candidate workspace is at {actual_version!r}, expected "
                        f"{request.candidate.commit!r}"
                    )
                if await candidate_workspace.is_dirty():
                    raise ValueError("candidate workspace must be clean before evaluation")
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
                completed_at=datetime.now(),
            )
            await store.save(record)
            return record
        except asyncio.CancelledError:
            try:
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
                        message="evaluation was cancelled",
                    )
                )
            finally:
                raise
        except Exception as error:
            code = "evaluation_timeout" if isinstance(error, TimeoutError) else "backend_error"
            message = f"{type(error).__name__}: {error}"
            sanitize_error = getattr(backend, "sanitize_error", None)
            if callable(sanitize_error):
                message = sanitize_error(message)
            try:
                await self._persist_failure(
                    store=store,
                    evaluation_id=evaluation_id,
                    backend_id=backend_id,
                    backend=backend,
                    request=request,
                    objective_spec=objective_spec,
                    created_at=created_at,
                    code=code,
                    message=message,
                )
            except Exception:
                pass
            raise EvaluationExecutionError(evaluation_id, message) from error
