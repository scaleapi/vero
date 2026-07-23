"""Optional FastAPI transport for the canonical Harbor sidecar."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.background import BackgroundTask

from vero.evaluation import (
    EvaluationBudgetExceeded,
    EvaluationDeniedError,
    EvaluationRequestError,
)
from vero.evaluation.exceptions import EvaluationExecutionError
from vero.harbor.auth import check_admin_token
from vero.harbor.session import create_harbor_session_archive
from vero.harbor.sidecar import (
    EvaluationAccessError,
    EvaluationJobNotFoundError,
    EvaluationJobStatus,
    EvaluationSidecar,
    SidecarEvaluationRequest,
    SidecarEvaluationResult,
    SubmissionDisabledError,
)
from vero.harbor.transport import CandidateTransferError
from vero.harbor.verifier import CanonicalVerifier

logger = logging.getLogger(__name__)


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str | None = None


class ScoreBaselineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replicates: int = 1


def _error(status_code: int, message: str):
    async def handler(_request, error):
        return JSONResponse(
            status_code=status_code,
            content={"error": message or str(error)},
        )

    return handler


def create_app(
    *,
    sidecar: EvaluationSidecar,
    verifier: CanonicalVerifier,
    admin_token: str,
) -> FastAPI:
    """Expose agent endpoints and token-gated admin endpoints on one app."""
    if not admin_token.strip():
        raise ValueError("admin_token must not be empty")
    app = FastAPI(title="VeRO evaluation sidecar", version="1")
    app.add_exception_handler(
        EvaluationBudgetExceeded,
        _error(429, "evaluation budget exhausted"),
    )
    app.add_exception_handler(EvaluationDeniedError, _error(403, "evaluation denied"))
    app.add_exception_handler(EvaluationAccessError, _error(403, "evaluation denied"))
    app.add_exception_handler(
        EvaluationRequestError, _error(400, "invalid evaluation request")
    )
    app.add_exception_handler(
        CandidateTransferError,
        _error(400, "candidate version could not be imported"),
    )
    app.add_exception_handler(
        SubmissionDisabledError,
        _error(409, "candidate submission is disabled"),
    )
    app.add_exception_handler(
        EvaluationJobNotFoundError,
        _error(404, "evaluation job not found"),
    )

    @app.exception_handler(EvaluationExecutionError)
    async def evaluation_failure(_request, error: EvaluationExecutionError):
        return JSONResponse(
            status_code=502,
            content={
                "error": "evaluation failed",
                "evaluation_id": error.evaluation_id,
            },
        )

    def require_admin(authorization: str | None) -> None:
        if not check_admin_token(authorization, admin_token):
            raise HTTPException(status_code=403, detail="admin token required")

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/eval")
    async def evaluate(body: SidecarEvaluationRequest):
        return await sidecar.evaluate(body)

    @app.post("/eval/jobs", status_code=202)
    async def start_evaluation_job(body: SidecarEvaluationRequest):
        return await sidecar.start_evaluation_job(body)

    @app.get("/eval/jobs/{job_id}")
    async def evaluation_job(job_id: str):
        return sidecar.evaluation_job(job_id)

    @app.get("/eval/jobs/{job_id}/result")
    async def evaluation_job_result(job_id: str):
        job = sidecar.evaluation_job(job_id)
        if job.receipt is None:
            status_code = (
                202
                if job.status
                in {EvaluationJobStatus.QUEUED, EvaluationJobStatus.RUNNING}
                else 409
            )
            return JSONResponse(
                status_code=status_code,
                content=job.model_dump(mode="json"),
            )
        return SidecarEvaluationResult(
            disclosure=job.receipt.disclosure,
            receipt=job.receipt,
        )

    @app.post("/submit")
    async def submit(body: SubmitRequest):
        return await sidecar.submit(body.version)

    @app.get("/status")
    async def status():
        return sidecar.status()

    @app.post("/finalize")
    async def finalize(authorization: Annotated[str | None, Header()] = None):
        require_admin(authorization)
        return await verifier.finalize()

    @app.post("/score/baseline")
    async def score_baseline(
        body: ScoreBaselineRequest,
        authorization: Annotated[str | None, Header()] = None,
    ):
        require_admin(authorization)
        return await verifier.measure_baseline(replicates=body.replicates)

    @app.get("/evaluations")
    async def evaluations(
        authorization: Annotated[str | None, Header()] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        require_admin(authorization)
        records = sidecar.engine.database.get_evaluations(
            limit=limit,
            offset=offset,
            reverse=True,
        )
        return {"evaluations": records}

    @app.get("/session/export")
    async def export_session(
        authorization: Annotated[str | None, Header()] = None,
    ):
        require_admin(authorization)
        directory = Path(tempfile.mkdtemp(prefix="vero-harbor-export-"))
        archive = directory / "session.tar.gz"
        try:
            await asyncio.to_thread(
                create_harbor_session_archive,
                sidecar.engine.evaluator.session_dir,
                archive,
            )
        except BaseException as error:
            shutil.rmtree(directory, ignore_errors=True)
            # Surface the real cause (admin-only endpoint) rather than a bare 500.
            logger.exception("session export failed")
            if isinstance(error, Exception):
                raise HTTPException(
                    status_code=500, detail=f"session export failed: {error}"
                ) from error
            raise
        return FileResponse(
            archive,
            media_type="application/gzip",
            filename="vero-session.tar.gz",
            background=BackgroundTask(shutil.rmtree, directory, ignore_errors=True),
        )

    return app
