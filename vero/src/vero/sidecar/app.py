"""Optional FastAPI transport for the canonical Harbor sidecar."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from vero.evaluation import (
    EvaluationBudgetExceeded,
    EvaluationDeniedError,
    EvaluationRequestError,
)
from vero.evaluation.exceptions import EvaluationExecutionError
from vero.models import StrictModel
from vero.sidecar.auth import check_admin_token
from vero.sidecar.session import create_harbor_session_archive
from vero.sidecar.sidecar import (
    EvaluationAccessError,
    EvaluationJobNotFoundError,
    EvaluationJobStatus,
    EvaluationSidecar,
    SidecarEvaluationRequest,
    SidecarEvaluationResult,
    SubmissionDisabledError,
)
from vero.sidecar.transport import CandidateTransferError
from vero.sidecar.verifier import CanonicalVerifier

if TYPE_CHECKING:
    from vero.runtime.wandb import InferenceTelemetryPoller

logger = logging.getLogger(__name__)


class SubmitRequest(StrictModel):
    version: str | None = None


class ScoreBaselineRequest(StrictModel):
    replicates: int = 1


_SESSION_EXPORT_PREFIX = "vero-harbor-export-"


def _sweep_stale_session_exports() -> None:
    """Remove export scratch directories left behind by earlier exports.

    Each export stages its archive in a fresh ``mkdtemp`` directory and removes
    it in the response's background task. That task never runs when the export
    crashes or the sidecar is killed mid-stream, and a sidecar lives for the whole
    run, so the leftovers accumulate until the volume fills and every later export
    fails on ENOSPC with the session still unexported. Scoped to this exact prefix
    inside the temporary root the exports are created in, so nothing else on the
    volume can be caught by it.
    """

    # An hour is far longer than any export takes to stream, and generous on
    # purpose: keeping a dead directory an extra hour costs some disk, sweeping a
    # live one costs the export that is still writing into it.
    cutoff = time.time() - 3600.0
    for stale in Path(tempfile.gettempdir()).glob(f"{_SESSION_EXPORT_PREFIX}*"):
        try:
            if stale.is_symlink() or not stale.is_dir():
                continue
            if stale.stat().st_mtime >= cutoff:
                continue
        except OSError:
            continue
        shutil.rmtree(stale, ignore_errors=True)


def _error(status_code: int, message: str, *, detail: bool = False):
    async def handler(_request, error):
        text = message or str(error)
        # `detail` opts a category into appending the raised message. Use it only
        # where the message states a backend capability the agent must know to
        # fix its own request — never for denials, whose whole point is opacity.
        if detail and message and str(error):
            text = f"{message}: {error}"
        return JSONResponse(status_code=status_code, content={"error": text})

    return handler


def create_app(
    *,
    sidecar: EvaluationSidecar,
    verifier: CanonicalVerifier,
    admin_token: str,
    telemetry: "InferenceTelemetryPoller | None" = None,
) -> FastAPI:
    """Expose agent endpoints and token-gated admin endpoints on one app."""
    if not admin_token.strip():
        raise ValueError("admin_token must not be empty")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = (
            asyncio.create_task(telemetry.run()) if telemetry is not None else None
        )
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="VeRO evaluation sidecar", version="1", lifespan=lifespan)
    app.add_exception_handler(
        EvaluationBudgetExceeded,
        _error(429, "evaluation budget exhausted"),
    )
    app.add_exception_handler(EvaluationDeniedError, _error(403, "evaluation denied"))
    app.add_exception_handler(EvaluationAccessError, _error(403, "evaluation denied"))
    app.add_exception_handler(
        EvaluationRequestError,
        # The agent cannot repair a request it is only told is "invalid": these
        # messages are backend-authored capability statements (an unsupported
        # flag, a fixed limit), so pass them through.
        _error(400, "invalid evaluation request", detail=True),
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
        # Off the event loop like the archive build below: removing a stale
        # export's tree is unbounded filesystem work and must not stall the
        # agent's own requests.
        await asyncio.to_thread(_sweep_stale_session_exports)
        directory = Path(tempfile.mkdtemp(prefix=_SESSION_EXPORT_PREFIX))
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
