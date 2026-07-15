"""Optional FastAPI transport for the canonical Harbor sidecar."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from vero.evaluation import (
    EvaluationBudgetExceeded,
    EvaluationDeniedError,
    EvaluationRequestError,
)
from vero.evaluation.exceptions import EvaluationExecutionError
from vero.harbor.auth import check_admin_token
from vero.harbor.sidecar import (
    EvaluationAccessError,
    EvaluationSidecar,
    SidecarEvaluationRequest,
    SubmissionDisabledError,
)
from vero.harbor.transport import CandidateTransferError
from vero.harbor.verifier import CanonicalVerifier


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str | None = None


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

    return app
