"""FastAPI app for the eval sidecar — the HTTP surface over the (transport-agnostic)
EvaluationSidecar handlers + the admin `finalize` over the Verifier.

Two roles over one app: agent (`/eval`, `/submit`, `/status`; unauthenticated, metered,
redacted) and admin (`/finalize`; bearer-token gated). `vero harbor serve` runs
this under uvicorn in the eval-sidecar container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vero.evaluation.engine import EvalRequest
from vero.exceptions import ExperimentBudgetExceeded, InvalidSplitError
from vero.harbor.auth import check_admin
from vero.harbor.server import KAnonymityError, SubmitDisabledError
from vero.harbor.verifier import NoCandidateError

if TYPE_CHECKING:
    from vero.harbor.server import EvaluationSidecar
    from vero.harbor.verifier import Verifier


class EvalBody(BaseModel):
    dataset_id: str
    split: str
    commit: str | None = None
    sample_ids: list[int] | None = None
    num_samples: int | None = None


class SubmitBody(BaseModel):
    commit: str | None = None


def create_app(
    *,
    sidecar: EvaluationSidecar,
    verifier: Verifier,
    admin_token: str,
) -> FastAPI:
    app = FastAPI(title="vero eval sidecar")

    # Known errors -> agent-facing status codes.
    app.add_exception_handler(
        ExperimentBudgetExceeded,
        lambda r, e: JSONResponse(status_code=429, content={"error": str(e)}),
    )
    app.add_exception_handler(
        InvalidSplitError,
        lambda r, e: JSONResponse(status_code=400, content={"error": str(e)}),
    )
    app.add_exception_handler(
        SubmitDisabledError,
        lambda r, e: JSONResponse(status_code=409, content={"error": str(e)}),
    )
    app.add_exception_handler(
        KAnonymityError,
        lambda r, e: JSONResponse(status_code=400, content={"error": str(e)}),
    )
    app.add_exception_handler(
        NoCandidateError,
        lambda r, e: JSONResponse(status_code=409, content={"error": str(e)}),
    )

    @app.get("/health")
    async def health():
        return {"ok": True}

    # --- agent endpoints (unauthenticated; metered + redacted) ---
    @app.post("/eval")
    async def eval_(body: EvalBody):
        summary = await sidecar.evaluate(EvalRequest(**body.model_dump()), admin=False)
        return summary.to_dict()

    @app.post("/submit")
    async def submit(body: SubmitBody):
        return await sidecar.submit(commit=body.commit)

    @app.get("/status")
    async def status():
        return sidecar.status().to_dict()

    # --- admin endpoints (bearer-token gated) ---
    @app.post("/finalize")
    async def finalize(authorization: str | None = Header(default=None)):
        if not check_admin(authorization, admin_token):
            raise HTTPException(status_code=403, detail="admin token required")
        return await verifier.finalize()

    @app.get("/experiments")
    async def experiments(authorization: str | None = Header(default=None)):
        """Mid-run observability: every recorded experiment, unredacted."""
        if not check_admin(authorization, admin_token):
            raise HTTPException(status_code=403, detail="admin token required")
        return {"experiments": sidecar.list_experiments()}

    return app
