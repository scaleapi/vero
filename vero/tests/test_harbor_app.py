"""Tests for vero.harbor.app — FastAPI routes + agent/admin auth."""

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from vero.exceptions import ExperimentBudgetExceeded
from vero.harbor.app import create_app
from vero.harbor.auth import check_admin, generate_token, read_admin_token, write_admin_token
from vero.harbor.protocol import EvalSummary, StatusSummary
from vero.harbor.server import SubmitDisabledError

TOKEN = "secret-admin-token"


def _client(sidecar=None, verifier=None):
    sidecar = sidecar or MagicMock()
    verifier = verifier or MagicMock()
    return TestClient(create_app(sidecar=sidecar, verifier=verifier, admin_token=TOKEN))


class TestAuthHelpers:
    def test_token_roundtrip_and_perms(self, tmp_path):
        tok = generate_token()
        p = write_admin_token(tmp_path / "t", tok)
        assert read_admin_token(p) == tok
        assert (p.stat().st_mode & 0o777) == 0o600

    def test_check_admin(self):
        assert check_admin(f"Bearer {TOKEN}", TOKEN) is True
        assert check_admin("Bearer wrong", TOKEN) is False
        assert check_admin(None, TOKEN) is False
        assert check_admin(TOKEN, TOKEN) is False  # missing "Bearer "


class TestAgentEndpoints:
    def test_eval(self):
        sidecar = MagicMock()
        sidecar.evaluate = AsyncMock(
            return_value=EvalSummary(
                commit="c1", split="train", dataset_id="ds", n_samples=2,
                mean_score=0.5, result_path="/r", budget_remaining=None,
            )
        )
        r = _client(sidecar=sidecar).post(
            "/eval", json={"dataset_id": "ds", "split": "train", "num_samples": 2}
        )
        assert r.status_code == 200
        assert r.json()["mean_score"] == 0.5
        assert sidecar.evaluate.await_args.kwargs["admin"] is False

    def test_status(self):
        sidecar = MagicMock()
        sidecar.status = MagicMock(
            return_value=StatusSummary(submit_enabled=True, splits=[{"split": "train"}])
        )
        r = _client(sidecar=sidecar).get("/status")
        assert r.status_code == 200 and r.json()["submit_enabled"] is True

    def test_submit_disabled_maps_to_409(self):
        sidecar = MagicMock()
        sidecar.submit = AsyncMock(side_effect=SubmitDisabledError("disabled"))
        r = _client(sidecar=sidecar).post("/submit", json={"commit": "c1"})
        assert r.status_code == 409

    def test_budget_exceeded_maps_to_429(self):
        sidecar = MagicMock()
        sidecar.evaluate = AsyncMock(side_effect=ExperimentBudgetExceeded("no budget"))
        r = _client(sidecar=sidecar).post("/eval", json={"dataset_id": "ds", "split": "train"})
        assert r.status_code == 429


class TestAdminEndpoint:
    def test_finalize_requires_token(self):
        verifier = MagicMock()
        verifier.finalize = AsyncMock(return_value={"reward": 1.0})
        client = _client(verifier=verifier)

        assert client.post("/finalize").status_code == 403  # no token
        assert client.post("/finalize", headers={"Authorization": "Bearer wrong"}).status_code == 403
        verifier.finalize.assert_not_awaited()

        r = client.post("/finalize", headers={"Authorization": f"Bearer {TOKEN}"})
        assert r.status_code == 200 and r.json() == {"reward": 1.0}
        verifier.finalize.assert_awaited_once()
