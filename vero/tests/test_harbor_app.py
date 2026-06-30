"""Tests for vero.harbor.app — FastAPI routes + agent/admin auth."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
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


class TestServeIntegrityGuards:
    @pytest.mark.asyncio
    async def test_mode_a_without_task_project_rejected(self, tmp_path):
        # Mode A (no `harbor`) with task_project unset would load the scorer from
        # the agent repo. build_components must refuse to start.
        from vero.harbor.serve import ServeConfig, build_components

        cfg = ServeConfig(
            repo_path=str(tmp_path / "repo"),
            agent_repo_path=str(tmp_path / "agent"),
            session_id="s",
            dataset_id="ds",
            split_accesses=[{"split": "test", "access": "no_access"}],
            budgets=[{"split": "validation", "dataset_id": "ds", "total_run_budget": 1}],
            task="math",
            task_project=None,  # the vulnerability
            agent_volume=str(tmp_path / "agent_vol"),
            admin_volume=str(tmp_path / "admin_vol"),
            admin_token_path=str(tmp_path / "admin_vol" / "token"),
        )
        with pytest.raises(ValueError, match="task_project"):
            await build_components(cfg)


class TestNoAccessRejection:
    def test_eval_on_no_access_split_maps_to_400(self):
        # A no_access split is rejected in the engine (InvalidSplitError); the app
        # must surface that as 400 so the agent can never evaluate it. Mirrors the
        # 429 budget-exceeded mapping. The end-to-end engine guarantee (the gate
        # fires before any scoring) is covered by the no_access gate test in
        # test_engine.py on the core PR.
        from vero.exceptions import InvalidSplitError

        sidecar = MagicMock()
        sidecar.evaluate = AsyncMock(side_effect=InvalidSplitError("no_access split"))
        r = _client(sidecar=sidecar).post(
            "/eval", json={"dataset_id": "ds", "split": "test"}
        )
        assert r.status_code == 400
        sidecar.evaluate.assert_awaited_once()


class TestTokenFilePermissions:
    def test_token_file_not_world_or_group_readable(self, tmp_path):
        # The admin token gates /finalize; agent.user (a different uid) must not be
        # able to read it. The portable OS guarantee is the 0o600 mode bit.
        tok = generate_token()
        p = write_admin_token(tmp_path / "token", tok)
        mode = p.stat().st_mode & 0o777
        assert mode == 0o600
        assert mode & 0o077 == 0, "token must not be group/other readable"

    @pytest.mark.skipif(
        os.geteuid() != 0,
        reason="uid-drop read test requires running as root to seteuid to a non-owner",
    )
    def test_non_owner_uid_cannot_read_token(self, tmp_path):
        tok = generate_token()
        p = write_admin_token(tmp_path / "token", tok)
        os.chown(p, 0, 0)  # root-owned, like the real sidecar
        try:
            os.seteuid(65534)  # nobody
            with pytest.raises(PermissionError):
                read_admin_token(p)
        finally:
            os.seteuid(0)
