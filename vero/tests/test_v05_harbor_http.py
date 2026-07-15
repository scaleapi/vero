from __future__ import annotations

import json
import stat
from datetime import UTC, datetime
from types import SimpleNamespace

from click.testing import CliRunner
from fastapi.testclient import TestClient

from vero.candidate import Candidate
from vero.cli import main
from vero.evaluation import (
    DisclosureLevel,
    EvaluationAcknowledgement,
    EvaluationDatabase,
    EvaluationRequestError,
)
from vero.harbor.app import create_app
from vero.harbor.auth import (
    check_admin_token,
    read_admin_token,
    write_admin_token,
)
from vero.harbor.sidecar import (
    EvaluationAccessError,
    SidecarEvaluationResult,
    SidecarStatus,
    Submission,
)
from vero.harbor.verifier import VerificationResult


class FakeSidecar:
    def __init__(self):
        self.requests = []
        self.raise_access_error = False
        self.raise_request_error = False
        self.engine = SimpleNamespace(database=EvaluationDatabase(id="session"))

    async def evaluate(self, request):
        if self.raise_access_error:
            raise EvaluationAccessError("private details")
        if self.raise_request_error:
            raise EvaluationRequestError("unknown case")
        self.requests.append(request)
        return SidecarEvaluationResult(
            disclosure=DisclosureLevel.NONE,
            result=EvaluationAcknowledgement(
                evaluation_id="evaluation",
                status="success",
            ),
        )

    async def submit(self, version=None):
        return Submission(
            candidate=Candidate(
                id="candidate",
                version=version or "HEAD",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )

    def status(self):
        return SidecarStatus(submit_enabled=True, evaluation_access=[])


class FakeVerifier:
    def __init__(self):
        self.calls = 0

    async def finalize(self):
        self.calls += 1
        return VerificationResult(rewards={"reward": 0.75})


def test_admin_token_is_atomic_restrictive_and_constant_time_checked(tmp_path):
    path = write_admin_token(tmp_path / "admin/token", "secret-token")

    assert read_admin_token(path) == "secret-token"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert check_admin_token("Bearer secret-token", "secret-token")
    assert not check_admin_token("Bearer wrong", "secret-token")
    assert not check_admin_token(None, "secret-token")


def test_http_app_separates_agent_and_admin_surfaces():
    sidecar = FakeSidecar()
    verifier = FakeVerifier()
    client = TestClient(
        create_app(
            sidecar=sidecar,
            verifier=verifier,
            admin_token="admin-secret",
        )
    )

    assert client.get("/health").json() == {"ok": True}
    response = client.post(
        "/eval",
        json={
            "backend_id": "backend",
            "evaluation_set": {"name": "public"},
        },
    )
    assert response.status_code == 200
    assert response.json()["disclosure"] == "none"
    assert sidecar.requests[0].evaluation_set.name == "public"
    assert client.get("/status").json()["submit_enabled"] is True
    assert client.post("/finalize").status_code == 403
    finalized = client.post(
        "/finalize",
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert finalized.json()["rewards"] == {"reward": 0.75}
    assert verifier.calls == 1
    assert client.get("/evaluations").status_code == 403
    assert client.get(
        "/evaluations",
        headers={"Authorization": "Bearer admin-secret"},
    ).json() == {"evaluations": []}


def test_http_app_redacts_access_denial_details():
    sidecar = FakeSidecar()
    sidecar.raise_access_error = True
    client = TestClient(
        create_app(
            sidecar=sidecar,
            verifier=FakeVerifier(),
            admin_token="admin-secret",
        )
    )

    response = client.post(
        "/eval",
        json={
            "backend_id": "backend",
            "evaluation_set": {"name": "hidden"},
        },
    )

    assert response.status_code == 403
    assert response.json() == {"error": "evaluation denied"}


def test_http_app_maps_backend_request_rejection_to_400():
    sidecar = FakeSidecar()
    sidecar.raise_request_error = True
    client = TestClient(
        create_app(
            sidecar=sidecar,
            verifier=FakeVerifier(),
            admin_token="admin-secret",
        )
    )

    response = client.post(
        "/eval",
        json={
            "backend_id": "backend",
            "evaluation_set": {"name": "hidden"},
        },
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid evaluation request"}


def test_harbor_cli_builds_canonical_selection(monkeypatch):
    captured = {}

    def fake_request(method, path, *, payload=None, headers=None):
        captured.update(method=method, path=path, payload=payload, headers=headers)
        return {"ok": True}

    monkeypatch.setattr("vero.harbor.cli._request", fake_request)
    result = CliRunner().invoke(
        main,
        [
            "harbor",
            "eval",
            "--backend",
            "backend",
            "--evaluation-set",
            "benchmark",
            "--partition",
            "validation",
            "--case-id",
            "a",
            "--case-id",
            "b",
            "--parameter",
            "temperature=0.2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["method"] == "POST"
    assert captured["path"] == "/eval"
    assert captured["payload"]["evaluation_set"]["selection"] == {
        "kind": "ids",
        "ids": ["a", "b"],
    }
    assert captured["payload"]["parameters"] == {"temperature": 0.2}


def test_harbor_finalize_cli_writes_only_rewards(tmp_path, monkeypatch):
    token_file = write_admin_token(tmp_path / "token", "admin-secret")
    output = tmp_path / "logs/reward.json"

    def fake_request(method, path, *, payload=None, headers=None):
        assert headers == {"Authorization": "Bearer admin-secret"}
        return {
            "rewards": {"accuracy": 0.9},
            "baseline_rewards": {"accuracy": 0.7},
        }

    monkeypatch.setattr("vero.harbor.cli._request", fake_request)
    result = CliRunner().invoke(
        main,
        [
            "harbor",
            "finalize",
            "--token-file",
            str(token_file),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text()) == {"accuracy": 0.9}
