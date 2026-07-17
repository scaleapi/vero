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
    EvaluationReceipt,
    EvaluationRequestError,
    EvaluationStatus,
)
from vero.harbor.app import create_app
from vero.harbor.cli import _compiled_run_environment, _load_agent_trace, harbor
from vero.harbor.auth import (
    check_admin_token,
    read_admin_token,
    write_admin_token,
)
from vero.harbor.sidecar import (
    EvaluationAccessError,
    EvaluationJobNotFoundError,
    EvaluationJobStatus,
    SidecarEvaluationJob,
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
        self.job = None
        self.engine = SimpleNamespace(database=EvaluationDatabase(id="session"))

    async def evaluate(self, request):
        if self.raise_access_error:
            raise EvaluationAccessError("private details")
        if self.raise_request_error:
            raise EvaluationRequestError("unknown case")
        self.requests.append(request)
        return SidecarEvaluationResult(
            disclosure=DisclosureLevel.NONE,
            receipt=EvaluationReceipt(
                evaluation_id="evaluation",
                status=EvaluationStatus.SUCCESS,
                disclosure=DisclosureLevel.NONE,
                result=EvaluationAcknowledgement(
                    evaluation_id="evaluation",
                    status=EvaluationStatus.SUCCESS,
                ),
                result_path=".vero/evaluations/evaluation/evaluation.json",
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

    async def start_evaluation_job(self, request):
        result = await self.evaluate(request)
        self.job = SidecarEvaluationJob(
            job_id="job-1",
            status=EvaluationJobStatus.COMPLETE,
            backend_id=request.backend_id,
            evaluation_set=request.evaluation_set,
            version=request.version,
            evaluation_id=result.receipt.evaluation_id,
            receipt=result.receipt,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            completed_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        return self.job

    def evaluation_job(self, job_id):
        if self.job is None or self.job.job_id != job_id:
            raise EvaluationJobNotFoundError(job_id)
        return self.job

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
    # Read-only file inside a root-only directory: an unprivileged agent that
    # shares the token volume can neither read the file nor traverse to it.
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert check_admin_token("Bearer secret-token", "secret-token")
    assert not check_admin_token("Bearer wrong", "secret-token")
    assert not check_admin_token(None, "secret-token")


def test_compiled_run_environment_keeps_upstream_credentials_from_agent(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "upstream-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")
    launch = tmp_path / "environment/gateway/launch.json"
    launch.parent.mkdir(parents=True)
    launch.write_text(
        json.dumps(
            {
                "upstream_api_key_source": "OPENAI_API_KEY",
                "upstream_api_key_target": "VERO_INFERENCE_UPSTREAM_API_KEY",
                "upstream_base_url_source": "OPENAI_BASE_URL",
                "upstream_base_url_target": "VERO_INFERENCE_UPSTREAM_BASE_URL",
                "producer_api_key": "producer-scope-token",
                "producer_base_url": "http://inference/scopes/producer/optimizer/v1",
            }
        )
    )

    environment = _compiled_run_environment(tmp_path)

    assert environment["OPENAI_API_KEY"] == "producer-scope-token"
    assert environment["OPENAI_BASE_URL"].startswith("http://inference/")
    assert environment["VERO_INFERENCE_UPSTREAM_API_KEY"] == "upstream-secret"
    assert environment["VERO_INFERENCE_UPSTREAM_BASE_URL"] == (
        "https://provider.example/v1"
    )


def test_codex_jsonl_is_converted_to_a_redacted_producer_trace(tmp_path):
    path = tmp_path / "codex.txt"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "Inspecting."},
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": "env",
                            "aggregated_output": "OPENAI_API_KEY=sk-secretvalue123\n",
                        },
                    }
                ),
            ]
        )
        + "\n"
    )

    trace = _load_agent_trace(path)

    assert trace[0] == {"role": "assistant", "content": "Inspecting."}
    assert trace[-1]["output"] == "OPENAI_API_KEY=[REDACTED]\n"


def test_harbor_run_uses_current_python_and_pinned_harbor_extra(tmp_path, monkeypatch):
    import vero.harbor.build as harbor_build
    import vero.harbor.cli as harbor_cli

    config_path = tmp_path / "build.yaml"
    config_path.write_text("task_name: unused\n")
    config = SimpleNamespace(harbor_requirement="harbor[modal]==0.18.0")
    observed = {}

    def compile_task(_config, output):
        output.mkdir(parents=True)
        return output

    def run(command, *, env):
        observed["command"] = command
        observed["environment"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harbor_build, "load_harbor_build_config", lambda _path: config)
    monkeypatch.setattr(harbor_build, "compile_harbor_task", compile_task)
    monkeypatch.setattr(harbor_cli.shutil, "which", lambda _name: "/usr/bin/uvx")
    monkeypatch.setattr(harbor_cli.subprocess, "run", run)

    result = CliRunner().invoke(
        harbor,
        [
            "run",
            "--config",
            str(config_path),
            "--agent",
            "codex",
            "--environment",
            "modal",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed["command"][:6] == [
        "/usr/bin/uvx",
        "--python",
        harbor_cli.sys.executable,
        "--from",
        "harbor[modal]==0.18.0",
        "harbor",
    ]


def test_http_app_separates_agent_and_admin_surfaces(tmp_path, monkeypatch):
    sidecar = FakeSidecar()
    sidecar.engine.evaluator = SimpleNamespace(session_dir=tmp_path / "session")
    verifier = FakeVerifier()

    def create_archive(_session_dir, destination):
        destination.write_bytes(b"portable-session")
        return destination

    monkeypatch.setattr("vero.harbor.app.create_harbor_session_archive", create_archive)
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
    assert client.get("/session/export").status_code == 403
    exported = client.get(
        "/session/export",
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert exported.content == b"portable-session"
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


def test_http_app_exposes_agent_evaluation_jobs():
    sidecar = FakeSidecar()
    client = TestClient(
        create_app(
            sidecar=sidecar,
            verifier=FakeVerifier(),
            admin_token="admin-secret",
        )
    )

    started = client.post(
        "/eval/jobs",
        json={
            "backend_id": "backend",
            "evaluation_set": {"name": "public"},
        },
    )

    assert started.status_code == 202
    assert started.json()["job_id"] == "job-1"
    assert client.get("/eval/jobs/job-1").json()["status"] == "complete"
    result = client.get("/eval/jobs/job-1/result")
    assert result.status_code == 200
    assert result.json()["receipt"]["evaluation_id"] == "evaluation"
    assert client.get("/eval/jobs/missing").status_code == 404


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
    assert captured["payload"]["limits"] is None


def test_harbor_cli_supports_detached_evaluation_jobs(monkeypatch):
    requests = []

    def fake_request(method, path, *, payload=None, headers=None):
        requests.append((method, path, payload))
        return {"job_id": "job-1", "status": "running"}

    monkeypatch.setattr("vero.harbor.cli._request", fake_request)
    runner = CliRunner()
    started = runner.invoke(
        main,
        [
            "harbor",
            "eval",
            "--backend",
            "backend",
            "--evaluation-set",
            "benchmark",
            "--detach",
        ],
    )
    status = runner.invoke(main, ["harbor", "eval-status", "job-1"])
    result = runner.invoke(main, ["harbor", "eval-result", "job-1"])

    assert started.exit_code == 0, started.output
    assert status.exit_code == 0, status.output
    assert result.exit_code == 0, result.output
    assert requests[0][:2] == ("POST", "/eval/jobs")
    assert requests[1][:2] == ("GET", "/eval/jobs/job-1")
    assert requests[2][:2] == ("GET", "/eval/jobs/job-1/result")


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


def test_harbor_export_session_persists_archive_report_and_checksum(
    tmp_path, monkeypatch
):
    import vero.harbor.cli as harbor_cli
    import vero.report as report_module

    token_file = write_admin_token(tmp_path / "token", "admin-secret")
    output = tmp_path / "logs/session.tar.gz"
    report = tmp_path / "logs/experiment.html"
    status_output = tmp_path / "logs/status.json"
    finalization_output = tmp_path / "logs/finalization.json"
    trace = tmp_path / "trajectory.json"
    trace.write_text("[]\n")

    def fake_request(method, path, *, payload=None, headers=None):
        if path == "/finalize":
            assert method == "POST"
            assert headers == {"Authorization": "Bearer admin-secret"}
            return {"candidate": None, "rewards": {"reward": 0.0}, "errors": {}}
        assert (method, path) == ("GET", "/status")
        return {"submit_enabled": False, "evaluation_access": []}

    def fake_download(path, destination, *, headers=None):
        assert path == "/session/export"
        assert headers == {"Authorization": "Bearer admin-secret"}
        destination.write_bytes(b"sidecar archive")

    def fake_extract(_archive, destination):
        session = destination / "session"
        session.mkdir(parents=True)
        (session / "harbor-session.json").write_text("{}\n")
        return session

    async def fake_report(_session, destination):
        destination.write_text("<html>experiment</html>\n")
        return destination

    monkeypatch.setattr(harbor_cli, "_request", fake_request)
    monkeypatch.setattr(harbor_cli, "_download", fake_download)
    monkeypatch.setattr(harbor_cli, "extract_harbor_session_archive", fake_extract)
    monkeypatch.setattr(report_module, "generate_experiment_report", fake_report)

    result = CliRunner().invoke(
        main,
        [
            "harbor",
            "export-session",
            "--token-file",
            str(token_file),
            "--output",
            str(output),
            "--report-output",
            str(report),
            "--status-output",
            str(status_output),
            "--finalization-output",
            str(finalization_output),
            "--agent-trace",
            str(trace),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert report.read_text() == "<html>experiment</html>\n"
    assert json.loads(status_output.read_text())["submit_enabled"] is False
    assert json.loads(finalization_output.read_text())["rewards"] == {"reward": 0.0}
    checksum = output.with_name(f"{output.name}.sha256").read_text()
    assert output.name in checksum
