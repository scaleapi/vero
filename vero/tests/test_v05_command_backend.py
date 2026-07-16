import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vero.candidate import Candidate
from vero.evaluation import (
    AllCases,
    CaseCheckpointStore,
    CaseIds,
    CaseRange,
    CommandBackend,
    CommandBackendConfig,
    EvaluationContext,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
)
from vero.sandbox import LocalSandbox


def write_harness(root: Path, body: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    script = root / "harness.py"
    script.write_text(body)
    return script


async def context(tmp_path: Path, workspace_path: Path) -> EvaluationContext:
    workspace_path.mkdir(parents=True, exist_ok=True)
    result_dir = tmp_path / "result"
    artifact_dir = result_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    workspace = SimpleNamespace(
        project_path=str(workspace_path),
        sandbox=await LocalSandbox.create(root=tmp_path),
    )
    return EvaluationContext(
        workspace=workspace,
        session_id="session",
        evaluation_id="evaluation",
        result_dir=result_dir,
        artifact_dir=artifact_dir,
        case_store=CaseCheckpointStore(result_dir / "cases"),
    )


def request() -> EvaluationRequest:
    return EvaluationRequest(
        candidate=Candidate(
            id="candidate",
            version="version-1",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )


@pytest.mark.asyncio
async def test_command_backend_uses_argv_without_shell_interpolation(tmp_path: Path):
    harness_root = tmp_path / "harness"
    script = write_harness(
        harness_root,
        """
import json
import sys
from pathlib import Path

workspace, request_path, report_path = map(Path, sys.argv[1:])
payload = json.loads(request_path.read_text())
Path(report_path).write_text(json.dumps({
    "schema_version": 1,
    "status": "success",
    "metrics": {
        "workspace_exists": float(workspace.exists()),
        "request_schema": float(payload["schema_version"]),
    },
}))
print(workspace)
""",
    )
    workspace_path = tmp_path / "target;touch SHOULD_NOT_EXIST"
    runtime_context = await context(tmp_path, workspace_path)
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=str(harness_root),
            command=[
                sys.executable,
                str(script),
                "{workspace}",
                "{request}",
                "{report}",
            ],
        )
    )

    report = await backend.evaluate(context=runtime_context, request=request())

    assert report.status == EvaluationStatus.SUCCESS
    assert report.metrics == {"workspace_exists": 1.0, "request_schema": 1.0}
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()
    assert [artifact.path for artifact in report.artifacts] == [
        "command/stdout.log",
        "command/stderr.log",
    ]


@pytest.mark.asyncio
async def test_command_backend_passes_only_declared_environment(tmp_path: Path):
    harness_root = tmp_path / "harness"
    script = write_harness(
        harness_root,
        """
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": "success",
    "metrics": {
        "configured": float(os.environ["CONFIGURED"]),
        "passed": float(os.environ["PASSED"]),
        "hidden_absent": float("HIDDEN" not in os.environ),
    },
}))
""",
    )
    os.environ["PASSED"] = "2"
    os.environ["HIDDEN"] = "3"
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=str(harness_root),
            command=[sys.executable, str(script), "{report}"],
            environment={"CONFIGURED": "1"},
            passthrough_environment=["PASSED"],
        )
    )

    report = await backend.evaluate(
        context=await context(tmp_path, tmp_path / "target"),
        request=request(),
    )

    assert report.metrics == {
        "configured": 1.0,
        "passed": 2.0,
        "hidden_absent": 1.0,
    }


@pytest.mark.asyncio
async def test_command_backend_redacts_secrets(tmp_path: Path):
    harness_root = tmp_path / "harness"
    secret = "highly-sensitive-token"
    script = write_harness(
        harness_root,
        """
import json
import os
import sys
from pathlib import Path

secret = os.environ["SECRET_TOKEN"]
print(f"stdout leaked {secret}")
print(f"stderr leaked {secret}", file=sys.stderr)
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": "failed",
    "diagnostics": [{
        "code": "harness_failed",
        "message": f"diagnostic leaked {secret}",
        "severity": "error"
    }]
}))
""",
    )
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=str(harness_root),
            command=[sys.executable, str(script), "{report}"],
            environment={"SECRET_TOKEN": secret},
        )
    )
    runtime_context = await context(tmp_path, tmp_path / "target")

    report = await backend.evaluate(context=runtime_context, request=request())

    persisted_text = (
        report.model_dump_json()
        + (runtime_context.artifact_dir / "command" / "stdout.log").read_text()
        + (runtime_context.artifact_dir / "command" / "stderr.log").read_text()
    )
    assert secret not in persisted_text
    assert "[REDACTED]" in persisted_text
    with pytest.raises(ValueError, match="must not contain configured secret"):
        backend.validate_request(
            request().model_copy(update={"parameters": {"token": secret}})
        )


@pytest.mark.parametrize(
    ("body", "arguments", "expected_code"),
    [
        ("raise SystemExit(7)", [], "command_failed"),
        ("print('no report')", [], "missing_report"),
        (
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{')",
            ["{report}"],
            "invalid_report",
        ),
    ],
)
@pytest.mark.asyncio
async def test_command_failures_return_failed_reports(
    tmp_path: Path,
    body: str,
    arguments: list[str],
    expected_code: str,
):
    harness_root = tmp_path / "harness"
    script = write_harness(harness_root, body)
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=str(harness_root),
            command=[sys.executable, str(script), *arguments],
        )
    )

    report = await backend.evaluate(
        context=await context(tmp_path, tmp_path / "target"),
        request=request(),
    )

    assert report.status == EvaluationStatus.FAILED
    assert report.diagnostics[0].code == expected_code


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (AllCases(), None),
        (CaseIds(ids=["a", "b"]), 2),
        (CaseRange(stop=5), 5),
        (CaseRange(start=5, stop=9), 4),
    ],
)
@pytest.mark.asyncio
async def test_command_backend_resolves_cost(tmp_path: Path, selection, expected):
    backend = CommandBackend(
        CommandBackendConfig(harness_root=str(tmp_path), command=["run"])
    )

    cost = await backend.resolve_cost(EvaluationSet(selection=selection))

    assert cost.cases == expected


@pytest.mark.asyncio
async def test_command_backend_exports_only_allowlisted_agent_inputs(tmp_path: Path):
    visible = tmp_path / "visible.json"
    visible.write_text('{"visible": true}\n', encoding="utf-8")
    hidden = tmp_path / "hidden.json"
    hidden.write_text('{"hidden": true}\n', encoding="utf-8")
    destination = tmp_path / "context"
    destination.mkdir()
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=str(tmp_path),
            command=["run"],
            staged_inputs={"visible": str(visible), "hidden": str(hidden)},
            agent_context_inputs=["visible"],
        )
    )

    await backend.export_case_resources(
        evaluation_set=EvaluationSet(name="validation"),
        destination=str(destination),
        sandbox=await LocalSandbox.create(root=tmp_path),
    )

    index = json.loads((destination / "index.json").read_text())
    assert index["resources"] == [{"name": "visible", "path": "visible"}]
    assert json.loads((destination / "visible").read_text()) == {"visible": True}
    assert not (destination / "hidden").exists()


def test_command_config_rejects_unsafe_shapes(tmp_path: Path):
    with pytest.raises(ValidationError, match="must be absolute"):
        CommandBackendConfig(harness_root="relative", command=["run"])
    with pytest.raises(ValidationError, match="must not be empty"):
        CommandBackendConfig(harness_root=str(tmp_path), command=[])
    with pytest.raises(ValidationError, match="unknown command placeholders"):
        CommandBackendConfig(
            harness_root=str(tmp_path),
            command=["run", "{secret}"],
        )
    with pytest.raises(ValidationError, match="overlap"):
        CommandBackendConfig(
            harness_root=str(tmp_path),
            command=["run"],
            environment={"TOKEN": "value"},
            passthrough_environment=["TOKEN"],
        )
    with pytest.raises(ValidationError, match="unknown staged inputs"):
        CommandBackendConfig(
            harness_root=str(tmp_path),
            command=["run"],
            agent_context_inputs=["missing"],
        )


@pytest.mark.asyncio
async def test_harness_cannot_live_inside_target(tmp_path: Path):
    target = tmp_path / "target"
    harness = target / "harness"
    harness.mkdir(parents=True)
    backend = CommandBackend(
        CommandBackendConfig(harness_root=str(harness), command=["run"])
    )

    with pytest.raises(ValueError, match="outside the editable target"):
        await backend.evaluate(
            context=await context(tmp_path, target),
            request=request(),
        )
