import os
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vero.core.db.candidate import Candidate
from vero.evaluation import (
    AllCases,
    CaseIds,
    CaseRange,
    CaseCheckpointStore,
    CommandBackend,
    CommandBackendConfig,
    EvaluationContext,
    EvaluationLimits,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
)
from vero.sandbox import LocalSandbox


def _write_harness(harness_root: Path, body: str) -> Path:
    harness_root.mkdir(parents=True)
    script = harness_root / "harness.py"
    script.write_text(body)
    return script


async def _context(tmp_path: Path, workspace_path: Path) -> EvaluationContext:
    workspace_path.mkdir(parents=True, exist_ok=True)
    sandbox = await LocalSandbox.create(root=tmp_path)
    workspace = SimpleNamespace(
        project_path=str(workspace_path),
        root=str(workspace_path),
        sandbox=sandbox,
    )
    result_dir = tmp_path / "result"
    artifact_dir = result_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    return EvaluationContext(
        workspace=workspace,
        session_id="session",
        evaluation_id="evaluation",
        result_dir=result_dir,
        artifact_dir=artifact_dir,
        case_store=CaseCheckpointStore(result_dir / "cases"),
    )


def _request(selection=None) -> EvaluationRequest:
    return EvaluationRequest(
        candidate=Candidate(commit="abc", repo_name="target"),
        evaluation_set=EvaluationSet(
            name="performance",
            selection=selection or AllCases(),
        ),
        parameters={"size": 10},
        seed=42,
    )


@pytest.mark.asyncio
async def test_command_backend_uses_versioned_json_without_shell_interpretation(
    tmp_path: Path,
):
    harness_root = tmp_path / "trusted-harness"
    script = _write_harness(
        harness_root,
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--workspace")
parser.add_argument("--request")
parser.add_argument("--report")
parser.add_argument("--artifacts")
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
assert request["schema_version"] == "1"
assert request["request"]["parameters"] == {"size": 10}
assert "authorization" not in request
Path(args.report).write_text(json.dumps({
    "schema_version": "1",
    "status": "success",
    "metrics": {"latency_ms": 1.25, "correct": 1.0},
    "cases": [],
    "diagnostics": [],
    "artifacts": [],
    "error": None,
}))
print(args.workspace)
""",
    )
    workspace_path = tmp_path / "target;touch SHOULD_NOT_EXIST"
    context = await _context(tmp_path, workspace_path)
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=str(harness_root),
            command=[
                sys.executable,
                str(script),
                "--workspace",
                "{workspace}",
                "--request",
                "{request}",
                "--report",
                "{report}",
                "--artifacts",
                "{artifacts}",
            ],
        )
    )

    report = await backend.evaluate(context=context, request=_request())

    assert report.status == EvaluationStatus.SUCCESS
    assert report.metrics == {"latency_ms": 1.25, "correct": 1.0}
    assert [artifact.path for artifact in report.artifacts] == [
        "command/stdout.log",
        "command/stderr.log",
    ]
    assert str(workspace_path) in (
        context.artifact_dir / "command" / "stdout.log"
    ).read_text()
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


@pytest.mark.asyncio
async def test_command_backend_passes_only_configured_environment(tmp_path: Path):
    harness_root = tmp_path / "harness"
    script = _write_harness(
        harness_root,
        """
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "1",
    "status": "success",
    "metrics": {
        "configured": float(os.environ["CONFIGURED"]),
        "passed": float(os.environ["PASSED"]),
        "hidden_absent": float("HIDDEN" not in os.environ),
    },
}))
""",
    )
    context = await _context(tmp_path, tmp_path / "target")
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

    report = await backend.evaluate(context=context, request=_request())

    assert report.metrics == {
        "configured": 1.0,
        "passed": 2.0,
        "hidden_absent": 1.0,
    }


@pytest.mark.asyncio
async def test_command_backend_redacts_configured_secrets_before_persistence(
    tmp_path: Path,
):
    harness_root = tmp_path / "harness"
    secret = "highly-sensitive-token"
    script = _write_harness(
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
    "schema_version": "1",
    "status": "failed",
    "diagnostics": [{
        "code": "harness_failed",
        "message": f"diagnostic leaked {secret}",
        "severity": "error",
    }],
    "error": f"report leaked {secret}",
}))
""",
    )
    context = await _context(tmp_path, tmp_path / "target")
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=str(harness_root),
            command=[sys.executable, str(script), "{report}"],
            environment={"SECRET_TOKEN": secret},
        )
    )

    report = await backend.evaluate(context=context, request=_request())

    serialized = report.model_dump_json()
    stdout = (context.artifact_dir / "command" / "stdout.log").read_text()
    stderr = (context.artifact_dir / "command" / "stderr.log").read_text()
    assert secret not in serialized
    assert secret not in stdout
    assert secret not in stderr
    assert "[REDACTED]" in serialized
    assert "[REDACTED]" in stdout
    assert "[REDACTED]" in stderr
    secret_request = _request().model_copy(
        update={"parameters": {"token": secret}}
    )
    with pytest.raises(ValueError, match="must not contain configured secret"):
        backend.validate_request(secret_request)


@pytest.mark.parametrize(
    ("body", "expected_code"),
    [
        ("raise SystemExit(7)", "command_failed"),
        ("print('no report')", "missing_report"),
        (
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{')",
            "invalid_report",
        ),
    ],
)
@pytest.mark.asyncio
async def test_command_failures_return_canonical_failed_reports(
    tmp_path: Path, body: str, expected_code: str
):
    harness_root = tmp_path / "harness"
    script = _write_harness(harness_root, body)
    context = await _context(tmp_path, tmp_path / "target")
    command = [sys.executable, str(script)]
    if expected_code == "invalid_report":
        command.append("{report}")
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=str(harness_root),
            command=command,
        )
    )

    report = await backend.evaluate(context=context, request=_request())

    assert report.status == EvaluationStatus.FAILED
    assert report.diagnostics[0].code == expected_code
    assert report.error
    assert [artifact.path for artifact in report.artifacts] == [
        "command/stdout.log",
        "command/stderr.log",
    ]


@pytest.mark.asyncio
async def test_command_report_validation_rejects_unsafe_artifact(tmp_path: Path):
    harness_root = tmp_path / "harness"
    script = _write_harness(
        harness_root,
        """
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "1",
    "status": "success",
    "artifacts": [{"path": "../escape"}],
}))
""",
    )
    context = await _context(tmp_path, tmp_path / "target")
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=str(harness_root),
            command=[sys.executable, str(script), "{report}"],
        )
    )

    report = await backend.evaluate(context=context, request=_request())

    assert report.status == EvaluationStatus.FAILED
    assert report.diagnostics[0].code == "invalid_report"


@pytest.mark.parametrize(
    ("selection", "expected_cases"),
    [
        (AllCases(), None),
        (CaseIds(ids=["a", "b"]), 2),
        (CaseRange(stop=5), 5),
        (CaseRange(start=5, stop=9), 4),
    ],
)
@pytest.mark.asyncio
async def test_command_backend_resolves_known_and_unknown_costs(
    tmp_path: Path, selection, expected_cases
):
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=str(tmp_path),
            command=["run"],
        )
    )

    cost = await backend.resolve_cost(EvaluationSet(selection=selection))

    assert cost.runs == 1
    assert cost.cases == expected_cases


def test_command_config_rejects_untrusted_path_and_argv_shapes(tmp_path: Path):
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


def test_command_provenance_does_not_depend_on_passthrough_runtime_value(
    tmp_path: Path, monkeypatch
):
    config = CommandBackendConfig(
        harness_root=str(tmp_path),
        command=["run"],
        passthrough_environment=["RUNTIME_TOKEN"],
    )
    backend = CommandBackend(config)
    monkeypatch.setenv("RUNTIME_TOKEN", "first-secret")
    first = backend.provenance
    monkeypatch.setenv("RUNTIME_TOKEN", "second-secret")
    second = backend.provenance

    assert first == second
    assert "first-secret" not in first.model_dump_json()
    assert "second-secret" not in second.model_dump_json()


@pytest.mark.asyncio
async def test_command_harness_cannot_live_inside_editable_target(tmp_path: Path):
    target = tmp_path / "target"
    harness = target / "harness"
    harness.mkdir(parents=True)
    context = await _context(tmp_path, target)
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=str(harness),
            command=["run"],
        )
    )

    with pytest.raises(ValueError, match="outside the editable target"):
        await backend.evaluate(context=context, request=_request())


@pytest.mark.asyncio
async def test_command_timeout_terminates_harness_process_tree(tmp_path: Path):
    harness_root = tmp_path / "harness"
    sentinel = tmp_path / "child-survived"
    script = _write_harness(
        harness_root,
        """
import subprocess
import sys
import time

subprocess.Popen([
    sys.executable,
    "-c",
    "import pathlib,sys,time; time.sleep(2); pathlib.Path(sys.argv[1]).write_text('alive')",
    sys.argv[1],
])
time.sleep(20)
""",
    )
    context = await _context(tmp_path, tmp_path / "target")
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=str(harness_root),
            command=[sys.executable, str(script), str(sentinel)],
        )
    )
    request = _request().model_copy(
        update={"limits": EvaluationLimits(timeout_seconds=1)}
    )

    report = await backend.evaluate(context=context, request=request)
    await asyncio.sleep(1.5)

    assert report.status == EvaluationStatus.FAILED
    assert report.diagnostics[0].code == "command_timeout"
    assert not sentinel.exists()
