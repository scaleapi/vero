from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from click.testing import CliRunner

from vero.candidate import Candidate
from vero.candidate_repository import GitCandidateRepository
from vero.cli import main
from vero.evaluation import (
    BackendProvenance,
    EvaluationArtifact,
    EvaluationDatabase,
    EvaluationPlan,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
)
from vero.runtime import (
    OptimizationComponentSpec,
    OptimizationRunSpec,
    RuntimeEvent,
    SessionManifest,
    SessionStatus,
)
from vero.sandbox import LocalSandbox
from vero.workspace import GitWorkspace


def git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


async def build_session(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    program = source / "program.py"
    program.write_text("score = 1\n", encoding="utf-8")
    git(source, "init", "-b", "main")
    git(source, "add", "--all")
    git(
        source,
        "-c",
        "user.name=vero",
        "-c",
        "user.email=vero@localhost",
        "commit",
        "-m",
        "baseline",
    )

    sandbox = await LocalSandbox.create(root=root)
    workspace = await GitWorkspace.from_path(sandbox, str(source))
    session_dir = root / "session"
    repository = await GitCandidateRepository.create(
        session_dir / "candidates", workspace=workspace
    )
    baseline = Candidate.from_version(
        git(source, "rev-parse", "HEAD"), candidate_id="baseline"
    )
    await repository.capture(baseline, workspace)

    program.write_text("score = 2\n", encoding="utf-8")
    version = await workspace.save("improve score")
    proposal_id = "proposal-1"
    candidate = Candidate.from_version(
        version,
        candidate_id="candidate-1",
        parent_id=baseline.id,
        description="Improve </script><script>alert('unsafe')</script>",
        metadata={"proposal_id": proposal_id, "producer_id": "test"},
    )
    await repository.capture(candidate, workspace)

    objective = ObjectiveSpec(selector=MetricSelector(metric="score"), direction="maximize")
    provenance = BackendProvenance.from_config(name="test", version="1", config={})
    created = datetime(2026, 1, 1, tzinfo=UTC)
    database = EvaluationDatabase(id="report-test")
    for index, (evaluated, value) in enumerate(((baseline, 1.0), (candidate, 2.0))):
        evaluation_id = f"evaluation-{index}"
        record = EvaluationRecord(
            id=evaluation_id,
            request=EvaluationRequest(
                candidate=evaluated,
                evaluation_set=EvaluationSet(name="development"),
            ),
            report=EvaluationReport(
                status=EvaluationStatus.SUCCESS,
                metrics={"score": value},
                artifacts=(
                    [
                        EvaluationArtifact(
                            path="preview.svg",
                            media_type="image/svg+xml",
                            description="Program output",
                        )
                    ]
                    if evaluated is candidate
                    else []
                ),
            ),
            backend_id="test",
            backend=provenance,
            objective_spec=objective,
            objective=ObjectiveResult(value=value, feasible=True),
            created_at=created + timedelta(minutes=index),
            completed_at=created + timedelta(minutes=index, seconds=1),
        )
        database.add_evaluation(record)
        if evaluated is candidate:
            artifact_dir = session_dir / "evaluations" / evaluation_id / "artifacts"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "preview.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="5" cy="5" r="4"/></svg>',
                encoding="utf-8",
            )
    database.save_to_file(session_dir / "database.json")

    component = OptimizationComponentSpec(type="test", config_digest="0" * 64)
    manifest = SessionManifest(
        id="report-test",
        status=SessionStatus.COMPLETED,
        backend_id="test",
        backend=provenance,
        candidate_repository_family="git",
        candidate_repository_format_version=1,
        evaluation_plan=EvaluationPlan.single(EvaluationSet(name="development")),
        objective=objective,
        run=OptimizationRunSpec(
            max_proposals=1,
            max_rounds=1,
            max_concurrency=1,
            strategy=component,
            producers={"test": component},
        ),
        baseline=baseline,
        best_candidate_id=candidate.id,
        best_evaluation_id="evaluation-1",
        created_at=created,
        updated_at=created + timedelta(minutes=2),
    )
    (session_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    event = RuntimeEvent(
        session_id=manifest.id,
        kind="evaluation_completed",
        created_at=created + timedelta(minutes=1),
        payload={"evaluation_id": "evaluation-1", "objective/value": 2.0},
    )
    (session_dir / "events.jsonl").write_text(
        event.model_dump_json() + "\n", encoding="utf-8"
    )
    trace_id = hashlib.sha256(proposal_id.encode()).hexdigest()[:16]
    trace_dir = session_dir / "artifacts" / "agents" / trace_id
    trace_dir.mkdir(parents=True)
    (trace_dir / "trace.json").write_text(
        json.dumps(
            [
                {"role": "user", "content": "Improve the score"},
                {
                    "type": "function_call",
                    "name": "edit",
                    "arguments": '{"path":"program.py"}',
                },
            ]
        ),
        encoding="utf-8",
    )
    return session_dir


def report_payload(html: str) -> dict[str, object]:
    opening = '<script id="report-data" type="application/json">'
    encoded = html.split(opening, 1)[1].split("</script>", 1)[0]
    return json.loads(encoded)


def test_report_command_builds_portable_full_experiment_view(tmp_path: Path):
    session_dir = asyncio.run(build_session(tmp_path))
    output = tmp_path / "report.html"

    result = CliRunner().invoke(
        main, ["report", str(session_dir), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert str(output) in result.output
    html = output.read_text(encoding="utf-8")
    payload = report_payload(html)
    assert len(payload["candidates"]) == 2
    assert len(payload["evaluations"]) == 2
    assert payload["candidates"][1]["trace_id"] is not None
    assert "+score = 2" in payload["candidates"][1]["diff"]["text"]
    assert payload["evaluations"][1]["artifacts"][0]["kind"] == "image"
    assert payload["evaluations"][1]["artifacts"][0]["content"].startswith(
        "data:image/svg+xml;base64,"
    )
    assert payload["traces"][0]["entries"][1]["kind"] == "tool-call"
    assert payload["events"][0]["kind"] == "evaluation_completed"
    assert "</script><script>alert('unsafe')</script>" not in html
    assert "Score trajectories by split" in html
    assert "Lines connect exact case selections only" in html
    assert "Comparable baseline" in html


def test_report_command_fails_clearly_without_a_manifest(tmp_path: Path):
    result = CliRunner().invoke(main, ["report", str(tmp_path)])

    assert result.exit_code == 1
    assert "session manifest not found" in result.output
