from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vero.evaluation import (
    CommandBackend,
    CommandBackendConfig,
    EvaluationSet,
    MetricSelector,
    ObjectiveSpec,
    allow_all_evaluations,
)
from vero.optimization import (
    CommandCandidateProducer,
    CommandCandidateProducerConfig,
)
from vero.runtime import (
    SessionStatus,
    create_local_optimization_session,
    create_optimization_session,
)
from vero.sandbox import LocalSandbox
from vero.workspace import GitWorkspace


def initialize_repository(path: Path) -> str:
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "--all"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=vero",
            "-c",
            "user.email=vero@localhost",
            "commit",
            "-m",
            "baseline",
        ],
        cwd=path,
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def command_components(tmp_path: Path):
    harness = tmp_path / "harness"
    harness.mkdir()
    harness_script = harness / "evaluate.py"
    harness_script.write_text(
        """
import json
import sys
from pathlib import Path

workspace, report_path = map(Path, sys.argv[1:])
value = (workspace / "program.txt").read_text().strip()
score = {"improved": 1.0, "improved-again": 2.0}.get(value, 0.0)
report_path.write_text(json.dumps({
    "schema_version": 1,
    "status": "success",
    "metrics": {"score": score},
}))
""",
        encoding="utf-8",
    )
    producer_root = tmp_path / "producer"
    producer_root.mkdir()
    producer_script = producer_root / "improve.py"
    producer_script.write_text(
        """
import sys
from pathlib import Path
value = "improved" if sys.argv[2] == "0" else "improved-again"
Path(sys.argv[1], "program.txt").write_text(value + "\\n")
""",
        encoding="utf-8",
    )
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=str(harness),
            command=[
                sys.executable,
                str(harness_script),
                "{workspace}",
                "{report}",
            ],
        )
    )
    producer = CommandCandidateProducer(
        CommandCandidateProducerConfig(
            root=str(producer_root),
            command=[
                sys.executable,
                str(producer_script),
                "{workspace}",
                "{round}",
            ],
            description="Improve the program",
        )
    )
    return backend, producer


@pytest.mark.asyncio
async def test_generic_factory_accepts_a_provisioned_workspace(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("baseline\n", encoding="utf-8")
    initialize_repository(target)
    backend, _ = command_components(tmp_path)
    workspace = await GitWorkspace.from_path(LocalSandbox(tmp_path), str(target))

    session = await create_optimization_session(
        workspace=workspace,
        session_dir=tmp_path / "sessions" / "generic",
        backend_id="command",
        backend=backend,
        objective=ObjectiveSpec(
            selector=MetricSelector(metric="score"),
            direction="maximize",
        ),
        producers={},
        max_candidates=0,
        authorization_resolver=allow_all_evaluations,
    )
    result = await session.run()

    assert result.baseline.objective.value == 0.0


@pytest.mark.asyncio
async def test_local_factory_builds_and_resumes_generic_session(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("baseline\n", encoding="utf-8")
    baseline_version = initialize_repository(target)
    backend, producer = command_components(tmp_path)
    session_dir = tmp_path / "sessions" / "factory"
    objective = ObjectiveSpec(
        selector=MetricSelector(metric="score"),
        direction="maximize",
    )

    session = await create_local_optimization_session(
        project_path=target,
        session_dir=session_dir,
        session_id="stable-id",
        backend_id="command",
        backend=backend,
        objective=objective,
        evaluation_set=EvaluationSet(name="quality"),
        producers={"default": producer},
        max_candidates=1,
    )
    result = await session.run()

    assert session.id == "stable-id"
    assert result.baseline.request.candidate.version == baseline_version
    assert result.best.objective.value == 1.0
    assert (target / "program.txt").read_text(encoding="utf-8") == "baseline\n"
    assert session.load_manifest().status == SessionStatus.COMPLETED
    assert len(session.database.evaluations) == 2

    # Simulate a crash after the canonical evaluation directory was committed
    # but before database.json was updated.
    database_path = session_dir / "database.json"
    stale_database = json.loads(database_path.read_text(encoding="utf-8"))
    stale_database["evaluations"].pop(result.best.id)
    stale_database["candidates"].pop(result.best.request.candidate.id)
    database_path.write_text(json.dumps(stale_database), encoding="utf-8")

    resumed = await create_local_optimization_session(
        project_path=target,
        session_dir=session_dir,
        session_id=None,
        backend_id="command",
        backend=backend,
        objective=objective,
        evaluation_set=EvaluationSet(name="quality"),
        producers={},
        max_candidates=0,
    )
    resumed_result = await resumed.run(skip_baseline_evaluation=True)

    assert len(resumed.database.evaluations) == 2
    assert resumed.id == "stable-id"
    assert resumed_result.baseline.id == result.baseline.id
    assert len(resumed_result.evaluations) == 2
    assert resumed_result.best.id == result.best.id


@pytest.mark.asyncio
async def test_local_factory_continues_candidate_rounds_after_resume(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("baseline\n", encoding="utf-8")
    initialize_repository(target)
    backend, producer = command_components(tmp_path)
    session_dir = tmp_path / "sessions" / "resume"
    objective = ObjectiveSpec(
        selector=MetricSelector(metric="score"),
        direction="maximize",
    )
    kwargs = {
        "project_path": target,
        "session_dir": session_dir,
        "session_id": "resume",
        "backend_id": "command",
        "backend": backend,
        "objective": objective,
        "evaluation_set": EvaluationSet(name="quality"),
        "producers": {"default": producer},
    }

    first = await create_local_optimization_session(max_candidates=1, **kwargs)
    first_result = await first.run()
    resumed = await create_local_optimization_session(max_candidates=2, **kwargs)
    resumed_result = await resumed.run(skip_baseline_evaluation=True)

    assert len(first_result.evaluations) == 2
    assert len(resumed_result.evaluations) == 3
    assert len(resumed_result.candidates) == 3
    assert resumed_result.best.objective.value == 2.0
    assert resumed_result.best.request.candidate.metadata["round"] == 1


@pytest.mark.asyncio
async def test_local_factory_can_evaluate_an_older_target_ref(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("baseline\n", encoding="utf-8")
    baseline_version = initialize_repository(target)
    (target / "program.txt").write_text("improved\n", encoding="utf-8")
    subprocess.run(["git", "add", "--all"], cwd=target, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=vero",
            "-c",
            "user.email=vero@localhost",
            "commit",
            "-m",
            "new head",
        ],
        cwd=target,
        check=True,
        capture_output=True,
    )
    head_version = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    backend, _ = command_components(tmp_path)

    session = await create_local_optimization_session(
        project_path=target,
        session_dir=tmp_path / "sessions" / "old-ref",
        session_id="old-ref",
        backend_id="command",
        backend=backend,
        objective=ObjectiveSpec(
            selector=MetricSelector(metric="score"),
            direction="maximize",
        ),
        producers={},
        max_candidates=0,
        base_ref=baseline_version,
    )
    result = await session.run()

    assert result.baseline.request.candidate.version == baseline_version
    assert result.baseline.objective.value == 0.0
    assert head_version != baseline_version
    assert (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=target,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == head_version
    )


@pytest.mark.asyncio
async def test_local_factory_rejects_state_inside_or_outside_version_control(
    tmp_path: Path,
):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("baseline\n", encoding="utf-8")
    initialize_repository(target)
    backend, producer = command_components(tmp_path)
    kwargs = {
        "project_path": target,
        "backend_id": "command",
        "backend": backend,
        "objective": ObjectiveSpec(
            selector=MetricSelector(metric="score"),
            direction="maximize",
        ),
        "producers": {"default": producer},
    }

    with pytest.raises(ValueError, match="outside the target repository"):
        await create_local_optimization_session(
            session_dir=target / ".vero" / "session",
            **kwargs,
        )

    (target / "program.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be clean"):
        await create_local_optimization_session(
            session_dir=tmp_path / "sessions" / "dirty",
            **kwargs,
        )
