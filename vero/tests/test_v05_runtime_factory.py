from __future__ import annotations

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
)
from vero.optimization import (
    CommandCandidateProducer,
    CommandCandidateProducerConfig,
)
from vero.runtime import SessionStatus, create_local_optimization_session


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
score = 1.0 if value == "improved" else 0.0
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
Path(sys.argv[1], "program.txt").write_text("improved\\n")
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
            command=[sys.executable, str(producer_script), "{workspace}"],
            description="Improve the program",
        )
    )
    return backend, producer


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
