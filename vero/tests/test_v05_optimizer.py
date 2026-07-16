from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from vero.candidate_repository import GitCandidateRepository
from vero.evaluation import (
    BackendRegistry,
    CommandBackend,
    CommandBackendConfig,
    EvaluationDatabase,
    EvaluationEngine,
    EvaluationSet,
    Evaluator,
    MetricSelector,
    ObjectiveSpec,
    allow_all_evaluations,
)
from vero.optimization import (
    CandidateChange,
    CandidateProposal,
    CommandCandidateProducer,
    CommandCandidateProducerConfig,
    Optimizer,
    SequentialStrategy,
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


@pytest.mark.asyncio
async def test_optimizer_improves_a_non_python_program_via_external_commands(
    tmp_path: Path,
):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("slow\n")
    baseline_version = initialize_repository(target)

    harness = tmp_path / "harness"
    harness.mkdir()
    harness_script = harness / "evaluate.py"
    harness_script.write_text(
        """
import json
import sys
from pathlib import Path

workspace, report_path = map(Path, sys.argv[1:])
program = (workspace / "program.txt").read_text().strip()
latency = 1.0 if program == "fast" else 10.0
report_path.write_text(json.dumps({
    "schema_version": 1,
    "status": "success",
    "metrics": {"latency_ms": latency, "correct": 1.0},
}))
"""
    )

    producer_root = tmp_path / "producer"
    producer_root.mkdir()
    producer_script = producer_root / "optimize.py"
    producer_script.write_text(
        """
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
(workspace / "program.txt").write_text("fast\\n")
"""
    )

    sandbox = await LocalSandbox.create(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(target))
    session_dir = tmp_path / "sessions" / "optimization"
    candidate_repository = await GitCandidateRepository.create(
        session_dir / "candidates", workspace=workspace
    )
    database = EvaluationDatabase(id="optimization")
    engine = EvaluationEngine(
        evaluator=Evaluator(
            candidate_repository=candidate_repository,
            sandbox=workspace.sandbox,
            session_dir=session_dir,
        ),
        backends=BackendRegistry(
            {
                "command": CommandBackend(
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
            }
        ),
        database=database,
        database_path=session_dir / "database.json",
        authorization_resolver=allow_all_evaluations,
    )
    optimizer = Optimizer(
        workspace=workspace,
        candidate_repository=candidate_repository,
        engine=engine,
        backend_id="command",
        evaluation_set=EvaluationSet(name="performance"),
        objective=ObjectiveSpec(
            selector=MetricSelector(metric="latency_ms"),
            direction="minimize",
        ),
        strategy=SequentialStrategy(),
        producers={
            "default": CommandCandidateProducer(
                CommandCandidateProducerConfig(
                    root=str(producer_root),
                    command=[
                        sys.executable,
                        str(producer_script),
                        "{workspace}",
                    ],
                    description="Use fast implementation",
                )
            )
        },
        max_candidates=1,
    )

    result = await optimizer.run()

    assert result.baseline.request.candidate.version == baseline_version
    assert result.baseline.objective.value == 10.0
    assert len(result.evaluations) == 2
    assert len(result.candidates) == 2
    assert result.best.objective.value == 1.0
    assert result.best.request.candidate.parent_id == baseline_version
    assert result.best.request.candidate.version != baseline_version
    assert (target / "program.txt").read_text() == "slow\n"
    assert len(database.evaluations) == 2
    assert (session_dir / "database.json").exists()
    assert len(list((session_dir / "evaluations").iterdir())) == 2
    worktrees = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert worktrees.count("worktree ") == 1
    candidate_branches = subprocess.run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname)",
            "refs/heads/vero-candidate-",
        ],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert candidate_branches == ""
    assert len(candidate_repository.list()) == 2
    assert (candidate_repository.repository_path / "HEAD").exists()


class ReusedIdStrategy:
    async def propose(self, context):
        return [
            CandidateProposal(
                id=context.baseline.request.candidate.id,
                producer_id="default",
            )
        ]


class FailingBatchProducer:
    def __init__(self, sibling_started):
        self.sibling_started = sibling_started

    async def produce(self, **_kwargs):
        await self.sibling_started.wait()
        raise RuntimeError("producer failed")


class CancelledBatchProducer:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = False

    async def produce(self, **_kwargs):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return CandidateChange()


class FailingBatchStrategy:
    async def propose(self, _context):
        return [
            CandidateProposal(id="failing-proposal", producer_id="failing"),
            CandidateProposal(id="cancelled-proposal", producer_id="cancelled"),
        ]


@pytest.mark.asyncio
async def test_optimizer_rejects_strategy_candidate_id_reuse(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("slow\n")
    initialize_repository(target)
    harness = tmp_path / "harness"
    harness.mkdir()
    harness_script = harness / "evaluate.py"
    harness_script.write_text(
        """
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": "success",
    "metrics": {"score": 1.0},
}))
"""
    )
    producer_root = tmp_path / "producer"
    producer_root.mkdir()
    producer_script = producer_root / "noop.py"
    producer_script.write_text("pass\n")
    sandbox = await LocalSandbox.create(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(target))
    candidate_repository = await GitCandidateRepository.create(
        tmp_path / "session" / "candidates", workspace=workspace
    )
    engine = EvaluationEngine(
        evaluator=Evaluator(
            candidate_repository=candidate_repository,
            sandbox=workspace.sandbox,
            session_dir=tmp_path / "session",
        ),
        backends=BackendRegistry(
            {
                "command": CommandBackend(
                    CommandBackendConfig(
                        harness_root=str(harness),
                        command=[sys.executable, str(harness_script), "{report}"],
                    )
                )
            }
        ),
        database=EvaluationDatabase(id="session"),
        authorization_resolver=allow_all_evaluations,
    )
    optimizer = Optimizer(
        workspace=workspace,
        candidate_repository=candidate_repository,
        engine=engine,
        backend_id="command",
        evaluation_set=EvaluationSet(),
        objective=ObjectiveSpec(
            selector=MetricSelector(metric="score"),
            direction="maximize",
        ),
        strategy=ReusedIdStrategy(),
        producers={
            "default": CommandCandidateProducer(
                CommandCandidateProducerConfig(
                    root=str(producer_root),
                    command=[sys.executable, str(producer_script)],
                )
            )
        },
    )

    with pytest.raises(ValueError, match="reused an existing candidate ID"):
        await optimizer.run()


@pytest.mark.asyncio
async def test_optimizer_cancels_sibling_producers_and_cleans_worktrees(
    tmp_path: Path,
):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("slow\n")
    initialize_repository(target)
    harness = tmp_path / "harness"
    harness.mkdir()
    harness_script = harness / "evaluate.py"
    harness_script.write_text(
        """
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": "success",
    "metrics": {"score": 1.0},
}))
"""
    )
    sandbox = await LocalSandbox.create(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(target))
    candidate_repository = await GitCandidateRepository.create(
        tmp_path / "session" / "candidates", workspace=workspace
    )
    engine = EvaluationEngine(
        evaluator=Evaluator(
            candidate_repository=candidate_repository,
            sandbox=workspace.sandbox,
            session_dir=tmp_path / "session",
        ),
        backends=BackendRegistry(
            {
                "command": CommandBackend(
                    CommandBackendConfig(
                        harness_root=str(harness),
                        command=[sys.executable, str(harness_script), "{report}"],
                    )
                )
            }
        ),
        database=EvaluationDatabase(id="session"),
        authorization_resolver=allow_all_evaluations,
    )
    cancelled = CancelledBatchProducer()
    optimizer = Optimizer(
        workspace=workspace,
        candidate_repository=candidate_repository,
        engine=engine,
        backend_id="command",
        evaluation_set=EvaluationSet(),
        objective=ObjectiveSpec(
            selector=MetricSelector(metric="score"),
            direction="maximize",
        ),
        strategy=FailingBatchStrategy(),
        producers={
            "failing": FailingBatchProducer(cancelled.started),
            "cancelled": cancelled,
        },
        max_candidates=2,
        max_concurrency=2,
    )

    with pytest.raises(ExceptionGroup) as captured:
        await optimizer.run()

    assert any(
        isinstance(error, RuntimeError) and str(error) == "producer failed"
        for error in captured.value.exceptions
    )
    assert cancelled.cancelled
    worktrees = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert worktrees.count("worktree ") == 1
