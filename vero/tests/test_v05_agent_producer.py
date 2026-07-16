from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vero.agents import AgentCandidateProducer, AgentRequirements, AgentRunResult
from vero.candidate_repository import GitCandidateRepository
from vero.evaluation import (
    BackendRegistry,
    CommandBackend,
    CommandBackendConfig,
    DisclosureLevel,
    EvaluationAuthorization,
    EvaluationDatabase,
    EvaluationEngine,
    EvaluationSet,
    EvaluationSummary,
    Evaluator,
    MetricSelector,
    ObjectiveSpec,
)
from vero.optimization import Optimizer, SequentialStrategy
from vero.runtime import ArtifactStore
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


class CheckpointingCodingAgent:
    def __init__(self):
        self.feedback: EvaluationSummary | None = None

    async def run(self, *, context, prompt, max_turns, on_event=None):
        assert prompt == "Make the program faster"
        assert max_turns == 5
        program = Path(context.workspace.project_path) / "program.txt"
        program.write_text("fast\n", encoding="utf-8")
        feedback = await context.evaluation.evaluate_current(
            description="Try the fast implementation"
        )
        assert isinstance(feedback, EvaluationSummary)
        self.feedback = feedback

        # A later edit regresses. The evaluated checkpoint must remain selectable.
        program.write_text("slow\n", encoding="utf-8")
        return AgentRunResult(
            description="Finish agent attempt",
            state={"turn": 2},
            trace=[{"objective": feedback.objective.value}],
            metadata={"provider": "test"},
        )


def test_host_native_agent_rejects_isolated_workspace():
    class HostNativeAgent:
        requirements = AgentRequirements(host_visible_workspace=True)

    class IsolatedSandbox:
        def host_path(self, path):
            return None

    producer = AgentCandidateProducer(HostNativeAgent())

    with pytest.raises(ValueError, match="requires a host-visible workspace"):
        producer.validate_workspace(
            SimpleNamespace(
                project_path="/workspace/target",
                sandbox=IsolatedSandbox(),
            )
        )


@pytest.mark.asyncio
async def test_agent_checkpoint_is_a_selectable_candidate(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("slow\n", encoding="utf-8")
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
    "metrics": {"latency_ms": latency},
}))
""",
        encoding="utf-8",
    )

    sandbox = await LocalSandbox.create(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(target))
    session_dir = tmp_path / "sessions" / "agent"
    candidate_repository = await GitCandidateRepository.create(
        session_dir / "candidates", workspace=workspace
    )
    database = EvaluationDatabase(id="agent")

    async def authorize(backend_id, request):
        return EvaluationAuthorization(
            may_evaluate=True,
            disclosure=DisclosureLevel.AGGREGATE,
        )

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
        authorization_resolver=authorize,
    )
    agent = CheckpointingCodingAgent()
    artifacts = ArtifactStore(session_dir / "artifacts")
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
        strategy=SequentialStrategy(instruction="Make the program faster"),
        producers={
            "default": AgentCandidateProducer(
                agent,
                max_turns=5,
                artifacts=artifacts,
            )
        },
        max_candidates=1,
    )

    result = await optimizer.run()

    assert agent.feedback is not None
    assert agent.feedback.objective.value == 1.0
    assert len(result.evaluations) == 3
    assert len(result.candidates) == 3
    assert result.best.objective.value == 1.0
    assert result.best.request.candidate.id.endswith(":trial:1")
    assert result.best.request.candidate.parent_id == baseline_version
    final = next(
        candidate
        for candidate in result.candidates
        if candidate.id not in {baseline_version, result.best.request.candidate.id}
    )
    assert final.parent_id == result.best.request.candidate.id
    assert (target / "program.txt").read_text(encoding="utf-8") == "slow\n"
    assert len(database.evaluations) == 3

    agent_artifacts = list((session_dir / "artifacts" / "agents").iterdir())
    proposal_artifacts = [path for path in agent_artifacts if path.name != "producers"]
    assert len(proposal_artifacts) == 1
    assert (proposal_artifacts[0] / "state.json").exists()
    assert (proposal_artifacts[0] / "trace.json").exists()

    class ResumedAgent:
        def __init__(self):
            self.state = None

        def deserialize_state(self, state):
            self.state = state

    resumed_agent = ResumedAgent()
    resumed_producer = AgentCandidateProducer(resumed_agent)
    resumed_producer.bind_artifacts(artifacts)
    assert resumed_agent.state == {"turn": 2}
