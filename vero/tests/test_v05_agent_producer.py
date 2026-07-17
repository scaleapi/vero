from __future__ import annotations

import hashlib
import json
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
    EvaluationAccessPolicy,
    EvaluationDatabase,
    EvaluationDefinition,
    EvaluationEngine,
    EvaluationPlan,
    EvaluationReceipt,
    EvaluationSet,
    Evaluator,
    MetricSelector,
    ObjectiveSpec,
    authorize_evaluation_plan,
)
from vero.optimization import CandidateProposal, Optimizer, SequentialStrategy
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
        self.feedback: EvaluationReceipt | None = None
        self.initial_candidate_ids: set[str] = set()
        self.initial_evaluation_count = 0
        self.case_resource = None

    async def run(self, *, context, prompt, max_turns, on_event=None):
        assert prompt == "Make the program faster"
        assert max_turns == 5
        context_root = Path(context.workspace.project_path) / ".vero"
        assert not await context.workspace.is_dirty()
        assert context.workspace.can_read(".vero/manifest.json")
        assert not context.workspace.can_write(".vero")
        assert not context.workspace.can_write(".vero/manifest.json")
        manifest = json.loads((context_root / "manifest.json").read_text())
        assert manifest["parent_candidate_id"] == context.parent.id
        candidate_index = json.loads(
            (context_root / "candidates" / "index.json").read_text()
        )
        self.initial_candidate_ids = {
            candidate["candidate_id"] for candidate in candidate_index["candidates"]
        }
        evaluation_index = json.loads(
            (context_root / "evaluations" / "index.json").read_text()
        )
        self.initial_evaluation_count = len(evaluation_index["evaluations"])
        cases_index = json.loads((context_root / "cases" / "index.json").read_text())
        resource_path = cases_index["case_resources"][0]["path"]
        resource_index = json.loads(
            (
                context_root / "cases" / resource_path / "resources" / "index.json"
            ).read_text()
        )
        self.case_resource = json.loads(
            (
                context_root
                / "cases"
                / resource_path
                / "resources"
                / resource_index["resources"][0]["path"]
            ).read_text()
        )
        program = Path(context.workspace.project_path) / "program.txt"
        program.write_text("fast\n", encoding="utf-8")
        feedback = await context.evaluation.evaluate(
            evaluation="performance",
            description="Try the fast implementation"
        )
        assert isinstance(feedback, EvaluationReceipt)
        self.feedback = feedback
        assert (Path(context.workspace.project_path) / feedback.result_path).is_file()
        refreshed = json.loads(
            (context_root / "evaluations" / "index.json").read_text()
        )
        assert len(refreshed["evaluations"]) == self.initial_evaluation_count + 1

        # A later edit regresses. The evaluated checkpoint must remain selectable.
        program.write_text("slow\n", encoding="utf-8")
        return AgentRunResult(
            description="Finish agent attempt",
            state={"turn": 2},
            trace=[{"objective": feedback.result.objective.value}],
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
async def test_failed_agent_run_persists_state_and_trace(tmp_path: Path):
    class FailingAgent:
        async def run(self, **kwargs):
            raise RuntimeError("turn limit reached")

        def serialize_state(self):
            return {"turn": 5}

        def serialize_trace(self):
            return [{"tool": "file_read"}]

    artifacts = ArtifactStore(tmp_path / "artifacts")
    producer = AgentCandidateProducer(FailingAgent(), artifacts=artifacts)
    proposal = CandidateProposal(id="proposal", producer_id="default")
    baseline = object()
    context = SimpleNamespace(
        session_id="session",
        candidates={},
        baseline=baseline,
    )

    with pytest.raises(RuntimeError, match="turn limit reached"):
        await producer.produce(
            proposal=proposal,
            context=context,
            workspace=SimpleNamespace(),
            evaluation=SimpleNamespace(),
        )

    digest = hashlib.sha256(proposal.id.encode()).hexdigest()[:16]
    assert artifacts.read_json(f"agents/{digest}/state.json") == {"turn": 5}
    assert artifacts.read_json(f"agents/{digest}/trace.json") == [{"tool": "file_read"}]
    assert artifacts.read_json(f"agents/{digest}/failure.json") == {
        "type": "RuntimeError",
        "message": "turn limit reached",
    }
    assert artifacts.read_json(producer._producer_state_path("default")) == {"turn": 5}


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
    cases = harness / "cases.json"
    cases.write_text(
        json.dumps([{"id": "case-1", "size": 128}]) + "\n",
        encoding="utf-8",
    )

    sandbox = await LocalSandbox.create(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(target))
    session_dir = tmp_path / "sessions" / "agent"
    candidate_repository = await GitCandidateRepository.create(
        session_dir / "candidates", workspace=workspace
    )
    database = EvaluationDatabase(id="agent")

    plan = EvaluationPlan(
        evaluations=[
            EvaluationDefinition(
                evaluation_set=EvaluationSet(name="performance"),
                access=EvaluationAccessPolicy(
                    disclosure=DisclosureLevel.AGGREGATE,
                    expose_case_resources=True,
                ),
            ),
            EvaluationDefinition(
                evaluation_set=EvaluationSet(name="test", partition="test"),
                access=EvaluationAccessPolicy(
                    agent_can_evaluate=False,
                    agent_visible=False,
                    disclosure=DisclosureLevel.NONE,
                ),
            ),
        ],
        selection_evaluation="performance",
        final_evaluation="test",
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
                        staged_inputs={"cases": str(cases)},
                        agent_context_inputs={"performance": ["cases"]},
                    )
                )
            }
        ),
        database=database,
        database_path=session_dir / "database.json",
        authorization_resolver=authorize_evaluation_plan(plan),
    )
    agent = CheckpointingCodingAgent()
    artifacts = ArtifactStore(session_dir / "artifacts")
    optimizer = Optimizer(
        workspace=workspace,
        candidate_repository=candidate_repository,
        engine=engine,
        backend_id="command",
        evaluation_plan=plan,
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
        max_proposals=1,
    )

    result = await optimizer.run()

    assert agent.feedback is not None
    assert agent.feedback.result.objective.value == 1.0
    assert agent.feedback.result_path.startswith(".vero/evaluations/")
    assert agent.initial_candidate_ids == {baseline_version}
    assert agent.initial_evaluation_count == 1
    assert agent.case_resource == [{"id": "case-1", "size": 128}]
    assert len(result.evaluations) == 5
    assert len(result.candidates) == 3
    assert result.best.objective.value == 1.0
    assert result.best.request.candidate.id.endswith(":trial:1")
    assert result.final is not None
    assert result.final.request.evaluation_set.name == "test"
    assert result.best.request.candidate.parent_id == baseline_version
    final = next(
        candidate
        for candidate in result.candidates
        if candidate.id not in {baseline_version, result.best.request.candidate.id}
    )
    assert final.parent_id == result.best.request.candidate.id
    assert (target / "program.txt").read_text(encoding="utf-8") == "slow\n"
    assert len(database.evaluations) == 5

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
