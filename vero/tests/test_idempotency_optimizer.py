from __future__ import annotations

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
    EvaluationPlan,
    EvaluationSet,
    Evaluator,
    MetricSelector,
    ObjectiveSpec,
    allow_all_evaluations,
)
from vero.optimization import (
    CandidateChange,
    CandidateProposal,
    Optimizer,
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


class FixedProposalStrategy:
    """Propose one proposal with a stable ID so trial IDs are predictable."""

    async def propose(self, _context):
        return [CandidateProposal(id="proposal-a", producer_id="default")]


class RepeatingProducer:
    """Score one checkpoint twice, then edit and score the new checkpoint."""

    def __init__(self):
        self.receipts = []

    async def produce(self, *, proposal, context, workspace, evaluation):
        program = Path(workspace.project_path) / "program.txt"
        program.write_text("fast\n", encoding="utf-8")
        self.receipts.append(
            await evaluation.evaluate(
                evaluation="performance",
                description="Score the fast implementation",
            )
        )
        # The retry a flaky harness or an impatient agent produces: nothing in
        # the workspace moved between these two calls.
        self.receipts.append(
            await evaluation.evaluate(
                evaluation="performance",
                description="Re-score the identical checkpoint",
            )
        )
        program.write_text("faster\n", encoding="utf-8")
        self.receipts.append(
            await evaluation.evaluate(
                evaluation="performance",
                description="Score the faster implementation",
            )
        )
        return CandidateChange(description="Finish repeating producer")


@pytest.mark.asyncio
async def test_repeated_agent_evaluation_reuses_the_unchanged_trial_candidate(
    tmp_path: Path,
):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("slow\n", encoding="utf-8")
    baseline_version = initialize_repository(target)

    runs = tmp_path / "harness-runs.txt"
    harness = tmp_path / "harness"
    harness.mkdir()
    harness_script = harness / "evaluate.py"
    harness_script.write_text(
        """
import json
import sys
from pathlib import Path

workspace, report_path, runs_path = map(Path, sys.argv[1:])
program = (workspace / "program.txt").read_text().strip()
latency = {"slow": 10.0, "fast": 1.0, "faster": 0.5}[program]
with runs_path.open("a", encoding="utf-8") as handle:
    handle.write(program + "\\n")
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
    session_dir = tmp_path / "sessions" / "repeat"
    candidate_repository = await GitCandidateRepository.create(
        session_dir / "candidates", workspace=workspace
    )
    database = EvaluationDatabase(id="repeat")
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
                            str(runs),
                        ],
                    )
                )
            }
        ),
        database=database,
        database_path=session_dir / "database.json",
        authorization_resolver=allow_all_evaluations,
    )
    producer = RepeatingProducer()
    optimizer = Optimizer(
        workspace=workspace,
        candidate_repository=candidate_repository,
        engine=engine,
        backend_id="command",
        evaluation_plan=EvaluationPlan.single(EvaluationSet(name="performance")),
        objective=ObjectiveSpec(
            selector=MetricSelector(metric="latency_ms"),
            direction="minimize",
        ),
        strategy=FixedProposalStrategy(),
        producers={"default": producer},
        max_proposals=1,
    )

    result = await optimizer.run()

    # The repeat did not skip the measurement: the harness ran once for the
    # baseline and once per gateway call, and every call returned its own
    # evaluation with a real score for the content it saw.
    assert runs.read_text(encoding="utf-8").split() == [
        "slow",
        "fast",
        "fast",
        "faster",
    ]
    assert len({receipt.evaluation_id for receipt in producer.receipts}) == 3
    assert [receipt.result.objective.value for receipt in producer.receipts] == [
        1.0,
        1.0,
        0.5,
    ]

    # The unchanged repeat reused trial 1 instead of minting a second identity
    # for byte-identical content, so only the genuine edit advanced the counter.
    trials = sorted(
        (
            candidate
            for candidate in result.candidates
            if "trial" in candidate.metadata
        ),
        key=lambda candidate: int(candidate.metadata["trial"]),
    )
    assert [candidate.id for candidate in trials] == [
        "proposal-a:trial:1",
        "proposal-a:trial:2",
    ]
    assert [candidate.metadata["trial"] for candidate in trials] == [1, 2]
    assert trials[0].parent_id == baseline_version
    assert trials[1].parent_id == "proposal-a:trial:1"
    # The durable archive holds the baseline plus exactly one candidate per
    # distinct checkpoint, which is what lets a resume recognize the same work.
    assert len(candidate_repository.list()) == 3

    trial_records = [
        record
        for record in result.evaluations
        if "trial" in record.request.candidate.metadata
    ]
    assert len(trial_records) == 3
    candidate_ids = [record.request.candidate.id for record in trial_records]
    assert candidate_ids == [
        "proposal-a:trial:1",
        "proposal-a:trial:1",
        "proposal-a:trial:2",
    ]
    versions = [record.request.candidate.version for record in trial_records]
    assert versions[0] == versions[1]
    assert versions[2] != versions[0]

    # Scores are untouched by the deduplication: the baseline and every trial
    # keep the value their own measurement produced, and the winner is still
    # the fastest checkpoint.
    assert result.baseline.objective.value == 10.0
    assert [record.objective.value for record in trial_records] == [1.0, 1.0, 0.5]
    assert result.best.objective.value == 0.5
    assert result.best.request.candidate.id == "proposal-a:trial:2"
