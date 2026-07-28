from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vero.candidate import Candidate
from vero.evaluation import (
    BackendProvenance,
    CaseResult,
    CaseStatus,
    DisclosureLevel,
    EvaluationArtifact,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    EvaluationSummary,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
    project_evaluation,
)
from vero.runtime.context import (
    AgentContextDirectory,
    AgentDisclosureLedger,
    context_digest,
    make_evaluation_receipt,
)
from vero.sandbox import LocalSandbox


def record(
    evaluation_id: str,
    *,
    artifact: bool = False,
    trace_marker: str = "trace-marker",
) -> EvaluationRecord:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    objective = ObjectiveSpec(
        selector=MetricSelector(metric="score"),
        direction="maximize",
    )
    artifacts = (
        [
            EvaluationArtifact(path="logs/output.txt", media_type="text/plain"),
            EvaluationArtifact(path="logs/leak.txt", media_type="text/plain"),
        ]
        if artifact
        else []
    )
    return EvaluationRecord(
        id=evaluation_id,
        request=EvaluationRequest(
            candidate=Candidate(
                id=f"candidate:{evaluation_id}",
                version="a" * 40,
                created_at=created_at,
            ),
            evaluation_set=EvaluationSet(name="validation"),
        ),
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": 0.75},
            cases=[
                CaseResult(
                    case_id="case/one",
                    status=CaseStatus.SUCCESS,
                    metrics={"score": 0.75},
                    input={"prompt": "private case"},
                    output={"answer": "candidate answer"},
                    execution_trace=[{"message": trace_marker}],
                    evaluation_trace=[{"grader": "accepted"}],
                    artifacts=artifacts,
                )
            ],
        ),
        backend_id="backend",
        backend=BackendProvenance(
            name="test",
            version="1",
            config_digest="0" * 64,
        ),
        objective_spec=objective,
        objective=ObjectiveResult(value=0.75, feasible=True),
        created_at=created_at,
        completed_at=created_at + timedelta(seconds=1),
    )


@pytest.mark.asyncio
async def test_agent_context_splits_full_traces_and_honors_disclosure(tmp_path: Path):
    session_dir = tmp_path / "session"
    full = record("evaluation:full", artifact=True, trace_marker="x" * 10_000)
    aggregate = record("evaluation:aggregate")
    hidden = record("evaluation:hidden")
    source_artifact = (
        session_dir / "evaluations" / full.id / "artifacts" / "logs" / "output.txt"
    )
    source_artifact.parent.mkdir(parents=True)
    source_artifact.write_text("artifact details\n", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("must not be copied\n", encoding="utf-8")
    os.symlink(secret, source_artifact.parent / "leak.txt")

    project = tmp_path / "project"
    project.mkdir()
    directory = AgentContextDirectory(
        sandbox=await LocalSandbox.create(root=tmp_path),
        root=str(project / ".evals"),
        session_dir=session_dir,
    )
    await directory.reset()
    await directory.write_header(
        session_id="session",
        round_number=2,
        proposal_id="proposal",
        parent_candidate_id="parent",
    )
    await directory.write_evaluations(
        [
            (
                full,
                DisclosureLevel.FULL,
                project_evaluation(full, DisclosureLevel.FULL),
            ),
            (
                aggregate,
                DisclosureLevel.AGGREGATE,
                project_evaluation(aggregate, DisclosureLevel.AGGREGATE),
            ),
            (
                hidden,
                DisclosureLevel.NONE,
                project_evaluation(hidden, DisclosureLevel.NONE),
            ),
        ]
    )
    await directory.seal()

    try:
        evaluations = project / ".evals" / "results"
        full_root = evaluations / context_digest(full.id)
        full_document = json.loads(
            (full_root / "evaluation.json").read_text(encoding="utf-8")
        )
        assert full_document["disclosure"] == "full"
        assert "cases" not in full_document["result"]["report"]
        case_path = full_document["result"]["case_files"][0]["path"]
        case_document = json.loads((full_root / case_path).read_text(encoding="utf-8"))
        assert case_document["execution_trace_path"] == "execution-trace.json"
        trace = (full_root / Path(case_path).parent / "execution-trace.json").read_text(
            encoding="utf-8"
        )
        assert "x" * 10_000 in trace
        assert (full_root / "artifacts" / "logs" / "output.txt").read_text() == (
            "artifact details\n"
        )
        assert full_document["missing_artifacts"] == ["logs/leak.txt"]
        assert not (full_root / "artifacts" / "logs" / "leak.txt").exists()

        aggregate_root = evaluations / context_digest(aggregate.id)
        aggregate_document = json.loads(
            (aggregate_root / "evaluation.json").read_text(encoding="utf-8")
        )
        assert aggregate_document["disclosure"] == "aggregate"
        assert aggregate_document["result"]["metrics"] == {"score": 0.75}
        assert not (aggregate_root / "cases").exists()

        hidden_document = json.loads(
            (evaluations / context_digest(hidden.id) / "evaluation.json").read_text(
                encoding="utf-8"
            )
        )
        assert hidden_document == {
            "schema_version": 1,
            "disclosure": "none",
            "result": {
                "evaluation_id": hidden.id,
                "status": "success",
            },
        }
        mode = stat.S_IMODE((project / ".evals" / "manifest.json").stat().st_mode)
        assert mode & 0o222 == 0

        receipt = make_evaluation_receipt(full, DisclosureLevel.FULL)
        assert isinstance(receipt.result, EvaluationSummary)
        assert receipt.result_path == (
            f".evals/results/{context_digest(full.id)}/evaluation.json"
        )
        assert "x" * 100 not in receipt.model_dump_json()
    finally:
        await directory.unseal()


@pytest.mark.asyncio
async def test_agent_context_reset_preserves_mounted_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "agent-context"
    root.mkdir()
    (root / "stale.txt").write_text("stale\n", encoding="utf-8")
    sandbox = await LocalSandbox.create(root=tmp_path)
    remove = sandbox.remove

    async def reject_root_removal(path: str, *, recursive: bool = False) -> None:
        if Path(path).resolve() == root.resolve():
            raise AssertionError("context mount root must not be removed")
        await remove(path, recursive=recursive)

    monkeypatch.setattr(sandbox, "remove", reject_root_removal)
    directory = AgentContextDirectory(
        sandbox=sandbox,
        root=str(root),
        session_dir=tmp_path / "session",
    )
    await directory.seal()

    await directory.reset()

    assert root.is_dir()
    assert list(root.iterdir()) == []


@pytest.mark.asyncio
async def test_disclosure_ledger_survives_restart_and_never_broadens(tmp_path: Path):
    path = tmp_path / "agent-context.json"
    ledger = AgentDisclosureLedger(path)

    assert await ledger.remember("evaluation", DisclosureLevel.AGGREGATE) == (
        DisclosureLevel.AGGREGATE
    )
    reopened = AgentDisclosureLedger(path)
    assert await reopened.remember("evaluation", DisclosureLevel.FULL) == (
        DisclosureLevel.AGGREGATE
    )
    assert await reopened.remember("evaluation", DisclosureLevel.NONE) == (
        DisclosureLevel.NONE
    )
    assert AgentDisclosureLedger(path).get("evaluation") == DisclosureLevel.NONE


@pytest.mark.asyncio
async def test_evaluation_plan_reports_partition_size(tmp_path: Path):
    """The plan carries each partition's case count.

    Without it an agent sizing a subset can only guess a `--stop`, and learns the
    bound from a rejected request — observed live, where an optimizer asked for
    cases 20-70 of a 49-case partition and lost two evaluations to it.
    """
    from vero.evaluation import EvaluationAccessPolicy, EvaluationCost, EvaluationSet
    from vero.runtime.context import ContextPlanEntry

    class SizedBackend:
        async def resolve_cost(self, evaluation_set) -> EvaluationCost:
            return EvaluationCost(cases=49)

    class UncostableBackend:
        async def resolve_cost(self, evaluation_set):
            raise RuntimeError("cannot cost this set")

    directory = AgentContextDirectory(
        sandbox=await LocalSandbox.create(root=tmp_path),
        root=str(tmp_path / "ctx"),
        session_dir=tmp_path / "session",
    )
    await directory.reset()
    await directory.write_evaluation_plan(
        [
            ContextPlanEntry(
                backend_id="harbor-development",
                backend=SizedBackend(),
                evaluation_set=EvaluationSet(name="bench", partition="development"),
                access=EvaluationAccessPolicy(agent_can_evaluate=True),
            ),
            ContextPlanEntry(
                backend_id="harbor-validation",
                backend=UncostableBackend(),
                evaluation_set=EvaluationSet(name="bench", partition="validation"),
                access=EvaluationAccessPolicy(agent_can_evaluate=True),
            ),
        ]
    )

    plan = json.loads((tmp_path / "ctx" / "plan.json").read_text(encoding="utf-8"))
    # The backend serving each partition, so `evals run --backend/--partition`
    # can be taken from one row. Mismatching them is refused as an opaque
    # "evaluation denied", which cost swe-atlas-qna's optimizer a round trip.
    assert {row["partition"]: row["backend"] for row in plan["evaluations"]} == {
        "development": "harbor-development",
        "validation": "harbor-validation",
    }
    sizes = {row["partition"]: row["cases"] for row in plan["evaluations"]}
    # A backend that cannot cost itself degrades to no count, not a broken plan.
    assert sizes == {"development": 49, "validation": None}
