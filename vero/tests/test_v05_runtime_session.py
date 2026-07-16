import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from vero.candidate import Candidate
from vero.evaluation import (
    BackendProvenance,
    EvaluationDatabase,
    EvaluationLimits,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
)
from vero.optimization import OptimizationResult
from vero.runtime import ArtifactStore, EventBus, OptimizationSession, SessionStatus


def evaluation(candidate: Candidate) -> EvaluationRecord:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    objective = ObjectiveSpec(
        selector=MetricSelector(metric="score"),
        direction="maximize",
    )
    return EvaluationRecord(
        id=f"evaluation:{candidate.id}",
        request=EvaluationRequest(candidate=candidate),
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": 1.0},
        ),
        backend_id="default",
        backend=BackendProvenance(
            name="fake",
            version="1",
            config_digest="0" * 64,
        ),
        objective_spec=objective,
        objective=ObjectiveResult(value=1.0, feasible=True),
        created_at=created_at,
        completed_at=created_at + timedelta(seconds=1),
    )


class StubWorkspace:
    async def current_version(self):
        return "baseline-version"


class StubCandidateRepository:
    family = "stub"
    format_version = 1


class StubOptimizer:
    def __init__(self, session_dir: Path, *, failure: Exception | None = None):
        self.workspace = StubWorkspace()
        self.candidate_repository = StubCandidateRepository()
        self.backend_id = "default"
        self.evaluation_set = EvaluationSet(name="performance")
        self.objective = ObjectiveSpec(
            selector=MetricSelector(metric="score"),
            direction="maximize",
        )
        self.parameters = {}
        self.limits = EvaluationLimits()
        self.seed = None
        self.session_id = None
        self.engine = SimpleNamespace(
            evaluator=SimpleNamespace(
                session_dir=session_dir,
                candidate_repository=self.candidate_repository,
            ),
            database=EvaluationDatabase(id=session_dir.name),
            budget_ledger=None,
            backends=SimpleNamespace(
                resolve=lambda _: SimpleNamespace(
                    provenance=BackendProvenance(
                        name="fake",
                        version="1",
                        config_digest="0" * 64,
                    )
                )
            ),
        )
        self.failure = failure
        self.calls: list[tuple[Candidate, bool]] = []

    async def run(self, *, baseline, skip_baseline_evaluation=False):
        self.calls.append((baseline, skip_baseline_evaluation))
        if self.failure is not None:
            raise self.failure
        baseline_record = evaluation(baseline)
        return OptimizationResult(
            baseline=baseline_record,
            evaluations=(baseline_record,),
            candidates=(baseline,),
            best=baseline_record,
        )


@pytest.mark.asyncio
async def test_session_persists_lifecycle_events_and_best_result(tmp_path: Path):
    session_dir = tmp_path / "sessions" / "run-1"
    optimizer = StubOptimizer(session_dir)
    session = OptimizationSession(
        id="run-1",
        session_dir=session_dir,
        optimizer=optimizer,
        metadata={"purpose": "test"},
    )

    result = await session.run()

    manifest = session.load_manifest()
    assert manifest.schema_version == 2
    assert manifest.status == SessionStatus.COMPLETED
    assert manifest.baseline.version == "baseline-version"
    assert manifest.best_candidate_id == "baseline-version"
    assert manifest.best_evaluation_id == result.best.id
    assert manifest.metadata == {"purpose": "test"}
    events = [json.loads(line) for line in session.events_path.read_text().splitlines()]
    assert [event["kind"] for event in events] == [
        "session_started",
        "evaluation_completed",
        "session_completed",
    ]
    assert session.database is optimizer.engine.database


@pytest.mark.asyncio
async def test_session_persists_failure_before_reraising(tmp_path: Path):
    session_dir = tmp_path / "sessions" / "failed"
    session = OptimizationSession(
        id="failed",
        session_dir=session_dir,
        optimizer=StubOptimizer(session_dir, failure=RuntimeError("producer exploded")),
    )

    with pytest.raises(RuntimeError, match="producer exploded"):
        await session.run()

    manifest = session.load_manifest()
    assert manifest.status == SessionStatus.FAILED
    assert manifest.failure.type.endswith("RuntimeError")
    assert manifest.failure.message == "producer exploded"
    events = [json.loads(line) for line in session.events_path.read_text().splitlines()]
    assert events[-1]["kind"] == "session_failed"


@pytest.mark.asyncio
async def test_session_rejects_resume_with_changed_objective(tmp_path: Path):
    session_dir = tmp_path / "sessions" / "changed-objective"
    optimizer = StubOptimizer(session_dir)
    session = OptimizationSession(
        id="changed-objective",
        session_dir=session_dir,
        optimizer=optimizer,
    )
    await session.run()
    optimizer.objective = ObjectiveSpec(
        selector=MetricSelector(metric="different"),
        direction="minimize",
    )

    with pytest.raises(ValueError, match="objective does not match"):
        await session.run(skip_baseline_evaluation=True)


@pytest.mark.asyncio
async def test_session_rejects_resume_with_changed_evaluation_parameters(
    tmp_path: Path,
):
    session_dir = tmp_path / "sessions" / "changed-parameters"
    optimizer = StubOptimizer(session_dir)
    session = OptimizationSession(
        id="changed-parameters",
        session_dir=session_dir,
        optimizer=optimizer,
    )
    await session.run()
    optimizer.parameters = {"temperature": 0.5}

    with pytest.raises(ValueError, match="parameters do not match"):
        await session.run(skip_baseline_evaluation=True)


def test_session_rejects_mismatched_evaluator_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="must match"):
        OptimizationSession(
            id="run",
            session_dir=tmp_path / "expected",
            optimizer=StubOptimizer(tmp_path / "different"),
        )


def test_session_binds_and_validates_optimizer_session_id(tmp_path: Path):
    session_dir = tmp_path / "directory-name-can-differ"
    optimizer = StubOptimizer(session_dir)

    OptimizationSession(
        id="canonical-session-id",
        session_dir=session_dir,
        optimizer=optimizer,
    )

    assert optimizer.session_id == "canonical-session-id"
    optimizer.session_id = "different"
    with pytest.raises(ValueError, match="session ID"):
        OptimizationSession(
            id="canonical-session-id",
            session_dir=session_dir,
            optimizer=optimizer,
        )


def test_artifact_store_rejects_escaping_paths(tmp_path: Path):
    store = ArtifactStore(tmp_path / "artifacts")

    path = store.write_json("agent/state.json", {"turn": 3})

    assert path == tmp_path / "artifacts" / "agent" / "state.json"
    assert store.read_json("agent/state.json") == {"turn": 3}
    for unsafe in ("", "../escape", "/absolute", "a//b", "a\\b"):
        with pytest.raises(ValueError):
            store.write_text(unsafe, "no")


@pytest.mark.asyncio
async def test_event_sink_failure_does_not_break_runtime():
    captured = []

    def broken(_event):
        raise RuntimeError("sink failed")

    bus = EventBus([broken, captured.append])

    event = await bus.emit(session_id="session", kind="test")

    assert captured == [event]
