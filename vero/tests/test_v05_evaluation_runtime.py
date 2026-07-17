from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vero.candidate import Candidate
from vero.evaluation import (
    AgentSelectionMode,
    BackendProvenance,
    BackendRegistry,
    BudgetLedger,
    CaseError,
    CaseResult,
    CaseStatus,
    CaseRange,
    DisclosureLevel,
    EvaluationAuthorization,
    EvaluationAccessPolicy,
    EvaluationBudget,
    EvaluationBudgetExceeded,
    EvaluationCost,
    EvaluationDatabase,
    EvaluationDeniedError,
    EvaluationEngine,
    EvaluationLimits,
    EvaluationDefinition,
    EvaluationExecutionError,
    EvaluationRecord,
    EvaluationPlan,
    EvaluationPrincipal,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    EvaluationStore,
    Evaluator,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
    allow_all_evaluations,
    authorize_evaluation_plan,
)
from vero.evaluation import persistence
import vero.evaluation.budget as budget_module
from vero.filesystem import AccessType, Filesystem
from vero.workspace import Workspace


class StubWorkspace(Workspace):
    def __init__(self, root: Path, version: str = "main", *, dirty: bool = False):
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self._version = version
        self._dirty = dirty
        self.at_calls: list[str] = []
        self.copy_calls: list[str] = []
        self._fs = Filesystem(root=root, default_access=AccessType.WRITE)

    @property
    def sandbox(self):
        return None

    @property
    def root(self) -> str:
        return str(self._root)

    @property
    def project_path(self) -> str:
        return str(self._root)

    @property
    def name(self) -> str:
        return "workspace"

    async def current_version(self) -> str:
        return self._version

    async def save(self, message: str = "Save") -> str:
        return self._version

    async def restore(self, version_id: str, message: str | None = None) -> str:
        self._version = version_id
        return version_id

    async def diff(self, from_version=None, to_version=None) -> str:
        return ""

    async def log(self, max_count=10, since_version=None) -> str:
        return ""

    async def is_ancestor(self, version_a: str, version_b: str) -> bool:
        return True

    async def copy(self, name=None, from_version=None):
        return StubWorkspace(self._root, from_version or self._version)

    @asynccontextmanager
    async def temp_copy(self, from_version=None):
        version = from_version or self._version
        self.copy_calls.append(version)
        yield StubWorkspace(self._root, version)

    @asynccontextmanager
    async def at(self, version_id: str):
        previous = self._version
        self.at_calls.append(version_id)
        self._version = version_id
        try:
            yield
        finally:
            self._version = previous

    async def is_dirty(self) -> bool:
        return self._dirty


class StubCandidateRepository:
    family = "stub"

    def __init__(self, workspace: StubWorkspace):
        self.workspace = workspace
        self.checkout_calls: list[str] = []

    @asynccontextmanager
    async def checkout(self, candidate, *, sandbox, name=None):
        self.checkout_calls.append(candidate.version)
        yield StubWorkspace(self.workspace._root, candidate.version)


class StubBackend:
    def __init__(
        self,
        *,
        report: EvaluationReport | None = None,
        cost: EvaluationCost | None = None,
        error: Exception | None = None,
    ):
        self.report = report or EvaluationReport(status=EvaluationStatus.SUCCESS)
        self.cost = cost or EvaluationCost(cases=0)
        self.error = error
        self.resolve_calls = 0
        self.evaluate_calls = 0
        self.running_manifests: list[dict] = []

    @property
    def provenance(self) -> BackendProvenance:
        return BackendProvenance(
            name="stub",
            version="1",
            config_digest="0" * 64,
        )

    async def resolve_cost(self, evaluation_set: EvaluationSet) -> EvaluationCost:
        self.resolve_calls += 1
        return self.cost

    async def evaluate(self, *, context, request):
        self.evaluate_calls += 1
        self.running_manifests.append(
            json.loads((context.result_dir / "evaluation.json").read_text())
        )
        await context.case_store.save(
            CaseResult(case_id="checkpoint", status=CaseStatus.SUCCESS)
        )
        if self.error is not None:
            raise self.error
        return self.report


def request(version: str = "candidate") -> EvaluationRequest:
    return EvaluationRequest(
        candidate=Candidate(
            id=f"id:{version}",
            version=version,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        evaluation_set=EvaluationSet(name="performance"),
    )


def evaluator(tmp_path: Path, workspace: StubWorkspace) -> Evaluator:
    runtime = Evaluator(
        candidate_repository=StubCandidateRepository(workspace),
        sandbox=workspace.sandbox,
        session_dir=tmp_path / "sessions" / "session",
    )
    return runtime


def record(candidate_id: str = "candidate") -> EvaluationRecord:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    specification = ObjectiveSpec(
        selector=MetricSelector(metric="score"),
        direction="maximize",
    )
    return EvaluationRecord(
        id=f"evaluation:{candidate_id}",
        request=EvaluationRequest(
            candidate=Candidate(
                id=candidate_id,
                version=f"version:{candidate_id}",
                created_at=created_at,
            )
        ),
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": 1.0},
            cases=[
                CaseResult(
                    case_id="case/with/path-characters",
                    status=CaseStatus.SUCCESS,
                    metrics={"score": 1.0},
                )
            ],
        ),
        backend_id="default",
        backend=BackendProvenance(
            name="stub",
            version="1",
            config_digest="0" * 64,
        ),
        objective_spec=specification,
        objective=ObjectiveResult(value=1.0, feasible=True),
        created_at=created_at,
        completed_at=created_at + timedelta(seconds=1),
    )


@pytest.mark.asyncio
async def test_store_splits_cases_from_manifest_and_round_trips(tmp_path: Path):
    value = record()
    store = EvaluationStore(tmp_path / value.id)

    await store.save(value)

    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["lifecycle"] == "complete"
    assert manifest["report"]["cases"] == []
    assert manifest["case_files"][0]["case_id"] == "case/with/path-characters"
    assert "/" not in Path(manifest["case_files"][0]["path"]).name
    assert store.load() == value


def test_running_manifest_is_not_a_completed_record(tmp_path: Path):
    value = record()
    store = EvaluationStore(tmp_path / value.id)
    store.write_running(
        evaluation_id=value.id,
        request=value.request,
        backend_id=value.backend_id,
        backend=value.backend,
        objective_spec=value.objective_spec,
        created_at=value.created_at,
    )

    with pytest.raises(ValueError, match="invalid evaluation manifest"):
        store.load()


def test_atomic_json_write_closes_descriptor_when_fdopen_fails(
    tmp_path: Path,
    monkeypatch,
):
    closed = []
    real_close = persistence.os.close

    def fail_fdopen(*_args, **_kwargs):
        raise RuntimeError("fdopen failed")

    def tracked_close(descriptor):
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(persistence.os, "fdopen", fail_fdopen)
    monkeypatch.setattr(persistence.os, "close", tracked_close)

    with pytest.raises(RuntimeError, match="fdopen failed"):
        persistence._atomic_write_json(tmp_path / "value.json", {"value": 1})

    assert len(closed) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_database_round_trips_schema_one_and_distinguishes_empty_filter(tmp_path: Path):
    database = EvaluationDatabase(id="session")
    value = record()
    database.add_evaluation(value)
    path = tmp_path / "database.json"
    database.save_to_file(path)

    restored = EvaluationDatabase.load_from_file(path)

    assert restored.get_evaluations() == [value]
    assert restored.get_evaluations([]) == []
    assert restored.get_best(value.objective_spec) == value
    assert json.loads(path.read_text())["schema_version"] == 1


@pytest.mark.asyncio
async def test_database_repairs_crash_window_from_completed_evaluations(
    tmp_path: Path,
):
    database_path = tmp_path / "database.json"
    EvaluationDatabase(id="session").save_to_file(database_path)
    value = record()
    await EvaluationStore(tmp_path / "evaluations" / value.id).save(value)

    restored = EvaluationDatabase.load_reconciled(
        database_path=database_path,
        evaluations_dir=tmp_path / "evaluations",
        database_id="session",
    )

    assert restored.get_evaluation(value.id) == value
    assert (
        EvaluationDatabase.load_from_file(database_path).get_evaluation(value.id)
        == value
    )


def test_database_reconciliation_ignores_running_evaluations(tmp_path: Path):
    value = record()
    store = EvaluationStore(tmp_path / "evaluations" / value.id)
    store.write_running(
        evaluation_id=value.id,
        request=value.request,
        backend_id=value.backend_id,
        backend=value.backend,
        objective_spec=value.objective_spec,
        created_at=value.created_at,
    )

    restored = EvaluationDatabase.load_reconciled(
        database_path=tmp_path / "database.json",
        evaluations_dir=tmp_path / "evaluations",
        database_id="session",
    )

    assert restored.evaluations == {}


def test_database_rejects_conflicting_candidate_identity():
    database = EvaluationDatabase(id="session")
    database.add_evaluation(record("same"))
    conflicting = record("other").model_copy(
        update={
            "request": EvaluationRequest(
                candidate=Candidate(
                    id="same",
                    version="different-version",
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
        }
    )

    with pytest.raises(ValueError, match="different identity"):
        database.add_evaluation(conflicting)


@pytest.mark.asyncio
async def test_evaluator_runs_at_candidate_version_and_persists(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo")
    backend = StubBackend(
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"latency": 2.0},
        )
    )
    specification = ObjectiveSpec(
        selector=MetricSelector(metric="latency"),
        direction="minimize",
    )

    runtime = evaluator(tmp_path, workspace)
    value = await runtime.evaluate(
        backend_id="command",
        backend=backend,
        request=request(),
        objective_spec=specification,
    )

    assert runtime.candidate_repository.checkout_calls == ["candidate"]
    assert backend.running_manifests[0]["lifecycle"] == "running"
    assert value.objective.value == 2.0
    result_dir = runtime.evaluations_dir / value.id
    assert EvaluationStore(result_dir).load() == value


@pytest.mark.asyncio
async def test_evaluator_fails_reports_at_the_error_rate_threshold(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo")
    cases = [
        CaseResult(
            case_id=f"success-{index}",
            status=CaseStatus.SUCCESS,
            metrics={"score": 1.0},
        )
        for index in range(9)
    ]
    cases.append(
        CaseResult(
            case_id="error",
            status=CaseStatus.ERROR,
            errors=[CaseError(message="failed", terminal=True)],
        )
    )
    backend = StubBackend(
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": 1.0},
            cases=cases,
        )
    )

    value = await evaluator(tmp_path, workspace).evaluate(
        backend_id="default",
        backend=backend,
        request=request(),
        objective_spec=ObjectiveSpec(
            selector=MetricSelector(metric="score"),
            direction="maximize",
            failure_value=0.0,
        ),
    )

    assert value.report.status == EvaluationStatus.FAILED
    assert value.report.metrics["error_rate"] == pytest.approx(0.1)
    assert value.report.diagnostics[-1].code == "error_rate_threshold_exceeded"
    assert value.objective == ObjectiveResult(value=0.0, feasible=False)


@pytest.mark.asyncio
async def test_error_rate_threshold_can_be_disabled(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo")
    backend = StubBackend(
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": 1.0, "error_rate": 1.0},
        )
    )
    evaluation_request = request().model_copy(
        update={"limits": EvaluationLimits(error_rate_threshold=None)}
    )

    value = await evaluator(tmp_path, workspace).evaluate(
        backend_id="default",
        backend=backend,
        request=evaluation_request,
    )

    assert value.report.status == EvaluationStatus.SUCCESS


@pytest.mark.asyncio
async def test_evaluator_uses_isolated_copy(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo")
    backend = StubBackend()

    runtime = evaluator(tmp_path, workspace)
    await runtime.evaluate(
        backend_id="default",
        backend=backend,
        request=request(),
    )

    assert runtime.candidate_repository.checkout_calls == ["candidate"]
    assert workspace.copy_calls == []
    assert workspace.at_calls == []


@pytest.mark.asyncio
async def test_backend_exception_is_recorded_then_raised(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo")
    backend = StubBackend(error=RuntimeError("boom"))
    runtime = evaluator(tmp_path, workspace)

    with pytest.raises(EvaluationExecutionError) as captured:
        await runtime.evaluate(
            backend_id="default",
            backend=backend,
            request=request(),
        )

    failure = EvaluationStore(
        runtime.evaluations_dir / captured.value.evaluation_id
    ).load()
    assert failure.report.status == EvaluationStatus.FAILED
    assert failure.report.diagnostics[0].code == "backend_error"


@pytest.mark.asyncio
async def test_engine_indexes_a_durable_backend_exception(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo")
    backend = StubBackend(error=RuntimeError("boom"))
    database = EvaluationDatabase(id="session")
    engine = EvaluationEngine(
        evaluator=evaluator(tmp_path, workspace),
        backends=BackendRegistry({"default": backend}),
        database=database,
        database_path=tmp_path / "database.json",
        authorization_resolver=allow_all_evaluations,
    )

    with pytest.raises(EvaluationExecutionError) as captured:
        await engine.evaluate_record(
            backend_id="default",
            request=request(),
        )

    failure = database.get_evaluation(captured.value.evaluation_id)
    assert failure is not None
    assert failure.report.status == EvaluationStatus.FAILED
    assert (
        EvaluationDatabase.load_from_file(tmp_path / "database.json").get_evaluation(
            failure.id
        )
        == failure
    )


@pytest.mark.asyncio
async def test_budget_ledger_reserves_and_restores(tmp_path: Path):
    evaluation_set = EvaluationSet(name="performance")
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(
        [
            EvaluationBudget(
                backend_id="command",
                evaluation_set_key=evaluation_set.budget_key("command"),
                total_runs=2,
                total_cases=10,
            )
        ],
        path=path,
    )

    remaining = await ledger.reserve(
        "command", evaluation_set, EvaluationCost(runs=1, cases=7)
    )

    assert remaining.remaining_runs == 1
    assert remaining.remaining_cases == 3
    assert BudgetLedger.load(path).get("command", evaluation_set) == remaining

    with pytest.raises(EvaluationBudgetExceeded):
        await ledger.reserve("command", evaluation_set, EvaluationCost(runs=1, cases=4))


@pytest.mark.asyncio
async def test_budget_reservation_stays_consistent_when_write_is_cancelled(
    tmp_path: Path,
    monkeypatch,
):
    evaluation_set = EvaluationSet(name="performance")
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(
        [
            EvaluationBudget(
                backend_id="command",
                evaluation_set_key=evaluation_set.budget_key("command"),
                total_runs=2,
            )
        ],
        path=path,
    )
    ledger.save()
    started = threading.Event()
    release = threading.Event()
    real_write = budget_module._atomic_write_json

    def delayed_write(write_path, value):
        started.set()
        assert release.wait(timeout=5)
        real_write(write_path, value)

    monkeypatch.setattr(budget_module, "_atomic_write_json", delayed_write)
    reservation = asyncio.create_task(
        ledger.reserve("command", evaluation_set, EvaluationCost())
    )
    assert await asyncio.to_thread(started.wait, 5)
    reservation.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await reservation

    in_memory = ledger.get("command", evaluation_set)
    on_disk = BudgetLedger.load(path).get("command", evaluation_set)
    assert in_memory is not None
    assert in_memory.remaining_runs == 1
    assert on_disk == in_memory


@pytest.mark.asyncio
async def test_cancelled_evaluation_is_terminal_indexed_and_refunded(tmp_path: Path):
    class BlockingBackend(StubBackend):
        async def evaluate(self, *, context, request):
            self.evaluate_calls += 1
            await asyncio.Event().wait()

    workspace = StubWorkspace(tmp_path / "repo")
    backend = BlockingBackend()
    evaluation_set = request().evaluation_set
    ledger = BudgetLedger(
        [
            EvaluationBudget(
                backend_id="default",
                evaluation_set_key=evaluation_set.budget_key("default"),
                total_runs=1,
            )
        ],
        path=tmp_path / "budgets.json",
    )
    ledger.save()
    database = EvaluationDatabase(id="session")
    engine = EvaluationEngine(
        evaluator=evaluator(tmp_path, workspace),
        backends=BackendRegistry({"default": backend}),
        database=database,
        database_path=tmp_path / "database.json",
        budget_ledger=ledger,
        authorization_resolver=allow_all_evaluations,
    )
    evaluation = asyncio.create_task(
        engine.evaluate_record(
            backend_id="default",
            request=request(),
        )
    )
    while backend.evaluate_calls == 0:
        await asyncio.sleep(0)
    evaluation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await evaluation

    assert len(database.evaluations) == 1
    cancelled = next(iter(database.evaluations.values()))
    assert cancelled.report.status == EvaluationStatus.CANCELLED
    assert cancelled.report.diagnostics[0].code == "evaluation_cancelled"
    assert (
        EvaluationStore(
            evaluator(tmp_path, workspace).evaluations_dir / cancelled.id
        ).load()
        == cancelled
    )
    remaining = ledger.get("default", evaluation_set)
    assert remaining is not None
    assert remaining.remaining_runs == 1


@pytest.mark.asyncio
async def test_plan_authorization_separates_agent_and_system_selection_rights(
    tmp_path: Path,
):
    canonical = EvaluationSet(
        name="validation",
        selection=CaseRange(stop=10),
    )
    plan = EvaluationPlan(
        evaluations=[
            EvaluationDefinition(
                evaluation_set=canonical,
                access=EvaluationAccessPolicy(
                    agent_selection=AgentSelectionMode.FIXED,
                    disclosure=DisclosureLevel.AGGREGATE,
                ),
            )
        ],
        selection_evaluation="validation",
    )
    workspace = StubWorkspace(tmp_path / "repo")
    engine = EvaluationEngine(
        evaluator=evaluator(tmp_path, workspace),
        backends=BackendRegistry({"default": StubBackend()}),
        database=EvaluationDatabase(id="session"),
        authorization_resolver=authorize_evaluation_plan(plan),
    )
    subset_request = request().model_copy(
        update={
            "evaluation_set": canonical.model_copy(
                update={"selection": CaseRange(stop=2)}
            )
        }
    )

    agent = await engine.authorize(
        "default",
        subset_request,
        EvaluationPrincipal.AGENT,
    )
    system = await engine.authorize(
        "default",
        subset_request,
        EvaluationPrincipal.SYSTEM,
    )

    assert agent.may_evaluate is False
    assert agent.viewable is True
    assert system.may_evaluate is True


@pytest.mark.asyncio
async def test_engine_denial_stops_before_cost_and_evaluation(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo")
    backend = StubBackend()
    database = EvaluationDatabase(id="session")
    engine = EvaluationEngine(
        evaluator=evaluator(tmp_path, workspace),
        backends=BackendRegistry({"default": backend}),
        database=database,
    )

    with pytest.raises(EvaluationDeniedError, match="private set"):
        await engine.evaluate(
            backend_id="default",
            request=request(),
            authorization=EvaluationAuthorization(
                may_evaluate=False,
                reason="private set",
            ),
        )

    assert backend.resolve_calls == 0
    assert backend.evaluate_calls == 0
    assert database.evaluations == {}


@pytest.mark.asyncio
async def test_engine_denies_by_default_without_authorization(tmp_path: Path):
    backend = StubBackend()
    engine = EvaluationEngine(
        evaluator=evaluator(tmp_path, StubWorkspace(tmp_path / "repo")),
        backends=BackendRegistry({"default": backend}),
        database=EvaluationDatabase(id="session"),
    )

    with pytest.raises(
        EvaluationDeniedError,
        match="authorization was not configured",
    ):
        await engine.evaluate_record(backend_id="default", request=request())

    assert backend.resolve_calls == 0
    assert backend.evaluate_calls == 0


@pytest.mark.asyncio
async def test_engine_selects_backend_persists_and_projects(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo")
    first = StubBackend(
        report=EvaluationReport(status=EvaluationStatus.SUCCESS, metrics={"value": 1})
    )
    second = StubBackend(
        report=EvaluationReport(status=EvaluationStatus.SUCCESS, metrics={"value": 2})
    )
    database = EvaluationDatabase(id="session")
    database_path = tmp_path / "database.json"
    engine = EvaluationEngine(
        evaluator=evaluator(tmp_path, workspace),
        backends=BackendRegistry({"first": first, "second": second}),
        database=database,
        database_path=database_path,
    )

    summary = await engine.evaluate(
        backend_id="second",
        request=request(),
        authorization=EvaluationAuthorization(
            may_evaluate=True,
            meter_budget=False,
            disclosure=DisclosureLevel.AGGREGATE,
        ),
    )

    assert first.resolve_calls == 0
    assert second.resolve_calls == 1
    assert summary.metrics == {"value": 2.0}
    assert len(database.evaluations) == 1
    assert database_path.exists()
