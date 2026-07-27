from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import vero.evaluation.store.budget as budget_module
import vero.evaluation.store.persistence as persistence
from vero.candidate import Candidate
from vero.evaluation import (
    AgentSelectionMode,
    AllCases,
    BackendProvenance,
    BackendRegistry,
    BudgetLedger,
    CaseError,
    CaseRange,
    CaseResult,
    CaseStatus,
    DisclosureLevel,
    EvaluationAccessPolicy,
    EvaluationAuthorization,
    EvaluationBudget,
    EvaluationBudgetExceeded,
    EvaluationCost,
    EvaluationDatabase,
    EvaluationDefinition,
    EvaluationDeniedError,
    EvaluationEngine,
    EvaluationExecutionError,
    EvaluationLimits,
    EvaluationPlan,
    EvaluationPrincipal,
    EvaluationRecord,
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
from vero.workspace import Workspace


class StubWorkspace(Workspace):
    def __init__(self, root: Path, version: str = "main", *, dirty: bool = False):
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self._version = version
        self._dirty = dirty
        self.at_calls: list[str] = []
        self.copy_calls: list[str] = []

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


@pytest.mark.asyncio
async def test_store_preserves_principal_across_reconstruction(tmp_path: Path):
    value = record().model_copy(update={"principal": EvaluationPrincipal.ADMIN})
    store = EvaluationStore(tmp_path / value.id)

    await store.save(value)

    # Reconstructing from the source-of-truth directory must keep the real
    # provenance, not silently default it to SYSTEM.
    assert store.load().principal == EvaluationPrincipal.ADMIN
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
async def test_evaluator_marks_reports_invalid_at_the_error_rate_threshold(
    tmp_path: Path,
):
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

    assert value.report.status == EvaluationStatus.INVALID
    assert value.report.metrics["error_rate"] == pytest.approx(0.1)
    assert (
        value.report.diagnostics[-1].code
        == "infrastructure_invalidity_threshold_exceeded"
    )
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
async def test_budget_reservation_rolls_back_when_cancelled(
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

    # The reservation is atomic: a run cancelled while the charge was being
    # written keeps its full budget rather than leaking it, and disk and memory
    # agree on the rolled-back state.
    in_memory = ledger.get("command", evaluation_set)
    on_disk = BudgetLedger.load(path).get("command", evaluation_set)
    assert in_memory is not None
    assert in_memory.remaining_runs == 2
    assert on_disk == in_memory


@pytest.mark.asyncio
async def test_reserve_rollback_write_failure_preserves_cancellation(
    tmp_path: Path,
    monkeypatch,
):
    # If the rollback write itself fails, the run's cancellation must still
    # propagate (not be replaced by the write's OSError), with the write error
    # chained for diagnosis.
    evaluation_set = EvaluationSet(name="performance")
    ledger = BudgetLedger(
        [
            EvaluationBudget(
                backend_id="command",
                evaluation_set_key=evaluation_set.budget_key("command"),
                total_runs=2,
            )
        ],
        path=tmp_path / "budget.json",
    )
    ledger.save()
    started = threading.Event()
    release = threading.Event()
    real_write = budget_module._atomic_write_json
    calls = {"n": 0}

    def failing_rollback_write(write_path, value):
        calls["n"] += 1
        if calls["n"] == 1:
            # the charge write: block so the reservation can be cancelled here
            started.set()
            assert release.wait(timeout=5)
            real_write(write_path, value)
        else:
            # the rollback write fails durably
            raise OSError("disk gone")

    monkeypatch.setattr(budget_module, "_atomic_write_json", failing_rollback_write)
    reservation = asyncio.create_task(
        ledger.reserve("command", evaluation_set, EvaluationCost())
    )
    assert await asyncio.to_thread(started.wait, 5)
    reservation.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await reservation
    assert isinstance(excinfo.value.__cause__, OSError)


@pytest.mark.asyncio
async def test_reserve_charge_write_failure_preserves_cancellation(
    tmp_path: Path,
    monkeypatch,
):
    # The third of the three durable writes in this module, and the one that
    # was still hand-rolling the drain loop: if the charge write itself fails
    # while the run is being cancelled, the caller must still see the
    # cancellation. Returning OSError instead breaks structured cancellation
    # and hides that the run went away.
    evaluation_set = EvaluationSet(name="performance")
    ledger = BudgetLedger(
        [
            EvaluationBudget(
                backend_id="command",
                evaluation_set_key=evaluation_set.budget_key("command"),
                total_runs=2,
            )
        ],
        path=tmp_path / "budget.json",
    )
    started = threading.Event()
    release = threading.Event()

    def blocking_failing_write(write_path, value):
        started.set()
        assert release.wait(timeout=5)
        raise OSError("disk gone")

    monkeypatch.setattr(budget_module, "_atomic_write_json", blocking_failing_write)
    reservation = asyncio.create_task(
        ledger.reserve("command", evaluation_set, EvaluationCost(runs=1))
    )
    assert await asyncio.to_thread(started.wait, 5)
    reservation.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await reservation
    assert isinstance(excinfo.value.__cause__, OSError)
    # The charge never landed, so the budget is untouched on both sides.
    assert ledger.get("command", evaluation_set).remaining_runs == 2


@pytest.mark.asyncio
async def test_refund_write_failure_preserves_cancellation(tmp_path: Path, monkeypatch):
    # A cancellation racing the durable refund write must win over the write's
    # own failure rather than being swallowed.
    evaluation_set = EvaluationSet(name="performance")
    ledger = BudgetLedger(
        [
            EvaluationBudget(
                backend_id="command",
                evaluation_set_key=evaluation_set.budget_key("command"),
                total_runs=2,
            )
        ],
        path=tmp_path / "budget.json",
    )
    await ledger.reserve("command", evaluation_set, EvaluationCost(runs=1))
    started = threading.Event()
    release = threading.Event()

    def blocking_failing_write(write_path, value):
        started.set()
        assert release.wait(timeout=5)
        raise OSError("disk gone")

    monkeypatch.setattr(budget_module, "_atomic_write_json", blocking_failing_write)
    refund = asyncio.create_task(
        ledger.refund("command", evaluation_set, EvaluationCost(runs=1))
    )
    assert await asyncio.to_thread(started.wait, 5)
    refund.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await refund
    assert isinstance(excinfo.value.__cause__, OSError)


@pytest.mark.asyncio
async def test_execution_error_refunds_reservation(tmp_path: Path):
    # A backend exception (EvaluationExecutionError) must refund the reservation
    # via the shielded cleanup path, restoring the budget in memory and on disk.
    workspace = StubWorkspace(tmp_path / "repo")
    backend = StubBackend(error=RuntimeError("boom"))
    evaluation_set = request().evaluation_set
    ledger = BudgetLedger(
        [
            EvaluationBudget(
                backend_id="default",
                evaluation_set_key=evaluation_set.budget_key("default"),
                total_runs=2,
            )
        ],
        path=tmp_path / "budgets.json",
    )
    ledger.save()
    engine = EvaluationEngine(
        evaluator=evaluator(tmp_path, workspace),
        backends=BackendRegistry({"default": backend}),
        database=EvaluationDatabase(id="session"),
        database_path=tmp_path / "database.json",
        budget_ledger=ledger,
        authorization_resolver=allow_all_evaluations,
    )

    with pytest.raises(EvaluationExecutionError):
        await engine.evaluate_record(backend_id="default", request=request())

    remaining = ledger.get("default", evaluation_set)
    assert remaining is not None and remaining.remaining_runs == 2
    assert (
        BudgetLedger.load(tmp_path / "budgets.json")
        .get("default", evaluation_set)
        .remaining_runs
        == 2
    )


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
async def test_finalization_drains_admitted_agent_evaluations_and_closes_admission(
    tmp_path: Path,
):
    class BlockingBackend(StubBackend):
        def __init__(self):
            super().__init__(
                report=EvaluationReport(
                    status=EvaluationStatus.SUCCESS,
                    metrics={"score": 0.75},
                )
            )
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def evaluate(self, *, context, request):
            self.evaluate_calls += 1
            self.started.set()
            await self.release.wait()
            return self.report

    workspace = StubWorkspace(tmp_path / "repo")
    backend = BlockingBackend()
    database = EvaluationDatabase(id="session")
    engine = EvaluationEngine(
        evaluator=evaluator(tmp_path, workspace),
        backends=BackendRegistry({"default": backend}),
        database=database,
        authorization_resolver=allow_all_evaluations,
    )
    evaluation = asyncio.create_task(
        engine.evaluate_record(backend_id="default", request=request())
    )
    await backend.started.wait()

    drain = asyncio.create_task(engine.quiesce_agent_evaluations(timeout_seconds=1.0))
    await asyncio.sleep(0)
    assert not drain.done()
    with pytest.raises(EvaluationDeniedError, match="finalization has started"):
        await engine.evaluate_record(backend_id="default", request=request("late"))

    backend.release.set()

    assert await drain == 1
    assert (await evaluation).report.metrics == {"score": 0.75}
    assert len(database.evaluations) == 1
    admin = await engine.evaluate_record(
        backend_id="default",
        request=request("admin"),
        principal=EvaluationPrincipal.ADMIN,
    )
    assert admin.request.candidate.version == "admin"


@pytest.mark.asyncio
async def test_finalization_cancels_an_agent_evaluation_after_drain_timeout(
    tmp_path: Path,
):
    class BlockingBackend(StubBackend):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()

        async def evaluate(self, *, context, request):
            self.evaluate_calls += 1
            self.started.set()
            await asyncio.Event().wait()

    workspace = StubWorkspace(tmp_path / "repo")
    backend = BlockingBackend()
    database = EvaluationDatabase(id="session")
    engine = EvaluationEngine(
        evaluator=evaluator(tmp_path, workspace),
        backends=BackendRegistry({"default": backend}),
        database=database,
        authorization_resolver=allow_all_evaluations,
    )
    evaluation = asyncio.create_task(
        engine.evaluate_record(backend_id="default", request=request())
    )
    await backend.started.wait()

    assert (
        await engine.quiesce_agent_evaluations(
            timeout_seconds=0.01,
            cancellation_grace_seconds=1.0,
        )
        == 1
    )
    with pytest.raises(asyncio.CancelledError):
        await evaluation

    assert len(database.evaluations) == 1
    cancelled = next(iter(database.evaluations.values()))
    assert cancelled.report.status == EvaluationStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancellation_during_the_success_record_still_completes_it(
    tmp_path: Path,
):
    """A charged run must finish being recorded even if the caller goes away.

    The success path was the one _record call not shielded, so a cancellation
    arriving while it ran abandoned it part-way: the budget stayed charged for
    an evaluation the engine never finished publishing.

    The cancellable window is the listener loop, not the database write --
    asyncio.to_thread cannot be interrupted, so that part completes either way.
    Listeners are where the trusted side mirrors an evaluation to W&B and the
    session archive, so silently skipping them loses the run from both.
    """
    workspace = StubWorkspace(tmp_path / "repo")
    database = EvaluationDatabase(id="session")
    engine = EvaluationEngine(
        evaluator=evaluator(tmp_path, workspace),
        backends=BackendRegistry({"default": StubBackend()}),
        database=database,
        authorization_resolver=allow_all_evaluations,
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def slow_listener(record: EvaluationRecord) -> None:
        entered.set()
        await release.wait()
        finished.set()

    engine.listeners.append(slow_listener)
    evaluation = asyncio.create_task(
        engine.evaluate_record(backend_id="default", request=request())
    )
    await asyncio.wait_for(entered.wait(), timeout=5)

    evaluation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await evaluation
    # shield lets the caller unwind at once while _record finishes in the
    # background, so wait for the listener rather than assuming it ran.
    release.set()
    await asyncio.wait_for(finished.wait(), timeout=5)
    assert len(database.evaluations) == 1


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


@pytest.mark.asyncio
async def test_engine_enforces_aggregate_k_anonymity_floor(tmp_path: Path):
    """The floor is a core-engine property, not a sidecar courtesy."""
    canonical = EvaluationSet(name="validation", selection=CaseRange(stop=10))
    plan = EvaluationPlan(
        evaluations=[
            EvaluationDefinition(
                evaluation_set=canonical,
                # min_aggregate_cases omitted: resolves to the safe floor (5).
                access=EvaluationAccessPolicy(disclosure=DisclosureLevel.AGGREGATE),
            )
        ],
        selection_evaluation="validation",
    )

    def engine_with(cost: EvaluationCost) -> EvaluationEngine:
        return EvaluationEngine(
            evaluator=evaluator(tmp_path, StubWorkspace(tmp_path / "repo")),
            backends=BackendRegistry({"default": StubBackend(cost=cost)}),
            database=EvaluationDatabase(id="session"),
            authorization_resolver=authorize_evaluation_plan(plan),
        )

    def request_with(selection) -> EvaluationRequest:
        return request().model_copy(
            update={
                "evaluation_set": canonical.model_copy(update={"selection": selection})
            }
        )

    # 1) An agent-chosen 2-case aggregate subset is refused by the engine.
    with pytest.raises(EvaluationDeniedError, match="at least 5 cases"):
        await engine_with(EvaluationCost(cases=2)).evaluate(
            backend_id="default",
            request=request_with(CaseRange(stop=2)),
            principal=EvaluationPrincipal.AGENT,
        )

    # 2) A subset without exact case costs cannot be floored: refused.
    with pytest.raises(EvaluationDeniedError, match="exact case costs"):
        await engine_with(EvaluationCost(cases=None)).evaluate(
            backend_id="default",
            request=request_with(CaseRange(stop=2)),
            principal=EvaluationPrincipal.AGENT,
        )

    # 3) A subset at the floor passes.
    await engine_with(EvaluationCost(cases=5)).evaluate(
        backend_id="default",
        request=request_with(CaseRange(stop=5)),
        principal=EvaluationPrincipal.AGENT,
    )

    # 4) The complete selection is exempt: its aggregate is the intended
    #    disclosure, however small the set.
    await engine_with(EvaluationCost(cases=2)).evaluate(
        backend_id="default",
        request=request_with(AllCases()),
        principal=EvaluationPrincipal.AGENT,
    )

    # 5) Trusted principals are never floored.
    await engine_with(EvaluationCost(cases=2)).evaluate(
        backend_id="default",
        request=request_with(CaseRange(stop=2)),
        principal=EvaluationPrincipal.ADMIN,
    )


@pytest.mark.asyncio
async def test_raw_cancellation_during_cleanup_still_refunds(tmp_path: Path):
    """A CancelledError that escapes the evaluator must not leak the reservation.

    The evaluator turns cancellation and failure into two typed errors, but each
    has to await a persist before it can raise them. A cancellation delivered
    during that await -- which quiesce_agent_evaluations does deliver, when it
    drains agent evaluations at finalization -- unwinds as a raw CancelledError
    instead, matching neither typed handler. Without a bare handler in the engine
    the reservation stays charged forever.
    """

    class CancellingEvaluator:
        """Stands in for an evaluator interrupted mid-cleanup."""

        def __init__(self, evaluations_dir):
            self.evaluations_dir = evaluations_dir

        async def evaluate(self, **_kwargs):
            raise asyncio.CancelledError()

    workspace = StubWorkspace(tmp_path / "repo")
    evaluation_set = request().evaluation_set
    ledger = BudgetLedger(
        [
            EvaluationBudget(
                backend_id="default",
                evaluation_set_key=evaluation_set.budget_key("default"),
                total_runs=2,
            )
        ],
        path=tmp_path / "budgets.json",
    )
    ledger.save()
    engine = EvaluationEngine(
        evaluator=CancellingEvaluator(evaluator(tmp_path, workspace).evaluations_dir),
        backends=BackendRegistry({"default": StubBackend()}),
        database=EvaluationDatabase(id="session"),
        database_path=tmp_path / "database.json",
        budget_ledger=ledger,
        authorization_resolver=allow_all_evaluations,
    )

    # The cancellation still propagates -- it is not swallowed.
    with pytest.raises(asyncio.CancelledError):
        await engine.evaluate_record(backend_id="default", request=request())

    remaining = ledger.get("default", evaluation_set)
    assert remaining is not None and remaining.remaining_runs == 2
    assert (
        BudgetLedger.load(tmp_path / "budgets.json")
        .get("default", evaluation_set)
        .remaining_runs
        == 2
    )
