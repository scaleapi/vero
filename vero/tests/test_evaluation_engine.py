from contextlib import asynccontextmanager
import json
from datetime import datetime
from pathlib import Path

import pytest

from vero.core.db.candidate import Candidate
from vero.evaluation import (
    BackendProvenance,
    BackendRegistry,
    BudgetLedger,
    DisclosureLevel,
    EvaluationAuthorization,
    EvaluationBackend,
    EvaluationBudget,
    EvaluationBudgetExceeded,
    EvaluationCost,
    EvaluationDatabase,
    EvaluationDeniedError,
    EvaluationEngine,
    EvaluationExecutionError,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    EvaluationStore,
    Evaluator,
    MetricSelector,
    ObjectiveSpec,
    UnknownBackendError,
)
from vero.filesystem import AccessType, Filesystem
from vero.logging import log_evaluations_to_wandb
from vero.workspace import Workspace


class StubWorkspace(Workspace):
    def __init__(self, root: Path, version: str = "main", *, dirty: bool = False):
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self._version = version
        self._dirty = dirty
        self.at_calls: list[str] = []
        self.copy_calls: list[str | None] = []
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
        return "repo"

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
        self.copy_calls.append(from_version)
        yield StubWorkspace(self._root, from_version or self._version)

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


class StubBackend:
    def __init__(
        self,
        *,
        report: EvaluationReport | None = None,
        cost: EvaluationCost | None = None,
        error: Exception | None = None,
    ):
        self._report = report or EvaluationReport(status=EvaluationStatus.SUCCESS)
        self._cost = cost or EvaluationCost(runs=1, cases=0)
        self._error = error
        self.resolve_calls = 0
        self.evaluate_calls = 0
        self.contexts = []
        self.manifests_during_evaluation = []

    @property
    def provenance(self) -> BackendProvenance:
        return BackendProvenance(
            name="stub",
            version="1",
            config_digest="0" * 64,
        )

    async def resolve_cost(self, evaluation_set: EvaluationSet) -> EvaluationCost:
        self.resolve_calls += 1
        return self._cost

    async def evaluate(self, *, context, request):
        self.evaluate_calls += 1
        self.contexts.append(context)
        self.manifests_during_evaluation.append(
            json.loads((context.result_dir / "evaluation.json").read_text())
        )
        if self._error is not None:
            raise self._error
        return self._report


def _request(commit: str = "candidate") -> EvaluationRequest:
    return EvaluationRequest(
        candidate=Candidate(
            commit=commit,
            repo_name="repo",
            created_at=datetime(2026, 1, 1),
        ),
        evaluation_set=EvaluationSet(name="performance"),
    )


def _evaluator(tmp_path: Path, workspace: StubWorkspace, *, use_copy=False):
    return Evaluator(
        workspace=workspace,
        sessions_dir=tmp_path / "sessions",
        session_id="session",
        use_copy=use_copy,
    )


@pytest.mark.asyncio
async def test_evaluator_runs_backend_at_candidate_and_persists_record(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo")
    backend = StubBackend(
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"latency": 2.0},
        )
    )
    objective = ObjectiveSpec(
        selector=MetricSelector(metric="latency"),
        direction="minimize",
    )

    record = await _evaluator(tmp_path, workspace).evaluate(
        backend_id="command",
        backend=backend,
        request=_request(),
        objective_spec=objective,
    )

    assert workspace.at_calls == ["candidate"]
    assert backend.evaluate_calls == 1
    assert backend.manifests_during_evaluation[0]["lifecycle"] == "running"
    assert backend.contexts[0].workspace.current_version
    assert record.backend_id == "command"
    assert record.objective.value == 2.0
    result_dir = tmp_path / "sessions" / "session" / "experiments" / record.id
    assert (result_dir / "evaluation.json").exists()
    assert EvaluationStore(result_dir).load() == record


@pytest.mark.asyncio
async def test_canonical_wandb_payload_uses_objective_and_set_fields(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo")
    backend = StubBackend(
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"latency": 2.0},
        )
    )
    objective = ObjectiveSpec(
        selector=MetricSelector(metric="latency"),
        direction="minimize",
    )
    record = await _evaluator(tmp_path, workspace).evaluate(
        backend_id="command",
        backend=backend,
        request=_request(),
        objective_spec=objective,
    )

    class Run:
        def __init__(self):
            self.payloads = []

        def log(self, payload):
            self.payloads.append(payload)

    run = Run()
    log_evaluations_to_wandb(run, [record])

    assert run.payloads == [
        {
            "performance/candidate_commit": "candidate",
            "performance/evaluation_id": record.id,
            "performance/status": "success",
            "performance/objective_metric": "latency",
            "performance/objective": 2.0,
            "performance/feasible": True,
            "performance/metric/latency": 2.0,
        }
    ]


@pytest.mark.asyncio
async def test_evaluator_uses_isolated_copy_when_enabled(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo")
    backend = StubBackend()

    await _evaluator(tmp_path, workspace, use_copy=True).evaluate(
        backend_id="default",
        backend=backend,
        request=_request(),
    )

    assert workspace.copy_calls == ["candidate"]
    assert workspace.at_calls == []


@pytest.mark.asyncio
async def test_backend_returned_failure_is_a_normal_durable_record(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo")
    backend = StubBackend(
        report=EvaluationReport(
            status=EvaluationStatus.FAILED,
            error="candidate did not compile",
        )
    )

    record = await _evaluator(tmp_path, workspace).evaluate(
        backend_id="default",
        backend=backend,
        request=_request(),
    )

    assert record.report.status == EvaluationStatus.FAILED
    assert record.report.error == "candidate did not compile"


@pytest.mark.asyncio
async def test_thrown_backend_failure_is_recorded_then_raised(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo")
    backend = StubBackend(error=RuntimeError("boom"))
    evaluator = _evaluator(tmp_path, workspace)

    with pytest.raises(EvaluationExecutionError) as captured:
        await evaluator.evaluate(
            backend_id="default",
            backend=backend,
            request=_request(),
        )

    result_dir = evaluator.experiments_dir / captured.value.evaluation_id
    failure = EvaluationStore(result_dir).load()
    assert failure.report.status == EvaluationStatus.FAILED
    assert failure.report.diagnostics[0].code == "backend_error"


@pytest.mark.asyncio
async def test_dirty_direct_workspace_fails_before_backend_work(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo", dirty=True)
    backend = StubBackend()

    with pytest.raises(EvaluationExecutionError, match="clean workspace"):
        await _evaluator(tmp_path, workspace).evaluate(
            backend_id="default",
            backend=backend,
            request=_request(),
        )

    assert backend.evaluate_calls == 0


def test_backend_registry_rejects_duplicates_and_unknown_ids():
    backend = StubBackend()
    registry = BackendRegistry({"one": backend})

    assert isinstance(backend, EvaluationBackend)
    assert registry.resolve("one") is backend
    with pytest.raises(ValueError, match="already registered"):
        registry.register("one", backend)
    with pytest.raises(UnknownBackendError):
        registry.resolve("missing")


@pytest.mark.asyncio
async def test_budget_ledger_reserves_and_restores_remaining_values(tmp_path: Path):
    evaluation_set = EvaluationSet(name="performance")
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(
        [
            EvaluationBudget(
                backend_id="command",
                evaluation_set_key=evaluation_set.budget_key("command"),
                total_runs=2,
                total_cases=10,
                max_cases_per_run=6,
            )
        ],
        path=path,
    )

    remaining = await ledger.reserve(
        "command", evaluation_set, EvaluationCost(runs=1, cases=4)
    )

    assert remaining.remaining_runs == 1
    assert remaining.remaining_cases == 6
    restored = BudgetLedger.load(path)
    assert restored.get("command", evaluation_set) == remaining


@pytest.mark.asyncio
async def test_budget_rejects_unknown_case_cost_when_case_limited():
    evaluation_set = EvaluationSet(name="performance")
    ledger = BudgetLedger(
        [
            EvaluationBudget(
                backend_id="command",
                evaluation_set_key=evaluation_set.budget_key("command"),
                total_cases=10,
            )
        ]
    )

    with pytest.raises(EvaluationBudgetExceeded, match="unknown"):
        await ledger.reserve("command", evaluation_set, EvaluationCost(cases=None))


@pytest.mark.asyncio
async def test_denied_engine_request_stops_before_cost_checkout_and_persistence(
    tmp_path: Path,
):
    workspace = StubWorkspace(tmp_path / "repo")
    backend = StubBackend()
    database = EvaluationDatabase(id="session")
    engine = EvaluationEngine(
        evaluator=_evaluator(tmp_path, workspace),
        backends=BackendRegistry({"default": backend}),
        database=database,
    )

    with pytest.raises(EvaluationDeniedError, match="private set"):
        await engine.evaluate(
            backend_id="default",
            request=_request(),
            authorization=EvaluationAuthorization(
                may_evaluate=False,
                reason="private set",
            ),
        )

    assert backend.resolve_calls == 0
    assert backend.evaluate_calls == 0
    assert workspace.at_calls == []
    assert database.evaluations == {}


@pytest.mark.asyncio
async def test_engine_uses_selected_backend_and_projects_disclosure(tmp_path: Path):
    workspace = StubWorkspace(tmp_path / "repo")
    first = StubBackend(report=EvaluationReport(status="success", metrics={"value": 1}))
    second = StubBackend(report=EvaluationReport(status="success", metrics={"value": 2}))
    database = EvaluationDatabase(id="session")
    engine = EvaluationEngine(
        evaluator=_evaluator(tmp_path, workspace),
        backends=BackendRegistry({"first": first, "second": second}),
        database=database,
        database_path=tmp_path / "database.json",
    )

    summary = await engine.evaluate(
        backend_id="second",
        request=_request(),
        authorization=EvaluationAuthorization(
            may_evaluate=True,
            meter_budget=False,
            disclosure=DisclosureLevel.AGGREGATE,
        ),
    )

    assert first.resolve_calls == 0
    assert second.resolve_calls == 1
    assert summary.backend_id == "second"
    assert summary.metrics == {"value": 2.0}
    assert len(database.evaluations) == 1
    assert (tmp_path / "database.json").exists()
