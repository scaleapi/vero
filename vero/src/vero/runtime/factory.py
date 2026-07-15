"""Composition helpers for local optimization sessions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import JsonValue

from vero.candidate import Candidate
from vero.evaluation import (
    AuthorizationResolver,
    BackendRegistry,
    BudgetLedger,
    EvaluationBackend,
    EvaluationBudget,
    EvaluationDatabase,
    EvaluationEngine,
    EvaluationLimits,
    EvaluationSet,
    Evaluator,
    ObjectiveSpec,
)
from vero.optimization import (
    CandidateProducer,
    ObjectiveSelectionPolicy,
    OptimizationStrategy,
    Optimizer,
    SelectionPolicy,
    SequentialStrategy,
)
from vero.runtime.session import OptimizationSession, SessionManifest
from vero.sandbox import LocalSandbox
from vero.workspace import GitWorkspace, Workspace


def _load_database(session_dir: Path, session_id: str) -> EvaluationDatabase:
    database_path = session_dir / "database.json"
    if database_path.exists():
        database = EvaluationDatabase.load_from_file(database_path)
    else:
        database = EvaluationDatabase.from_evaluations_dir(
            session_dir / "evaluations",
            database_id=session_id,
        )
    if database.id != session_id:
        raise ValueError(
            f"evaluation database belongs to session {database.id!r}, "
            f"not {session_id!r}"
        )
    return database


def _load_budget_ledger(
    session_dir: Path,
    budgets: list[EvaluationBudget] | None,
) -> BudgetLedger | None:
    budget_path = session_dir / "budgets.json"
    if budget_path.exists():
        return BudgetLedger.load(budget_path)
    if budgets is None:
        return None
    ledger = BudgetLedger(budgets, path=budget_path)
    ledger.save()
    return ledger


async def create_optimization_session(
    *,
    workspace: Workspace,
    session_dir: Path | str,
    backend_id: str,
    backend: EvaluationBackend,
    objective: ObjectiveSpec,
    producers: Mapping[str, CandidateProducer],
    session_id: str | None = None,
    evaluation_set: EvaluationSet | None = None,
    strategy: OptimizationStrategy | None = None,
    selection: SelectionPolicy | None = None,
    parameters: dict[str, JsonValue] | None = None,
    limits: EvaluationLimits | None = None,
    budgets: list[EvaluationBudget] | None = None,
    authorization_resolver: AuthorizationResolver | None = None,
    metadata: dict[str, JsonValue] | None = None,
    seed: int | None = None,
    max_candidates: int = 1,
    max_rounds: int = 100,
    max_concurrency: int = 1,
    use_evaluation_copies: bool = True,
    base_ref: str | None = None,
) -> OptimizationSession:
    """Build a durable session around an already-provisioned workspace.

    ``session_dir`` is durable control-plane state on the host.  The workspace
    may live in any sandbox and is never interpreted as a host filesystem path.
    """

    session_dir = Path(session_dir).expanduser().resolve()
    manifest_path = session_dir / "manifest.json"
    if session_id is None and manifest_path.exists():
        session_id = SessionManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        ).id
    session_id = session_id or session_dir.name
    if not session_id.strip():
        raise ValueError("session ID must not be empty")
    if not producers and max_candidates:
        raise ValueError("at least one candidate producer is required")
    for producer in producers.values():
        validate_workspace = getattr(producer, "validate_workspace", None)
        if callable(validate_workspace):
            validate_workspace(workspace)

    if await workspace.is_dirty():
        raise ValueError("target workspace must be clean before optimization")
    baseline_version = (
        await workspace.resolve_ref(base_ref)
        if base_ref is not None
        else await workspace.current_version()
    )

    session_dir.mkdir(parents=True, exist_ok=True)
    database_path = session_dir / "database.json"
    database = _load_database(session_dir, session_id)
    budget_ledger = _load_budget_ledger(session_dir, budgets)
    evaluator = Evaluator(
        workspace=workspace,
        session_dir=session_dir,
        session_id=session_id,
        use_copy=use_evaluation_copies,
    )
    engine = EvaluationEngine(
        evaluator=evaluator,
        backends=BackendRegistry({backend_id: backend}),
        database=database,
        database_path=database_path,
        budget_ledger=budget_ledger,
        authorization_resolver=authorization_resolver,
    )
    optimizer = Optimizer(
        workspace=workspace,
        engine=engine,
        backend_id=backend_id,
        evaluation_set=evaluation_set or EvaluationSet(),
        objective=objective,
        strategy=strategy or SequentialStrategy(),
        producers=dict(producers),
        selection=selection or ObjectiveSelectionPolicy(),
        parameters=parameters or {},
        limits=limits or EvaluationLimits(),
        seed=seed,
        max_candidates=max_candidates,
        max_rounds=max_rounds,
        max_concurrency=max_concurrency,
        session_id=session_id,
    )
    session = OptimizationSession(
        id=session_id,
        session_dir=session_dir,
        optimizer=optimizer,
        baseline=Candidate.from_version(baseline_version),
        metadata=metadata or {},
    )
    for producer_id, producer in producers.items():
        bind_artifacts = getattr(producer, "bind_artifacts", None)
        if callable(bind_artifacts):
            bind_artifacts(session.artifacts, producer_id=producer_id)
    return session


async def create_local_optimization_session(
    *,
    project_path: Path | str,
    session_dir: Path | str,
    backend_id: str,
    backend: EvaluationBackend,
    objective: ObjectiveSpec,
    producers: Mapping[str, CandidateProducer],
    session_id: str | None = None,
    evaluation_set: EvaluationSet | None = None,
    strategy: OptimizationStrategy | None = None,
    selection: SelectionPolicy | None = None,
    parameters: dict[str, JsonValue] | None = None,
    limits: EvaluationLimits | None = None,
    budgets: list[EvaluationBudget] | None = None,
    authorization_resolver: AuthorizationResolver | None = None,
    metadata: dict[str, JsonValue] | None = None,
    seed: int | None = None,
    max_candidates: int = 1,
    max_rounds: int = 100,
    max_concurrency: int = 1,
    use_evaluation_copies: bool = True,
    base_ref: str | None = None,
) -> OptimizationSession:
    """Provision a local Git workspace and build an optimization session."""

    project_path = Path(project_path).expanduser().resolve()
    session_path = Path(session_dir).expanduser().resolve()
    sandbox = await LocalSandbox.create(root=project_path.parent)
    workspace = await GitWorkspace.from_path(sandbox, str(project_path))
    repository_root = Path(workspace.root).resolve()
    if session_path == repository_root or session_path.is_relative_to(repository_root):
        raise ValueError("session directory must live outside the target repository")
    return await create_optimization_session(
        workspace=workspace,
        session_dir=session_path,
        backend_id=backend_id,
        backend=backend,
        objective=objective,
        producers=producers,
        session_id=session_id,
        evaluation_set=evaluation_set,
        strategy=strategy,
        selection=selection,
        parameters=parameters,
        limits=limits,
        budgets=budgets,
        authorization_resolver=authorization_resolver,
        metadata=metadata,
        seed=seed,
        max_candidates=max_candidates,
        max_rounds=max_rounds,
        max_concurrency=max_concurrency,
        use_evaluation_copies=use_evaluation_copies,
        base_ref=base_ref,
    )
