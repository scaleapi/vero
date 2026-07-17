"""Composition helpers for local optimization sessions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import JsonValue

from vero.candidate import Candidate
from vero.candidate_repository import CandidateRepository, GitCandidateRepository
from vero.evaluation import (
    AuthorizationResolver,
    BackendRegistry,
    BudgetLedger,
    EvaluationBackend,
    EvaluationBudget,
    EvaluationDatabase,
    EvaluationEngine,
    EvaluationLimits,
    EvaluationPlan,
    Evaluator,
    ObjectiveSpec,
    authorize_evaluation_plan,
)
from vero.optimization import (
    CandidateProducer,
    ObjectiveSelectionPolicy,
    OptimizationStrategy,
    Optimizer,
    SelectionPolicy,
    SequentialStrategy,
)
from vero.runtime.session import (
    OptimizationRunSpec,
    OptimizationSession,
    SessionManifest,
)
from vero.sandbox import LocalSandbox
from vero.workspace import GitWorkspace, Workspace


def _load_database(session_dir: Path, session_id: str) -> EvaluationDatabase:
    database_path = session_dir / "database.json"
    return EvaluationDatabase.load_reconciled(
        database_path=database_path,
        evaluations_dir=session_dir / "evaluations",
        database_id=session_id,
    )


def _load_budget_ledger(
    session_dir: Path,
    budgets: list[EvaluationBudget],
) -> BudgetLedger | None:
    budget_path = session_dir / "budgets.json"
    if budget_path.exists():
        return BudgetLedger.load(budget_path)
    if not budgets:
        return None
    ledger = BudgetLedger(budgets, path=budget_path)
    ledger.save()
    return ledger


async def create_optimization_session(
    *,
    workspace: Workspace,
    candidate_repository: CandidateRepository,
    session_dir: Path | str,
    backend_id: str,
    backend: EvaluationBackend,
    objective: ObjectiveSpec,
    producers: Mapping[str, CandidateProducer],
    evaluation_plan: EvaluationPlan,
    session_id: str | None = None,
    strategy: OptimizationStrategy | None = None,
    selection: SelectionPolicy | None = None,
    parameters: dict[str, JsonValue] | None = None,
    limits: EvaluationLimits | None = None,
    authorization_resolver: AuthorizationResolver | None = None,
    metadata: dict[str, JsonValue] | None = None,
    run_spec: OptimizationRunSpec | None = None,
    seed: int | None = None,
    max_proposals: int = 1,
    max_rounds: int = 100,
    max_concurrency: int = 1,
    base_ref: str | None = None,
) -> OptimizationSession:
    """Build a durable session around an already-provisioned workspace.

    ``session_dir`` and ``candidate_repository`` are durable control-plane state
    on the host. The original workspace supplies the baseline, context, and
    default execution sandbox; all candidate checkouts come from the compatible
    repository. The evaluation plan is the default fail-closed authorization
    boundary; advanced deployments may supply an equivalent trusted resolver.
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
    if not candidate_repository.supports(workspace):
        raise ValueError(
            "candidate repository and workspace belong to different families"
        )
    mismatched_budgets = [
        budget.backend_id
        for budget in evaluation_plan.budgets
        if budget.backend_id != backend_id
    ]
    if mismatched_budgets:
        raise ValueError(
            "evaluation plan budgets must use the session backend "
            f"{backend_id!r}"
        )
    if not producers and max_proposals:
        raise ValueError("at least one candidate producer is required")
    for producer in producers.values():
        validate_workspace = getattr(producer, "validate_workspace", None)
        if callable(validate_workspace):
            validate_workspace(workspace)

    persisted_manifest = (
        SessionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    if persisted_manifest is not None:
        baseline = persisted_manifest.baseline
        if baseline is None:
            raise ValueError("persisted session is missing its baseline candidate")
        stored = candidate_repository.get(baseline.id)
        if stored != baseline:
            raise ValueError(
                "persisted baseline is missing from the candidate repository"
            )
    else:
        if await workspace.is_dirty():
            raise ValueError("target workspace must be clean before optimization")
        baseline_version = (
            await workspace.resolve_ref(base_ref)
            if base_ref is not None
            else await workspace.current_version()
        )
        baseline = candidate_repository.get(baseline_version)
        if baseline is None:
            baseline = Candidate.from_version(baseline_version)
            if await workspace.current_version() == baseline_version:
                await candidate_repository.capture(baseline, workspace)
            else:
                async with workspace.temp_copy(
                    from_version=baseline_version
                ) as baseline_workspace:
                    await candidate_repository.capture(baseline, baseline_workspace)
        elif (
            baseline.version != baseline_version
            or baseline.parent_id is not None
            or baseline.description is not None
            or baseline.metadata
        ):
            raise ValueError(
                "candidate repository contains a conflicting baseline record"
            )

    session_dir.mkdir(parents=True, exist_ok=True)
    database_path = session_dir / "database.json"
    database = _load_database(session_dir, session_id)
    budget_ledger = _load_budget_ledger(session_dir, evaluation_plan.budgets)
    evaluator = Evaluator(
        candidate_repository=candidate_repository,
        sandbox=workspace.sandbox,
        session_dir=session_dir,
        session_id=session_id,
    )
    engine = EvaluationEngine(
        evaluator=evaluator,
        backends=BackendRegistry({backend_id: backend}),
        database=database,
        database_path=database_path,
        budget_ledger=budget_ledger,
        authorization_resolver=(
            authorization_resolver or authorize_evaluation_plan(evaluation_plan)
        ),
    )
    optimizer = Optimizer(
        workspace=workspace,
        candidate_repository=candidate_repository,
        engine=engine,
        backend_id=backend_id,
        evaluation_plan=evaluation_plan,
        objective=objective,
        strategy=strategy or SequentialStrategy(),
        producers=dict(producers),
        selection=selection or ObjectiveSelectionPolicy(),
        parameters=parameters or {},
        limits=limits or EvaluationLimits(),
        seed=seed,
        max_proposals=max_proposals,
        max_rounds=max_rounds,
        max_concurrency=max_concurrency,
        session_id=session_id,
    )
    session = OptimizationSession(
        id=session_id,
        session_dir=session_dir,
        optimizer=optimizer,
        baseline=baseline,
        metadata=metadata or {},
        run_spec=run_spec,
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
    evaluation_plan: EvaluationPlan,
    session_id: str | None = None,
    strategy: OptimizationStrategy | None = None,
    selection: SelectionPolicy | None = None,
    parameters: dict[str, JsonValue] | None = None,
    limits: EvaluationLimits | None = None,
    authorization_resolver: AuthorizationResolver | None = None,
    metadata: dict[str, JsonValue] | None = None,
    run_spec: OptimizationRunSpec | None = None,
    seed: int | None = None,
    max_proposals: int = 1,
    max_rounds: int = 100,
    max_concurrency: int = 1,
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
    candidate_repository = await GitCandidateRepository.create(
        session_path / "candidates",
        workspace=workspace,
    )
    return await create_optimization_session(
        workspace=workspace,
        candidate_repository=candidate_repository,
        session_dir=session_path,
        backend_id=backend_id,
        backend=backend,
        objective=objective,
        producers=producers,
        evaluation_plan=evaluation_plan,
        session_id=session_id,
        strategy=strategy,
        selection=selection,
        parameters=parameters,
        limits=limits,
        authorization_resolver=authorization_resolver,
        metadata=metadata,
        run_spec=run_spec,
        seed=seed,
        max_proposals=max_proposals,
        max_rounds=max_rounds,
        max_concurrency=max_concurrency,
        base_ref=base_ref,
    )
