"""Standard component factory for compiled Harbor optimization tasks."""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from vero.candidate_repository import GitCandidateRepository
from vero.evaluation import (
    BackendRegistry,
    BudgetLedger,
    EvaluationBudget,
    EvaluationDatabase,
    EvaluationLimits,
    EvaluationModel,
    EvaluationSet,
    Evaluator,
    ObjectiveSpec,
)
from vero.evaluation.engine import EvaluationEngine
from vero.harbor.backend import HarborBackend, HarborBackendConfig
from vero.harbor.serve import SidecarComponents
from vero.harbor.session import initialize_harbor_session_manifest
from vero.harbor.sidecar import EvaluationSidecar, SidecarEvaluationPolicy
from vero.harbor.transport import GitCandidateTransport
from vero.harbor.verifier import (
    CanonicalVerifier,
    VerificationSelection,
    VerificationTarget,
)
from vero.sandbox import LocalSandbox
from vero.workspace import GitWorkspace

logger = logging.getLogger(__name__)


class DeploymentSelection(EvaluationModel):
    mode: Literal["submit", "auto_best"] = "auto_best"
    backend_id: str | None = None
    evaluation_set: EvaluationSet | None = None
    objective: ObjectiveSpec | None = None
    baseline_version: str | None = "HEAD"
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    limits: EvaluationLimits = Field(default_factory=EvaluationLimits)
    rescore_top_k: int = Field(default=3, ge=1)
    rescore_attempts: int = Field(default=1, ge=1)
    baseline_floor: bool = False
    baseline_selection_score: float | None = None
    selection_coverage_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    @field_validator("backend_id", "baseline_version")
    @classmethod
    def validate_optional_identity(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional deployment identity must not be empty")
        return value

    @model_validator(mode="after")
    def validate_selection(self) -> DeploymentSelection:
        if self.mode == "auto_best" and (
            self.backend_id is None
            or self.evaluation_set is None
            or self.objective is None
        ):
            raise ValueError(
                "auto_best deployment requires backend_id, evaluation_set, and objective"
            )
        return self


class SidecarWandbConfig(EvaluationModel):
    """Trusted-side Weights & Biases config for the eval-sidecar. The W&B
    credential (WANDB_API_KEY) is supplied to the sidecar container's
    environment, never to the untrusted optimizer agent."""

    project: str
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    mode: str | None = None
    notes: str | None = None
    run_id: str | None = None
    # Also upload each evaluation's trace artifacts (Harbor trial records,
    # stdout/stderr, agent trajectory) to W&B as a per-evaluation artifact.
    # Off by default: it uploads files and can be large.
    log_traces: bool = False


class HarborDeploymentConfig(EvaluationModel):
    task_name: str = "harbor-session"
    task_description: str = ""
    repo_path: str
    agent_repo_path: str
    session_dir: str
    session_id: str = "trial"
    backends: dict[str, HarborBackendConfig]
    access_policies: list[SidecarEvaluationPolicy]
    budgets: list[EvaluationBudget] = Field(default_factory=list)
    selection: DeploymentSelection
    targets: list[VerificationTarget]
    agent_volume: str | None = None
    admin_volume: str
    inference_usage_path: str | None = None
    inference_limits: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    submit_enabled: bool = False
    disclose_budget: bool = True
    score_baseline: bool = True
    evaluation_drain_timeout_seconds: float = Field(default=600.0, gt=0)
    wandb: SidecarWandbConfig | None = None

    @field_validator(
        "repo_path",
        "agent_repo_path",
        "session_dir",
        "admin_volume",
    )
    @classmethod
    def validate_absolute_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not value.startswith("/") or ".." in path.parts:
            raise ValueError("deployment paths must be absolute")
        return value

    @field_validator("inference_usage_path")
    @classmethod
    def validate_optional_file_path(cls, value: str | None) -> str | None:
        if value is not None:
            path = PurePosixPath(value)
            if not value.startswith("/") or ".." in path.parts:
                raise ValueError("deployment paths must be absolute")
        return value

    @field_validator("agent_volume")
    @classmethod
    def validate_optional_path(cls, value: str | None) -> str | None:
        if value is not None and (
            not value.startswith("/") or ".." in PurePosixPath(value).parts
        ):
            raise ValueError("deployment paths must be absolute")
        return value

    @field_validator("session_id", "task_name")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session_id must not be empty")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> HarborDeploymentConfig:
        if not self.backends:
            raise ValueError("deployment requires at least one backend")
        trusted = PurePosixPath(self.repo_path)
        agent = PurePosixPath(self.agent_repo_path)
        if trusted == agent:
            raise ValueError("trusted and agent repositories must be distinct")
        for name, value in (
            ("session_dir", self.session_dir),
            ("admin_volume", self.admin_volume),
            *(
                (("inference_usage_path", self.inference_usage_path),)
                if self.inference_usage_path is not None
                else ()
            ),
        ):
            path = PurePosixPath(value)
            if any(
                path == repository or path.is_relative_to(repository)
                for repository in (agent, trusted)
            ):
                raise ValueError(f"{name} must live outside candidate repositories")
        if self.submit_enabled != (self.selection.mode == "submit"):
            raise ValueError("submit_enabled must match selection mode")
        known = set(self.backends)
        referenced = {policy.backend_id for policy in self.access_policies} | {
            target.backend_id for target in self.targets
        }
        if self.selection.backend_id is not None:
            referenced.add(self.selection.backend_id)
        unknown = sorted(referenced - known)
        if unknown:
            raise ValueError(f"deployment references unknown backends: {unknown}")
        return self


def _database(session_dir: Path, session_id: str) -> EvaluationDatabase:
    database_path = session_dir / "database.json"
    return EvaluationDatabase.load_reconciled(
        database_path=database_path,
        evaluations_dir=session_dir / "evaluations",
        database_id=session_id,
    )


def _ledger(
    session_dir: Path,
    budgets: list[EvaluationBudget],
) -> BudgetLedger | None:
    path = session_dir / "budgets.json"
    if path.exists():
        return BudgetLedger.load(path)
    if not budgets:
        return None
    ledger = BudgetLedger(budgets, path=path)
    ledger.save()
    return ledger


async def build_harbor_components(config: dict) -> SidecarComponents:
    """Build the standard compiled-task topology from trusted JSON config."""
    parsed = HarborDeploymentConfig.model_validate(config)
    session_dir = Path(parsed.session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    # The session dir holds the trusted state — held-out evaluation records and
    # scores, the budget ledger, and every candidate's code. Lock it to the
    # owning (trusted) user so the unprivileged harness that executes candidate
    # code cannot read it. Harmless where the harness is not isolated (the owner
    # retains access); the admin token dir is already hardened in auth.py.
    session_dir.chmod(0o700)
    sandbox = await LocalSandbox.create(root=Path(parsed.repo_path).parent)
    workspace = await GitWorkspace.from_path(sandbox, parsed.repo_path)
    candidate_repository = await GitCandidateRepository.create(
        session_dir / "candidates",
        workspace=workspace,
    )
    database = _database(session_dir, parsed.session_id)
    ledger = _ledger(session_dir, parsed.budgets)
    engine = EvaluationEngine(
        evaluator=Evaluator(
            candidate_repository=candidate_repository,
            sandbox=workspace.sandbox,
            session_dir=session_dir,
            session_id=parsed.session_id,
        ),
        backends=BackendRegistry(
            {
                backend_id: HarborBackend(backend_config)
                for backend_id, backend_config in parsed.backends.items()
            }
        ),
        database=database,
        database_path=session_dir / "database.json",
        budget_ledger=ledger,
    )
    wandb_sink = None
    if parsed.wandb is not None:
        # Observability must never take down the eval path: on any failure to
        # construct the sink, log and continue without W&B.
        from vero.runtime.wandb import SidecarWandbSink

        try:
            wandb_sink = SidecarWandbSink(
                project=parsed.wandb.project,
                session_id=parsed.session_id,
                session_dir=session_dir,
                entity=parsed.wandb.entity,
                name=parsed.wandb.name,
                group=parsed.wandb.group,
                tags=list(parsed.wandb.tags),
                mode=parsed.wandb.mode,
                notes=parsed.wandb.notes,
                run_id=parsed.wandb.run_id,
                budget_ledger=ledger,
                log_traces=parsed.wandb.log_traces,
            )
            engine.listeners.append(wandb_sink)
        except Exception:
            logger.warning(
                "W&B reporting disabled: the sidecar sink could not be "
                "initialized; the evaluation sidecar continues without it",
                exc_info=True,
            )
            wandb_sink = None

    def _finish_wandb(result: object) -> None:
        # Close the W&B run at finalize (the session's end) with a summary.
        if wandb_sink is None:
            return
        rewards = getattr(result, "rewards", {}) or {}
        summary = {
            "shipped": getattr(result, "shipped", None),
            **{f"reward/{key}": value for key, value in rewards.items()},
        }
        wandb_sink.finish(summary, failed=not getattr(result, "shipped", True))
    transport = GitCandidateTransport(
        workspace=workspace,
        candidate_repository=candidate_repository,
        agent_repo_path=parsed.agent_repo_path,
    )
    baseline = (
        await transport.trusted_candidate(parsed.selection.baseline_version)
        if parsed.selection.baseline_version is not None
        else None
    )
    selection = VerificationSelection(
        mode=parsed.selection.mode,
        backend_id=parsed.selection.backend_id,
        evaluation_set=parsed.selection.evaluation_set,
        objective=parsed.selection.objective,
        baseline_candidate=baseline,
        parameters=parsed.selection.parameters,
        limits=parsed.selection.limits,
        rescore_top_k=parsed.selection.rescore_top_k,
        rescore_attempts=parsed.selection.rescore_attempts,
        baseline_floor=parsed.selection.baseline_floor,
        baseline_selection_score=parsed.selection.baseline_selection_score,
        selection_coverage_threshold=parsed.selection.selection_coverage_threshold,
    )
    initialize_harbor_session_manifest(
        session_dir,
        session_id=parsed.session_id,
        task_name=parsed.task_name,
        task_description=parsed.task_description,
        backends={
            backend_id: engine.backends.resolve(backend_id).provenance
            for backend_id in parsed.backends
        },
        selection=selection,
        targets=parsed.targets,
    )
    sidecar = EvaluationSidecar(
        engine=engine,
        candidate_transport=transport,
        access_policies=parsed.access_policies,
        agent_volume=(
            Path(parsed.agent_volume) if parsed.agent_volume is not None else None
        ),
        admin_volume=Path(parsed.admin_volume),
        submit_enabled=parsed.submit_enabled,
        disclose_budget=parsed.disclose_budget,
        inference_usage_path=(
            Path(parsed.inference_usage_path)
            if parsed.inference_usage_path is not None
            else None
        ),
        inference_limits=parsed.inference_limits,
    )
    await sidecar.initialize_context()
    verifier = CanonicalVerifier(
        engine=engine,
        selection=selection,
        targets=parsed.targets,
        admin_volume=Path(parsed.admin_volume),
        score_baseline=parsed.score_baseline,
        evaluation_drain_timeout_seconds=parsed.evaluation_drain_timeout_seconds,
        on_finalized=_finish_wandb if wandb_sink is not None else None,
    )
    return SidecarComponents(sidecar=sidecar, verifier=verifier)
