"""Standard component factory for compiled Harbor optimization tasks."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from vero.evaluation import (
    BackendRegistry,
    BudgetLedger,
    EvaluationBudget,
    EvaluationDatabase,
    EvaluationModel,
    EvaluationLimits,
    EvaluationSet,
    Evaluator,
    ObjectiveSpec,
)
from vero.evaluation.engine import EvaluationEngine
from vero.harbor.backend import HarborBackend, HarborBackendConfig
from vero.harbor.serve import SidecarComponents
from vero.harbor.sidecar import EvaluationAccessPolicy, EvaluationSidecar
from vero.harbor.transport import GitCandidateTransport
from vero.harbor.verifier import (
    CanonicalVerifier,
    VerificationSelection,
    VerificationTarget,
)
from vero.sandbox import LocalSandbox
from vero.workspace import GitWorkspace


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
    baseline_floor: bool = True

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


class HarborDeploymentConfig(EvaluationModel):
    repo_path: str
    agent_repo_path: str
    session_dir: str
    session_id: str = "trial"
    backends: dict[str, HarborBackendConfig]
    access_policies: list[EvaluationAccessPolicy]
    budgets: list[EvaluationBudget] = Field(default_factory=list)
    selection: DeploymentSelection
    targets: list[VerificationTarget]
    agent_volume: str | None = None
    admin_volume: str
    submit_enabled: bool = False
    score_baseline: bool = True
    use_evaluation_copies: bool = True

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

    @field_validator("agent_volume")
    @classmethod
    def validate_optional_path(cls, value: str | None) -> str | None:
        if value is not None and (
            not value.startswith("/") or ".." in PurePosixPath(value).parts
        ):
            raise ValueError("deployment paths must be absolute")
        return value

    @field_validator("session_id")
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
        ):
            path = PurePosixPath(value)
            if any(
                path == repository or path.is_relative_to(repository)
                for repository in (agent, trusted)
            ):
                raise ValueError(
                    f"{name} must live outside candidate repositories"
                )
        if self.submit_enabled != (self.selection.mode == "submit"):
            raise ValueError("submit_enabled must match selection mode")
        known = set(self.backends)
        referenced = {
            policy.backend_id for policy in self.access_policies
        } | {target.backend_id for target in self.targets}
        if self.selection.backend_id is not None:
            referenced.add(self.selection.backend_id)
        unknown = sorted(referenced - known)
        if unknown:
            raise ValueError(f"deployment references unknown backends: {unknown}")
        return self


def _database(session_dir: Path, session_id: str) -> EvaluationDatabase:
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
            f"evaluation database belongs to {database.id!r}, not {session_id!r}"
        )
    return database


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
    sandbox = await LocalSandbox.create(root=Path(parsed.repo_path).parent)
    workspace = await GitWorkspace.from_path(sandbox, parsed.repo_path)
    database = _database(session_dir, parsed.session_id)
    ledger = _ledger(session_dir, parsed.budgets)
    engine = EvaluationEngine(
        evaluator=Evaluator(
            workspace=workspace,
            session_dir=session_dir,
            session_id=parsed.session_id,
            use_copy=parsed.use_evaluation_copies,
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
    transport = GitCandidateTransport(
        workspace=workspace,
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
    )
    verifier = CanonicalVerifier(
        engine=engine,
        selection=selection,
        targets=parsed.targets,
        admin_volume=Path(parsed.admin_volume),
        score_baseline=parsed.score_baseline,
    )
    return SidecarComponents(sidecar=sidecar, verifier=verifier)
