"""Standard component factory for compiled Harbor optimization tasks."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from vero.candidate_repository import GitCandidateRepository
from vero.evaluation import (
    BackendRegistry,
    BudgetLedger,
    EvaluationBackend,
    EvaluationBudget,
    EvaluationDatabase,
    EvaluationLimits,
    EvaluationSet,
    Evaluator,
    ObjectiveSpec,
)
from vero.evaluation.command import CommandBackend, CommandBackendConfig
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
from vero.models import StrictModel
from vero.sandbox import LocalSandbox
from vero.workspace import GitWorkspace

logger = logging.getLogger(__name__)


class DeploymentSelection(StrictModel):
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


class SidecarWandbConfig(StrictModel):
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
    # How often the sidecar mirrors the gateway's usage ledger and request log
    # into the run (when the gateway state paths are configured below).
    telemetry_interval_seconds: float = Field(default=30.0, gt=0)


DeploymentBackendConfig = Annotated[
    HarborBackendConfig | CommandBackendConfig,
    Field(discriminator="type"),
]
"""One partition's evaluation backend.

Harbor for a nested ``harbor run`` against a target agent, command for a target
that is scored by running a program (a solver, an index build) rather than by
driving an agent. Both live behind the EvaluationBackend protocol, so
everything above this line — disclosure, budgets, verification — is unchanged.
"""


def _build_backend(config: DeploymentBackendConfig) -> EvaluationBackend:
    if isinstance(config, CommandBackendConfig):
        return CommandBackend(config)
    return HarborBackend(config)


class HarborDeploymentConfig(StrictModel):
    task_name: str = "harbor-session"
    task_description: str = ""
    repo_path: str
    agent_repo_path: str
    session_dir: str
    session_id: str = "trial"
    backends: dict[str, DeploymentBackendConfig]
    access_policies: list[SidecarEvaluationPolicy]
    budgets: list[EvaluationBudget] = Field(default_factory=list)
    selection: DeploymentSelection
    targets: list[VerificationTarget]
    agent_volume: str | None = None
    admin_volume: str
    inference_usage_path: str | None = None
    inference_request_log_dir: str | None = None
    inference_limits: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    submit_enabled: bool = False
    disclose_budget: bool = True
    score_baseline: bool = True
    evaluation_drain_timeout_seconds: float = Field(default=600.0, gt=0)
    wandb: SidecarWandbConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def default_backend_type(cls, value: object) -> object:
        """Treat a backend with no ``type`` as Harbor.

        A discriminated union needs the tag present in the input, but serve.json
        files compiled before the union existed omit it. Defaulting here keeps
        those (and any in-flight run resuming from one) loadable.
        """
        if not isinstance(value, dict):
            return value
        backends = value.get("backends")
        if not isinstance(backends, dict):
            return value
        patched = {
            backend_id: (
                {"type": "harbor", **config}
                if isinstance(config, dict) and "type" not in config
                else config
            )
            for backend_id, config in backends.items()
        }
        return {**value, "backends": patched}

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

    @field_validator("inference_usage_path", "inference_request_log_dir")
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
            *(
                (("inference_request_log_dir", self.inference_request_log_dir),)
                if self.inference_request_log_dir is not None
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
                backend_id: _build_backend(backend_config)
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

    usage_path = (
        Path(parsed.inference_usage_path)
        if parsed.inference_usage_path is not None
        else None
    )
    request_log_dir = (
        Path(parsed.inference_request_log_dir)
        if parsed.inference_request_log_dir is not None
        else None
    )
    telemetry = None
    if wandb_sink is not None and (
        usage_path is not None or request_log_dir is not None
    ):
        from vero.runtime.wandb import InferenceTelemetryPoller

        telemetry = InferenceTelemetryPoller(
            sink=wandb_sink,
            usage_path=usage_path,
            request_log_dir=request_log_dir,
            interval_seconds=parsed.wandb.telemetry_interval_seconds,
        )

    def _export_inference_state() -> None:
        # Preserve the gateway's usage ledger and request log with the session,
        # so /session/export (and the archived run record) carries every
        # request-response the gateway saw.
        destination = session_dir / "artifacts" / "inference"
        try:
            if usage_path is not None and usage_path.is_file():
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(usage_path, destination / usage_path.name)
            if request_log_dir is not None and request_log_dir.is_dir():
                shutil.copytree(
                    request_log_dir,
                    destination / "requests",
                    dirs_exist_ok=True,
                )
        except OSError:
            logger.warning("inference state export failed", exc_info=True)

    def _finalize_session_telemetry(result: object) -> None:
        # Session's end: archive gateway state, flush telemetry (including the
        # still-active request log file), and close the W&B run with a summary.
        _export_inference_state()
        if telemetry is not None:
            telemetry.poll_once(final=True)
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
        on_finalized=_finalize_session_telemetry,
    )
    return SidecarComponents(sidecar=sidecar, verifier=verifier, telemetry=telemetry)


FACTORY_PATH = f"{build_harbor_components.__module__}:{build_harbor_components.__qualname__}"
"""Dotted path the compiled task uses to load this factory.

Derived rather than written out, so renaming this module or function cannot leave
a stale literal in the compose template. A wrong value here would not fail at
import or in the type checker: it fails when the sidecar container starts.
"""
