"""Trusted ``vero.toml`` loading for generic program evaluation and optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import toml
from pydantic import Field, JsonValue, model_validator

from vero.evaluation import (
    AllCases,
    BackendRegistry,
    CaseIds,
    CaseRange,
    CommandBackend,
    CommandBackendConfig,
    ConstraintOperator,
    EvaluationDatabase,
    EvaluationEngine,
    EvaluationLimits,
    EvaluationSet,
    Evaluator,
    MetricAggregation,
    MetricConstraint,
    MetricSelector,
    ObjectiveSpec,
)
from vero.evaluation.models import EvaluationModel
from vero.optimization import (
    CommandCandidateProducer,
    CommandCandidateProducerConfig,
    ProgramPolicy,
)
from vero.sandbox import LocalSandbox
from vero.workspace import GitWorkspace


class TargetConfig(EvaluationModel):
    root: str
    ref: str = "HEAD"


class EvaluationConfig(EvaluationModel):
    backend: Literal["command"] = "command"
    harness_root: str
    command: list[str]
    working_directory: str = "."
    environment: dict[str, str] = Field(default_factory=dict)
    passthrough_environment: list[str] = Field(default_factory=list)
    evaluation_set: str = "default"
    partition: str | None = None
    case_ids: list[str] | None = None
    case_start: int = 0
    case_stop: int | None = None
    timeout_seconds: int = Field(default=600, gt=0)
    case_timeout_seconds: int = Field(default=180, gt=0)
    max_concurrency: int = Field(default=100, gt=0)
    use_copy: bool = True
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    seed: int | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> EvaluationConfig:
        if self.case_ids is not None and self.case_stop is not None:
            raise ValueError("case_ids and case range cannot both be configured")
        if self.case_ids is None and self.case_stop is None and self.case_start != 0:
            raise ValueError("case_start requires case_stop")
        return self

    def to_evaluation_set(self) -> EvaluationSet:
        if self.case_ids is not None:
            selection = CaseIds(ids=self.case_ids)
        elif self.case_stop is not None:
            selection = CaseRange(start=self.case_start, stop=self.case_stop)
        else:
            selection = AllCases()
        return EvaluationSet(
            name=self.evaluation_set,
            partition=self.partition,
            selection=selection,
        )

    def to_limits(self) -> EvaluationLimits:
        return EvaluationLimits(
            timeout_seconds=self.timeout_seconds,
            case_timeout_seconds=self.case_timeout_seconds,
            max_concurrency=self.max_concurrency,
        )


class ObjectiveConstraintConfig(EvaluationModel):
    metric: str
    aggregation: MetricAggregation = MetricAggregation.REPORT
    operator: ConstraintOperator
    value: float

    def to_model(self) -> MetricConstraint:
        return MetricConstraint(
            selector=MetricSelector(
                metric=self.metric,
                aggregation=self.aggregation,
            ),
            operator=self.operator,
            value=self.value,
        )


class ObjectiveConfig(EvaluationModel):
    metric: str
    aggregation: MetricAggregation = MetricAggregation.REPORT
    direction: Literal["maximize", "minimize"]
    failure_value: float | None = None
    constraints: list[ObjectiveConstraintConfig] = Field(default_factory=list)

    def to_model(self) -> ObjectiveSpec:
        return ObjectiveSpec(
            selector=MetricSelector(
                metric=self.metric,
                aggregation=self.aggregation,
            ),
            direction=self.direction,
            failure_value=self.failure_value,
            constraints=[constraint.to_model() for constraint in self.constraints],
        )


class OptimizerConfig(EvaluationModel):
    root: str = "."
    command: list[str]
    working_directory: str = "."
    environment: dict[str, str] = Field(default_factory=dict)
    passthrough_environment: list[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=600, gt=0)
    commit_message: str = "Optimize candidate"
    max_candidates: int = Field(default=1, ge=0)


class SessionConfig(EvaluationModel):
    id: str | None = None
    vero_home: str | None = None


class VeroConfig(EvaluationModel):
    target: TargetConfig
    evaluation: EvaluationConfig
    objective: ObjectiveConfig
    optimizer: OptimizerConfig | None = None
    session: SessionConfig = Field(default_factory=SessionConfig)


def load_config(path: Path | str = Path("vero.toml")) -> VeroConfig:
    """Load and validate configuration, resolving trusted roots beside the file."""
    config_path = Path(path).resolve()
    payload = toml.load(config_path)
    config = VeroConfig.model_validate(payload)
    base = config_path.parent
    target_root = (base / config.target.root).resolve()
    harness_root = (base / config.evaluation.harness_root).resolve()
    optimizer = config.optimizer
    updates = {
        "target": config.target.model_copy(update={"root": str(target_root)}),
        "evaluation": config.evaluation.model_copy(
            update={"harness_root": str(harness_root)}
        ),
    }
    if optimizer is not None:
        optimizer_root = (base / optimizer.root).resolve()
        updates["optimizer"] = optimizer.model_copy(
            update={"root": str(optimizer_root)}
        )
    if config.session.vero_home is not None:
        updates["session"] = config.session.model_copy(
            update={"vero_home": str((base / config.session.vero_home).resolve())}
        )
    return config.model_copy(update=updates)


@dataclass(frozen=True)
class ProgramRuntime:
    config: VeroConfig
    policy: ProgramPolicy
    session_id: str
    database_path: Path


async def build_program_runtime(
    config: VeroConfig,
    *,
    require_optimizer: bool = False,
) -> ProgramRuntime:
    target_root = Path(config.target.root).resolve()
    harness_root = Path(config.evaluation.harness_root).resolve()
    if not target_root.exists():
        raise ValueError(f"target root does not exist: {target_root}")
    if not harness_root.exists():
        raise ValueError(f"evaluation harness root does not exist: {harness_root}")
    if harness_root.is_relative_to(target_root):
        raise ValueError("evaluation harness must live outside the editable target")
    if require_optimizer and config.optimizer is None:
        raise ValueError("vero run requires an [optimizer] configuration")

    sandbox = await LocalSandbox.create()
    workspace = await GitWorkspace.from_path(sandbox, str(target_root))
    if await workspace.is_dirty():
        raise ValueError("target workspace must be clean before evaluation")
    base_version = await workspace.resolve_ref(config.target.ref)
    if config.optimizer is not None and await workspace.current_version() != base_version:
        raise ValueError("optimizer target ref must be the workspace's current version")

    session_id = config.session.id or str(uuid4())
    vero_home = (
        Path(config.session.vero_home).resolve()
        if config.session.vero_home is not None
        else Path.home() / ".vero"
    )
    sessions_dir = vero_home / "sessions"
    session_dir = sessions_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    database_path = session_dir / "database.json"
    database = (
        EvaluationDatabase.load_from_file(database_path)
        if database_path.exists()
        else EvaluationDatabase.from_evaluations_dir(
            session_dir / "experiments",
            db_id=session_id,
        )
    )

    command_config = CommandBackendConfig(
        harness_root=str(harness_root),
        command=config.evaluation.command,
        working_directory=config.evaluation.working_directory,
        environment=config.evaluation.environment,
        passthrough_environment=config.evaluation.passthrough_environment,
    )
    backend = CommandBackend(command_config)
    evaluator = Evaluator(
        workspace=workspace,
        sessions_dir=sessions_dir,
        session_id=session_id,
        use_copy=config.evaluation.use_copy,
    )
    engine = EvaluationEngine(
        evaluator=evaluator,
        backends=BackendRegistry({config.evaluation.backend: backend}),
        database=database,
        database_path=database_path,
    )

    producer = None
    max_candidates = 0
    if config.optimizer is not None:
        optimizer_root = Path(config.optimizer.root).resolve()
        if optimizer_root.is_relative_to(target_root):
            raise ValueError("optimizer configuration must live outside the editable target")
        producer = CommandCandidateProducer(
            CommandCandidateProducerConfig(
                root=str(optimizer_root),
                command=config.optimizer.command,
                working_directory=config.optimizer.working_directory,
                environment=config.optimizer.environment,
                passthrough_environment=config.optimizer.passthrough_environment,
                timeout_seconds=config.optimizer.timeout_seconds,
                commit_message=config.optimizer.commit_message,
            )
        )
        max_candidates = config.optimizer.max_candidates

    policy = ProgramPolicy(
        workspace=workspace,
        engine=engine,
        backend_id=config.evaluation.backend,
        evaluation_set=config.evaluation.to_evaluation_set(),
        objective=config.objective.to_model(),
        optimizer=producer,
        parameters=config.evaluation.parameters,
        limits=config.evaluation.to_limits(),
        seed=config.evaluation.seed,
        max_candidates=max_candidates,
        base_version=base_version,
    )
    return ProgramRuntime(
        config=config,
        policy=policy,
        session_id=session_id,
        database_path=database_path,
    )
