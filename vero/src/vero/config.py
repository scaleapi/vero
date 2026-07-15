"""Declarative configuration for generic program optimization."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, JsonValue, model_validator

from vero.evaluation import (
    AllCases,
    CaseIds,
    CaseRange,
    CommandBackend,
    CommandBackendConfig,
    ConstraintOperator,
    EvaluationLimits,
    EvaluationModel,
    EvaluationSet,
    MetricAggregation,
    MetricConstraint,
    MetricSelector,
    ObjectiveSpec,
)
from vero.optimization import (
    CommandCandidateProducer,
    CommandCandidateProducerConfig,
    SequentialStrategy,
)
from vero.runtime import (
    OptimizationSession,
    SessionManifest,
    WandbEventSink,
    create_local_optimization_session,
)


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
    case_start: int = Field(default=0, ge=0)
    case_stop: int | None = Field(default=None, ge=1)
    timeout_seconds: float = Field(default=600.0, gt=0)
    case_timeout_seconds: float = Field(default=180.0, gt=0)
    max_concurrency: int = Field(default=100, ge=1)
    use_copy: bool = True
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    seed: int | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> EvaluationConfig:
        if self.case_ids is not None and (
            self.case_start != 0 or self.case_stop is not None
        ):
            raise ValueError("case_ids and case range cannot both be configured")
        if self.case_ids is None and self.case_stop is None and self.case_start != 0:
            raise ValueError("case_start requires case_stop")
        if self.case_stop is not None and self.case_stop <= self.case_start:
            raise ValueError("case_stop must be greater than case_start")
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


class BaseOptimizerConfig(EvaluationModel):
    instruction: str | None = None
    max_candidates: int = Field(default=1, ge=0)
    max_rounds: int = Field(default=100, ge=1)
    max_concurrency: int = Field(default=1, ge=1)


class CommandOptimizerConfig(BaseOptimizerConfig):
    kind: Literal["command"] = "command"
    root: str = "."
    command: list[str]
    working_directory: str = "."
    environment: dict[str, str] = Field(default_factory=dict)
    passthrough_environment: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=600.0, gt=0)
    description: str = "Optimize candidate"

    @model_validator(mode="after")
    def validate_command(self) -> CommandOptimizerConfig:
        if not self.command:
            raise ValueError("command optimizer requires a non-empty command")
        return self


class AgentOptimizerConfig(BaseOptimizerConfig):
    kind: Literal["vero", "claude"]
    max_turns: int = Field(default=200, ge=1)


OptimizerConfig = Annotated[
    CommandOptimizerConfig | AgentOptimizerConfig,
    Field(discriminator="kind"),
]


class SessionConfig(EvaluationModel):
    id: str | None = None
    directory: str | None = None


class WandbConfig(EvaluationModel):
    project: str
    run_id: str | None = None
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    mode: Literal["online", "offline", "disabled"] | None = None
    notes: str | None = None
    config: dict[str, JsonValue] = Field(default_factory=dict)


class VeroConfig(EvaluationModel):
    target: TargetConfig
    evaluation: EvaluationConfig
    objective: ObjectiveConfig
    optimizer: OptimizerConfig | None = None
    session: SessionConfig = Field(default_factory=SessionConfig)
    wandb: WandbConfig | None = None


def load_config(path: Path | str = Path("vero.toml")) -> VeroConfig:
    """Load a trusted config and resolve its filesystem paths beside the file."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as config_file:
        config = VeroConfig.model_validate(tomllib.load(config_file))
    base = config_path.parent
    updates: dict[str, object] = {
        "target": config.target.model_copy(
            update={"root": str((base / config.target.root).resolve())}
        ),
        "evaluation": config.evaluation.model_copy(
            update={
                "harness_root": str((base / config.evaluation.harness_root).resolve())
            }
        ),
    }
    if isinstance(config.optimizer, CommandOptimizerConfig):
        updates["optimizer"] = config.optimizer.model_copy(
            update={"root": str((base / config.optimizer.root).resolve())}
        )
    if config.session.directory is not None:
        updates["session"] = config.session.model_copy(
            update={"directory": str((base / config.session.directory).resolve())}
        )
    return config.model_copy(update=updates)


@dataclass(frozen=True)
class ConfiguredRuntime:
    config: VeroConfig
    session: OptimizationSession
    producer: object | None


def _session_identity(config: VeroConfig) -> tuple[str, Path]:
    configured_dir = config.session.directory
    if configured_dir is not None:
        session_dir = Path(configured_dir).expanduser().resolve()
        manifest_path = session_dir / "manifest.json"
        if config.session.id is None and manifest_path.exists():
            session_id = SessionManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            ).id
        else:
            session_id = config.session.id or session_dir.name
        return session_id, session_dir
    session_id = config.session.id or str(uuid4())
    home = Path(os.environ.get("VERO_HOME", "~/.vero")).expanduser().resolve()
    return session_id, home / "sessions" / session_id


def _producer(config: CommandOptimizerConfig | AgentOptimizerConfig):
    if isinstance(config, CommandOptimizerConfig):
        return CommandCandidateProducer(
            CommandCandidateProducerConfig(
                root=config.root,
                command=config.command,
                working_directory=config.working_directory,
                environment=config.environment,
                passthrough_environment=config.passthrough_environment,
                timeout_seconds=config.timeout_seconds,
                description=config.description,
            )
        )
    if config.kind == "claude":
        from vero.agents import ClaudeCodeAgent

        agent = ClaudeCodeAgent()
    else:
        from vero.agents import VeroAgent

        agent = VeroAgent()
    from vero.agents import AgentCandidateProducer

    return AgentCandidateProducer(
        agent,
        prompt=config.instruction,
        max_turns=config.max_turns,
    )


async def build_configured_runtime(
    config: VeroConfig,
    *,
    optimize: bool,
) -> ConfiguredRuntime:
    """Compose a local session from a trusted declarative configuration."""

    target_root = Path(config.target.root)
    harness_root = Path(config.evaluation.harness_root)
    if not target_root.is_dir():
        raise ValueError(f"target root does not exist: {target_root}")
    if not harness_root.is_dir():
        raise ValueError(f"evaluation harness root does not exist: {harness_root}")
    if optimize and config.optimizer is None:
        raise ValueError("vero run requires an [optimizer] configuration")

    optimizer_config = config.optimizer if optimize else None
    producer = _producer(optimizer_config) if optimizer_config is not None else None
    producers = {"default": producer} if producer is not None else {}
    session_id, session_dir = _session_identity(config)
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=config.evaluation.harness_root,
            command=config.evaluation.command,
            working_directory=config.evaluation.working_directory,
            environment=config.evaluation.environment,
            passthrough_environment=config.evaluation.passthrough_environment,
        )
    )
    session = await create_local_optimization_session(
        project_path=target_root,
        session_dir=session_dir,
        session_id=session_id,
        backend_id=config.evaluation.backend,
        backend=backend,
        objective=config.objective.to_model(),
        evaluation_set=config.evaluation.to_evaluation_set(),
        strategy=SequentialStrategy(
            instruction=(optimizer_config.instruction if optimizer_config else None)
        ),
        producers=producers,
        parameters=config.evaluation.parameters,
        limits=config.evaluation.to_limits(),
        seed=config.evaluation.seed,
        max_candidates=(optimizer_config.max_candidates if optimizer_config else 0),
        max_rounds=(optimizer_config.max_rounds if optimizer_config else 1),
        max_concurrency=(optimizer_config.max_concurrency if optimizer_config else 1),
        use_evaluation_copies=config.evaluation.use_copy,
        base_ref=config.target.ref,
        metadata={
            "config": "vero.toml",
            "project_path": str(target_root),
        },
    )
    if config.wandb is not None:
        assert session.events is not None
        session.events.sinks.append(
            WandbEventSink(
                project=config.wandb.project,
                session_id=session.id,
                session_dir=session.session_dir,
                run_id=config.wandb.run_id,
                entity=config.wandb.entity,
                name=config.wandb.name,
                group=config.wandb.group,
                tags=config.wandb.tags,
                mode=config.wandb.mode,
                notes=config.wandb.notes,
                config={
                    **config.wandb.config,
                    "vero/target": str(target_root),
                    "vero/evaluation_set": config.evaluation.to_evaluation_set().model_dump(
                        mode="json"
                    ),
                    "vero/objective": config.objective.to_model().model_dump(
                        mode="json"
                    ),
                },
            )
        )
    return ConfiguredRuntime(config=config, session=session, producer=producer)
