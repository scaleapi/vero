"""Declarative configuration for generic program optimization."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, JsonValue, model_validator

from vero.evaluation import (
    AgentSelectionMode,
    AllCases,
    CaseIds,
    CaseRange,
    CommandBackend,
    CommandBackendConfig,
    ConstraintOperator,
    DisclosureLevel,
    EvaluationAccessPolicy,
    EvaluationBudget,
    EvaluationDefinition,
    EvaluationLimits,
    EvaluationPlan,
    EvaluationPrincipal,
    EvaluationSet,
    MetricAggregation,
    MetricConstraint,
    MetricSelector,
    ObjectiveSpec,
    RetryPolicy,
)
from vero.models import StrictModel
from vero.optimization import (
    CommandCandidateProducer,
    CommandCandidateProducerConfig,
    SequentialStrategy,
)
from vero.runtime import (
    OptimizationComponentSpec,
    OptimizationRunSpec,
    OptimizationSession,
    SessionManifest,
    WandbEventSink,
    create_local_optimization_session,
)


class TargetConfig(StrictModel):
    root: str
    ref: str = "HEAD"


class BackendConfig(StrictModel):
    id: str = "command"
    kind: Literal["command"] = "command"
    harness_root: str
    command: list[str]
    working_directory: str = "."
    environment: dict[str, str] = Field(default_factory=dict)
    passthrough_environment: list[str] = Field(default_factory=list)
    staged_inputs: dict[str, str] = Field(default_factory=dict)
    agent_context_inputs: dict[str, list[str]] = Field(default_factory=dict)


class BudgetConfig(StrictModel):
    total_runs: int | None = Field(default=None, ge=0)
    total_cases: int | None = Field(default=None, ge=0)

    def to_model(
        self,
        *,
        backend_id: str,
        evaluation_set: EvaluationSet,
        principal: EvaluationPrincipal,
    ) -> EvaluationBudget:
        return EvaluationBudget(
            backend_id=backend_id,
            evaluation_set_key=evaluation_set.budget_key(backend_id),
            principal=principal,
            total_runs=self.total_runs,
            total_cases=self.total_cases,
        )


class EvaluationConfig(StrictModel):
    name: str
    partition: str | None = None
    case_ids: list[str] | None = None
    case_start: int = Field(default=0, ge=0)
    case_stop: int | None = Field(default=None, ge=1)
    agent_can_evaluate: bool = True
    agent_visible: bool = True
    agent_selection: AgentSelectionMode = AgentSelectionMode.ARBITRARY
    disclosure: DisclosureLevel = DisclosureLevel.FULL
    expose_case_resources: bool = False
    # Omitted resolves to 5 under aggregate disclosure, 1 otherwise.
    min_aggregate_cases: int | None = Field(default=None, ge=1)
    agent_budget: BudgetConfig | None = None
    system_budget: BudgetConfig | None = None

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
            name=self.name,
            partition=self.partition,
            selection=selection,
        )

    def to_definition(self, backend_id: str) -> EvaluationDefinition:
        evaluation_set = self.to_evaluation_set()
        return EvaluationDefinition(
            evaluation_set=evaluation_set,
            access=EvaluationAccessPolicy(
                agent_can_evaluate=self.agent_can_evaluate,
                agent_visible=self.agent_visible,
                agent_selection=self.agent_selection,
                disclosure=self.disclosure,
                expose_case_resources=self.expose_case_resources,
                min_aggregate_cases=self.min_aggregate_cases,
            ),
            agent_budget=(
                self.agent_budget.to_model(
                    backend_id=backend_id,
                    evaluation_set=evaluation_set,
                    principal=EvaluationPrincipal.AGENT,
                )
                if self.agent_budget is not None
                else None
            ),
            system_budget=(
                self.system_budget.to_model(
                    backend_id=backend_id,
                    evaluation_set=evaluation_set,
                    principal=EvaluationPrincipal.SYSTEM,
                )
                if self.system_budget is not None
                else None
            ),
        )


class ProtocolConfig(StrictModel):
    selection_evaluation: str
    final_evaluation: str | None = None
    evaluate_final_baseline: bool = True
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    seed: int | None = None
    timeout_seconds: float = Field(default=600.0, gt=0)
    case_timeout_seconds: float = Field(default=180.0, gt=0)
    evaluation_concurrency: int = Field(default=100, ge=1)
    error_rate_threshold: float | None = Field(default=0.1, gt=0, le=1)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    max_proposals: int = Field(default=1, ge=0)
    max_rounds: int = Field(default=100, ge=1)
    max_concurrency: int = Field(default=1, ge=1)

    def to_limits(self) -> EvaluationLimits:
        return EvaluationLimits(
            timeout_seconds=self.timeout_seconds,
            case_timeout_seconds=self.case_timeout_seconds,
            max_concurrency=self.evaluation_concurrency,
            error_rate_threshold=self.error_rate_threshold,
            retry=self.retry,
        )


class ObjectiveConstraintConfig(StrictModel):
    metric: str
    aggregation: MetricAggregation = MetricAggregation.REPORT
    case_failure_value: float | None = None
    operator: ConstraintOperator
    value: float

    def to_model(self) -> MetricConstraint:
        return MetricConstraint(
            selector=MetricSelector(
                metric=self.metric,
                aggregation=self.aggregation,
                case_failure_value=self.case_failure_value,
            ),
            operator=self.operator,
            value=self.value,
        )


class ObjectiveConfig(StrictModel):
    metric: str
    aggregation: MetricAggregation = MetricAggregation.REPORT
    case_failure_value: float | None = None
    direction: Literal["maximize", "minimize"]
    failure_value: float | None = None
    constraints: list[ObjectiveConstraintConfig] = Field(default_factory=list)

    def to_model(self) -> ObjectiveSpec:
        return ObjectiveSpec(
            selector=MetricSelector(
                metric=self.metric,
                aggregation=self.aggregation,
                case_failure_value=self.case_failure_value,
            ),
            direction=self.direction,
            failure_value=self.failure_value,
            constraints=[constraint.to_model() for constraint in self.constraints],
        )


class BaseOptimizerConfig(StrictModel):
    instruction: str | None = None


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
    model: str | None = None
    max_turns: int = Field(default=200, ge=1)

    @model_validator(mode="after")
    def validate_model(self) -> AgentOptimizerConfig:
        if self.model is not None and not self.model.strip():
            raise ValueError("agent optimizer model must not be empty")
        return self


OptimizerConfig = Annotated[
    CommandOptimizerConfig | AgentOptimizerConfig,
    Field(discriminator="kind"),
]


class SessionConfig(StrictModel):
    id: str | None = None
    directory: str | None = None


class WandbConfig(StrictModel):
    project: str
    run_id: str | None = None
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    mode: Literal["online", "offline", "disabled"] | None = None
    notes: str | None = None
    config: dict[str, JsonValue] = Field(default_factory=dict)


class VeroConfig(StrictModel):
    target: TargetConfig
    backend: BackendConfig
    evaluations: list[EvaluationConfig]
    protocol: ProtocolConfig
    objective: ObjectiveConfig
    optimizer: OptimizerConfig | None = None
    session: SessionConfig = Field(default_factory=SessionConfig)
    wandb: WandbConfig | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> VeroConfig:
        self.to_evaluation_plan()
        names = {evaluation.name for evaluation in self.evaluations}
        unknown_context = sorted(set(self.backend.agent_context_inputs) - names)
        if unknown_context:
            raise ValueError(
                "backend.agent_context_inputs references unknown evaluations: "
                f"{unknown_context}"
            )
        return self

    def to_evaluation_plan(self) -> EvaluationPlan:
        return EvaluationPlan(
            evaluations=[
                evaluation.to_definition(self.backend.id)
                for evaluation in self.evaluations
            ],
            selection_evaluation=self.protocol.selection_evaluation,
            final_evaluation=self.protocol.final_evaluation,
            evaluate_final_baseline=self.protocol.evaluate_final_baseline,
        )

    @staticmethod
    def _component_spec(
        type_name: str,
        payload: dict[str, object],
    ) -> OptimizationComponentSpec:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode()
        return OptimizationComponentSpec(
            type=type_name,
            config_digest=hashlib.sha256(encoded).hexdigest(),
        )

    def to_run_spec(self) -> OptimizationRunSpec:
        producers = {}
        if self.optimizer is not None:
            producer_type = (
                "vero.optimization.command.CommandCandidateProducer"
                if isinstance(self.optimizer, CommandOptimizerConfig)
                else "vero.agents.producer.AgentCandidateProducer"
            )
            producers["default"] = self._component_spec(
                producer_type,
                self.optimizer.model_dump(mode="json"),
            )
        return OptimizationRunSpec(
            max_proposals=(
                self.protocol.max_proposals if self.optimizer is not None else 0
            ),
            max_rounds=(self.protocol.max_rounds if self.optimizer is not None else 1),
            max_concurrency=(
                self.protocol.max_concurrency if self.optimizer is not None else 1
            ),
            strategy=self._component_spec(
                "vero.optimization.strategy.SequentialStrategy",
                {
                    "producer_id": "default",
                    "instruction": (
                        self.optimizer.instruction
                        if self.optimizer is not None
                        else None
                    ),
                },
            ),
            producers=producers,
        )


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
        "backend": config.backend.model_copy(
            update={
                "harness_root": str((base / config.backend.harness_root).resolve()),
                "staged_inputs": {
                    name: str((base / source).resolve())
                    for name, source in config.backend.staged_inputs.items()
                },
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
        if config.model is not None:
            agent.options.model = config.model
    else:
        from vero.agents import VeroAgent

        agent = (
            VeroAgent.for_model(config.model)
            if config.model is not None
            else VeroAgent()
        )
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
    harness_root = Path(config.backend.harness_root)
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
            harness_root=config.backend.harness_root,
            command=config.backend.command,
            working_directory=config.backend.working_directory,
            environment=config.backend.environment,
            passthrough_environment=config.backend.passthrough_environment,
            staged_inputs=config.backend.staged_inputs,
            agent_context_inputs=config.backend.agent_context_inputs,
        )
    )
    session = await create_local_optimization_session(
        project_path=target_root,
        session_dir=session_dir,
        session_id=session_id,
        backend_id=config.backend.id,
        backend=backend,
        objective=config.objective.to_model(),
        evaluation_plan=config.to_evaluation_plan(),
        strategy=SequentialStrategy(
            instruction=(optimizer_config.instruction if optimizer_config else None)
        ),
        producers=producers,
        parameters=config.protocol.parameters,
        limits=config.protocol.to_limits(),
        seed=config.protocol.seed,
        max_proposals=(config.protocol.max_proposals if optimizer_config else 0),
        max_rounds=(config.protocol.max_rounds if optimizer_config else 1),
        max_concurrency=(config.protocol.max_concurrency if optimizer_config else 1),
        base_ref=config.target.ref,
        metadata={
            "config": "vero.toml",
            "project_path": str(target_root),
        },
        run_spec=config.to_run_spec(),
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
                    "vero/evaluation_plan": config.to_evaluation_plan().model_dump(
                        mode="json"
                    ),
                    "vero/objective": config.objective.to_model().model_dump(
                        mode="json"
                    ),
                },
            )
        )
    return ConfiguredRuntime(config=config, session=session, producer=producer)
