"""Canonical, backend-neutral evaluation contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from vero.candidate import Candidate


class EvaluationModel(BaseModel):
    """Strict base model for public evaluation contracts."""

    model_config = ConfigDict(extra="forbid")


def _non_empty(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _optional_non_empty(value: str | None, field_name: str) -> str | None:
    if value is not None:
        _non_empty(value, field_name)
    return value


def _finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    for name, value in metrics.items():
        _non_empty(name, "metric name")
        if not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite")
    return metrics


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class AllCases(EvaluationModel):
    kind: Literal["all"] = "all"


class CaseIds(EvaluationModel):
    kind: Literal["ids"] = "ids"
    ids: list[str]

    @field_validator("ids")
    @classmethod
    def validate_ids(cls, ids: list[str]) -> list[str]:
        if not ids:
            raise ValueError("ids must not be empty")
        for case_id in ids:
            _non_empty(case_id, "case ID")
        if len(set(ids)) != len(ids):
            raise ValueError("case IDs must be unique")
        return ids


class CaseRange(EvaluationModel):
    kind: Literal["range"] = "range"
    stop: int
    start: int = 0

    @model_validator(mode="after")
    def validate_range(self) -> CaseRange:
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.stop <= self.start:
            raise ValueError("stop must be greater than start")
        return self


CaseSelection = Annotated[AllCases | CaseIds | CaseRange, Field(discriminator="kind")]


class EvaluationSet(EvaluationModel):
    """A backend-owned collection of cases and a selection within it."""

    name: str = "default"
    partition: str | None = None
    selection: CaseSelection = Field(default_factory=AllCases)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _non_empty(value, "evaluation set name")

    @field_validator("partition")
    @classmethod
    def validate_partition(cls, value: str | None) -> str | None:
        return _optional_non_empty(value, "partition")

    def budget_key(self, backend_id: str) -> str:
        _non_empty(backend_id, "backend ID")
        return f"{backend_id}:{self.name}:{self.partition or ''}"


class EvaluationPrincipal(str, Enum):
    """Trusted caller class used for authorization and independent metering."""

    AGENT = "agent"
    SYSTEM = "system"
    ADMIN = "admin"


class AgentSelectionMode(str, Enum):
    """How an agent may vary an evaluation definition's base case selection."""

    FIXED = "fixed"
    ARBITRARY = "arbitrary"


class RetryPolicy(EvaluationModel):
    max_attempts: int = Field(default=3, ge=1)
    initial_delay_seconds: float = Field(default=4.0, ge=0.0)
    maximum_delay_seconds: float = Field(default=120.0, ge=0.0)
    multiplier: float = Field(default=2.0, ge=1.0)
    retry_on_timeout: bool = True
    retry_exception_names: list[str] = Field(
        default_factory=lambda: [
            "openai.RateLimitError",
            "anthropic.RateLimitError",
        ]
    )
    retry_status_codes: list[int] = Field(default_factory=lambda: [429, 503, 529])
    retry_message_patterns: list[str] = Field(
        default_factory=lambda: ["rate limit", "too many requests"]
    )

    @model_validator(mode="after")
    def validate_delays(self) -> RetryPolicy:
        if self.maximum_delay_seconds < self.initial_delay_seconds:
            raise ValueError("maximum retry delay cannot be less than initial delay")
        for name in self.retry_exception_names:
            _non_empty(name, "retry exception name")
        if len(set(self.retry_exception_names)) != len(self.retry_exception_names):
            raise ValueError("retry exception names must be unique")
        for status_code in self.retry_status_codes:
            if status_code < 100 or status_code > 599:
                raise ValueError("retry status codes must be between 100 and 599")
        if len(set(self.retry_status_codes)) != len(self.retry_status_codes):
            raise ValueError("retry status codes must be unique")
        for pattern in self.retry_message_patterns:
            _non_empty(pattern, "retry message pattern")
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"invalid retry message pattern {pattern!r}: {error}"
                ) from error
        return self

    @classmethod
    def disabled(cls) -> RetryPolicy:
        """Return an explicit no-retry policy for backends with their own retries."""
        return cls(max_attempts=1)


class EvaluationLimits(EvaluationModel):
    timeout_seconds: float = Field(default=600.0, gt=0.0)
    case_timeout_seconds: float = Field(default=180.0, gt=0.0)
    max_concurrency: int = Field(default=100, ge=1)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)


class EvaluationRequest(EvaluationModel):
    candidate: Candidate
    evaluation_set: EvaluationSet = Field(default_factory=EvaluationSet)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    limits: EvaluationLimits = Field(default_factory=EvaluationLimits)
    seed: int | None = None

    @field_validator("parameters")
    @classmethod
    def validate_parameter_names(
        cls, parameters: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        for name in parameters:
            _non_empty(name, "parameter name")
        return parameters

    def fingerprint(self) -> str:
        """Return the stable identity used to group repeat measurements."""
        payload = {
            "candidate": {
                "id": self.candidate.id,
                "version": self.candidate.version,
            },
            "evaluation_set": self.evaluation_set.model_dump(mode="json"),
            "parameters": self.parameters,
            "limits": self.limits.model_dump(mode="json"),
            "seed": self.seed,
        }
        return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


class CommandEvaluationInput(EvaluationModel):
    """Versioned JSON input passed to an external evaluation harness."""

    schema_version: Literal[1] = 1
    request: EvaluationRequest


class EvaluationArtifact(EvaluationModel):
    path: str
    media_type: str | None = None
    description: str | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        _non_empty(value, "artifact path")
        if "\\" in value or value.startswith("/") or PurePosixPath(value).is_absolute():
            raise ValueError("artifact paths must be relative POSIX paths")
        if any(part in {"", ".", ".."} for part in value.split("/")):
            raise ValueError(
                "artifact paths must not contain empty, '.' or '..' segments"
            )
        return value

    @field_validator("media_type", "description")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return _optional_non_empty(value, "artifact metadata")


class CaseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class CaseError(EvaluationModel):
    message: str
    code: str | None = None
    phase: str | None = None
    attempt: int | None = Field(default=None, ge=1)
    retryable: bool | None = None
    terminal: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        return _non_empty(value, "error message")

    @field_validator("code", "phase")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return _optional_non_empty(value, "error code or phase")


class CaseResult(EvaluationModel):
    case_id: str
    status: CaseStatus
    metrics: dict[str, float] = Field(default_factory=dict)
    input: JsonValue | None = None
    output: JsonValue | None = None
    feedback: str | None = None
    errors: list[CaseError] = Field(default_factory=list)
    execution_trace: list[JsonValue] | None = None
    evaluation_trace: list[JsonValue] | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: list[EvaluationArtifact] = Field(default_factory=list)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        return _non_empty(value, "case ID")

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        return _finite_metrics(value)

    @field_validator("feedback")
    @classmethod
    def validate_feedback(cls, value: str | None) -> str | None:
        return _optional_non_empty(value, "feedback")

    @model_validator(mode="after")
    def validate_status_errors(self) -> CaseResult:
        has_terminal_error = any(error.terminal for error in self.errors)
        if self.status == CaseStatus.ERROR and not has_terminal_error:
            raise ValueError("errored cases require at least one terminal error")
        if self.status != CaseStatus.ERROR and has_terminal_error:
            raise ValueError("only errored cases may contain terminal errors")
        return self


class EvaluationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class EvaluationDiagnostic(EvaluationModel):
    code: str
    message: str
    severity: DiagnosticSeverity
    phase: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("code", "message")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _non_empty(value, "diagnostic code or message")

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, value: str | None) -> str | None:
        return _optional_non_empty(value, "diagnostic phase")


class EvaluationReport(EvaluationModel):
    schema_version: Literal[1] = 1
    status: EvaluationStatus
    metrics: dict[str, float] = Field(default_factory=dict)
    cases: list[CaseResult] = Field(default_factory=list)
    diagnostics: list[EvaluationDiagnostic] = Field(default_factory=list)
    artifacts: list[EvaluationArtifact] = Field(default_factory=list)

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        return _finite_metrics(value)

    @model_validator(mode="after")
    def validate_report(self) -> EvaluationReport:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case IDs must be unique within an evaluation report")
        return self


class BackendProvenance(EvaluationModel):
    name: str
    version: str
    config_digest: str

    @field_validator("name", "version")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return _non_empty(value, "backend name or version")

    @field_validator("config_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("config_digest must be a lowercase SHA-256 digest")
        return value

    @classmethod
    def from_config(
        cls,
        *,
        name: str,
        version: str,
        config: BaseModel | Mapping[str, JsonValue],
    ) -> BackendProvenance:
        config_value = (
            config.model_dump(mode="json")
            if isinstance(config, BaseModel)
            else dict(config)
        )
        payload = {"name": name, "version": version, "config": config_value}
        digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        return cls(name=name, version=version, config_digest=digest)


class MetricAggregation(str, Enum):
    REPORT = "report"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"


class MetricSelector(EvaluationModel):
    metric: str
    aggregation: MetricAggregation = MetricAggregation.REPORT

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, value: str) -> str:
        return _non_empty(value, "metric name")


class ConstraintOperator(str, Enum):
    EQ = "=="
    NE = "!="
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="


class MetricConstraint(EvaluationModel):
    selector: MetricSelector
    operator: ConstraintOperator
    value: float

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("constraint value must be finite")
        return value


class ObjectiveSpec(EvaluationModel):
    selector: MetricSelector
    direction: Literal["maximize", "minimize"]
    failure_value: float | None = None
    constraints: list[MetricConstraint] = Field(default_factory=list)

    @field_validator("failure_value")
    @classmethod
    def validate_failure_value(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("failure_value must be finite")
        return value


class ConstraintViolation(EvaluationModel):
    constraint: MetricConstraint
    observed: float | None
    reason: str

    @field_validator("observed")
    @classmethod
    def validate_observed(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("observed constraint value must be finite")
        return value

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _non_empty(value, "constraint violation reason")


class ObjectiveResult(EvaluationModel):
    value: float | None
    feasible: bool
    violations: list[ConstraintViolation] = Field(default_factory=list)

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("objective value must be finite")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> ObjectiveResult:
        if self.feasible and self.value is None:
            raise ValueError("feasible objective results require a value")
        if self.feasible and self.violations:
            raise ValueError("feasible objective results cannot contain violations")
        return self


class EvaluationRecord(EvaluationModel):
    schema_version: Literal[2] = 2
    id: str
    request: EvaluationRequest
    report: EvaluationReport
    backend_id: str
    backend: BackendProvenance
    principal: EvaluationPrincipal = EvaluationPrincipal.SYSTEM
    objective_spec: ObjectiveSpec | None = None
    objective: ObjectiveResult | None = None
    created_at: datetime
    completed_at: datetime

    @field_validator("id", "backend_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return _non_empty(value, "evaluation identity")

    @field_validator("created_at", "completed_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime) -> datetime:
        return _aware_utc(value, "evaluation timestamp")

    @model_validator(mode="after")
    def validate_record(self) -> EvaluationRecord:
        if (self.objective_spec is None) != (self.objective is None):
            raise ValueError(
                "objective_spec and objective must both be present or absent"
            )
        if self.completed_at < self.created_at:
            raise ValueError("completed_at must not be before created_at")
        return self


class DisclosureLevel(str, Enum):
    FULL = "full"
    AGGREGATE = "aggregate"
    NONE = "none"


class EvaluationSummary(EvaluationModel):
    evaluation_id: str
    candidate_id: str
    candidate_version: str
    backend_id: str
    evaluation_set: EvaluationSet
    status: EvaluationStatus
    metrics: dict[str, float]
    objective: ObjectiveResult | None
    total_cases: int = Field(ge=0)
    successful_cases: int = Field(ge=0)
    errored_cases: int = Field(ge=0)
    skipped_cases: int = Field(ge=0)

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        return _finite_metrics(value)

    @model_validator(mode="after")
    def validate_counts(self) -> EvaluationSummary:
        total = self.successful_cases + self.errored_cases + self.skipped_cases
        if total != self.total_cases:
            raise ValueError("case status counts must sum to total_cases")
        return self


class EvaluationAcknowledgement(EvaluationModel):
    evaluation_id: str
    status: EvaluationStatus


class EvaluationReceipt(EvaluationModel):
    """Bounded agent-facing pointer to filesystem evaluation feedback."""

    evaluation_id: str
    status: EvaluationStatus
    disclosure: DisclosureLevel
    result: EvaluationSummary | EvaluationAcknowledgement
    result_path: str

    @field_validator("evaluation_id")
    @classmethod
    def validate_evaluation_id(cls, value: str) -> str:
        return _non_empty(value, "evaluation ID")

    @field_validator("result_path")
    @classmethod
    def validate_result_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value
            or "\\" in value
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError("receipt result_path must be a safe relative POSIX path")
        return value

    @model_validator(mode="after")
    def validate_projection(self) -> EvaluationReceipt:
        if self.disclosure == DisclosureLevel.NONE:
            if not isinstance(self.result, EvaluationAcknowledgement):
                raise ValueError("none disclosure requires an acknowledgement")
        elif not isinstance(self.result, EvaluationSummary):
            raise ValueError("aggregate and full disclosure require a summary")
        if (
            self.result.evaluation_id != self.evaluation_id
            or self.result.status != self.status
        ):
            raise ValueError("receipt identity and status must match its result")
        return self


class EvaluationAuthorization(EvaluationModel):
    may_evaluate: bool
    may_view: bool | None = None
    meter_budget: bool = True
    disclosure: DisclosureLevel = DisclosureLevel.FULL
    expose_case_resources: bool = False
    reason: str | None = None

    @property
    def viewable(self) -> bool:
        return self.may_evaluate if self.may_view is None else self.may_view

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        return _optional_non_empty(value, "authorization reason")


class EvaluationCost(EvaluationModel):
    runs: int = Field(default=1, ge=1)
    cases: int | None = Field(default=None, ge=0)


class EvaluationBudget(EvaluationModel):
    backend_id: str
    evaluation_set_key: str
    principal: EvaluationPrincipal = EvaluationPrincipal.AGENT
    total_runs: int | None = Field(default=None, ge=0)
    remaining_runs: int | None = Field(default=None, ge=0)
    total_cases: int | None = Field(default=None, ge=0)
    remaining_cases: int | None = Field(default=None, ge=0)
    max_cases_per_run: int | None = Field(default=None, ge=1)

    @field_validator("backend_id", "evaluation_set_key")
    @classmethod
    def validate_keys(cls, value: str) -> str:
        return _non_empty(value, "budget key")

    @model_validator(mode="after")
    def validate_remaining(self) -> EvaluationBudget:
        if (
            self.total_runs is not None
            and self.remaining_runs is not None
            and self.remaining_runs > self.total_runs
        ):
            raise ValueError("remaining_runs cannot exceed total_runs")
        if (
            self.total_cases is not None
            and self.remaining_cases is not None
            and self.remaining_cases > self.total_cases
        ):
            raise ValueError("remaining_cases cannot exceed total_cases")
        return self


class EvaluationAccessPolicy(EvaluationModel):
    """Agent visibility and invocation rights for one named evaluation set."""

    agent_can_evaluate: bool = True
    agent_visible: bool = True
    agent_selection: AgentSelectionMode = AgentSelectionMode.ARBITRARY
    disclosure: DisclosureLevel = DisclosureLevel.FULL
    expose_case_resources: bool = False

    @model_validator(mode="after")
    def validate_visibility(self) -> EvaluationAccessPolicy:
        if self.agent_can_evaluate and not self.agent_visible:
            raise ValueError("agent-evaluable evaluations must be agent-visible")
        if self.expose_case_resources and not self.agent_visible:
            raise ValueError("agent-invisible evaluations cannot expose case resources")
        return self


class EvaluationDefinition(EvaluationModel):
    """One named evaluation together with access and principal-scoped budgets."""

    evaluation_set: EvaluationSet
    access: EvaluationAccessPolicy = Field(default_factory=EvaluationAccessPolicy)
    agent_budget: EvaluationBudget | None = None
    system_budget: EvaluationBudget | None = None

    @model_validator(mode="after")
    def validate_budgets(self) -> EvaluationDefinition:
        expected_suffix = (
            f":{self.evaluation_set.name}:{self.evaluation_set.partition or ''}"
        )
        for principal, budget in (
            (EvaluationPrincipal.AGENT, self.agent_budget),
            (EvaluationPrincipal.SYSTEM, self.system_budget),
        ):
            if budget is None:
                continue
            if budget.principal != principal:
                raise ValueError(
                    f"{principal.value} budget must use principal {principal.value!r}"
                )
            if not budget.evaluation_set_key.endswith(expected_suffix):
                raise ValueError(
                    "evaluation budget key does not match its evaluation set"
                )
        return self


class EvaluationPlan(EvaluationModel):
    """All evaluations available to one optimization protocol."""

    evaluations: list[EvaluationDefinition]
    selection_evaluation: str
    final_evaluation: str | None = None
    evaluate_final_baseline: bool = True

    @model_validator(mode="after")
    def validate_plan(self) -> EvaluationPlan:
        names = [item.evaluation_set.name for item in self.evaluations]
        if not names:
            raise ValueError("evaluation plan must contain at least one evaluation")
        if len(names) != len(set(names)):
            raise ValueError("evaluation plan names must be unique")
        if self.selection_evaluation not in names:
            raise ValueError("selection evaluation is not present in the plan")
        if self.final_evaluation is not None:
            if self.final_evaluation not in names:
                raise ValueError("final evaluation is not present in the plan")
            final = self.get(self.final_evaluation)
            if final.access.agent_can_evaluate or final.access.agent_visible:
                raise ValueError(
                    "final evaluation must be agent-invisible and not agent-evaluable"
                )
        return self

    def get(self, name: str) -> EvaluationDefinition:
        for definition in self.evaluations:
            if definition.evaluation_set.name == name:
                return definition
        raise KeyError(name)

    def for_evaluation_set(
        self,
        evaluation_set: EvaluationSet,
    ) -> EvaluationDefinition | None:
        for definition in self.evaluations:
            owned = definition.evaluation_set
            if (
                owned.name == evaluation_set.name
                and owned.partition == evaluation_set.partition
            ):
                return definition
        return None

    @property
    def selection(self) -> EvaluationDefinition:
        return self.get(self.selection_evaluation)

    @property
    def final(self) -> EvaluationDefinition | None:
        if self.final_evaluation is None:
            return None
        return self.get(self.final_evaluation)

    @property
    def budgets(self) -> list[EvaluationBudget]:
        values: list[EvaluationBudget] = []
        for definition in self.evaluations:
            if definition.agent_budget is not None:
                values.append(definition.agent_budget)
            if definition.system_budget is not None:
                values.append(definition.system_budget)
        return values

    @classmethod
    def single(
        cls,
        evaluation_set: EvaluationSet | None = None,
        *,
        access: EvaluationAccessPolicy | None = None,
        agent_budget: EvaluationBudget | None = None,
        system_budget: EvaluationBudget | None = None,
    ) -> EvaluationPlan:
        resolved = evaluation_set or EvaluationSet()
        return cls(
            evaluations=[
                EvaluationDefinition(
                    evaluation_set=resolved,
                    access=access or EvaluationAccessPolicy(),
                    agent_budget=agent_budget,
                    system_budget=system_budget,
                )
            ],
            selection_evaluation=resolved.name,
        )
