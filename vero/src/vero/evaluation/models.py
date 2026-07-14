"""Canonical models for evaluating versioned program candidates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
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

from vero.core.db.candidate import Candidate
from vero.core.utils import RetryConfig


class EvaluationModel(BaseModel):
    """Strict base model for versioned evaluation contracts."""

    model_config = ConfigDict(extra="forbid")


def _non_empty(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _optional_non_empty(value: str | None, field_name: str) -> str | None:
    if value is not None:
        _non_empty(value, field_name)
    return value


def _validate_finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    for name, value in metrics.items():
        _non_empty(name, "metric name")
        if not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite")
    return metrics


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


CaseSelection = Annotated[
    AllCases | CaseIds | CaseRange,
    Field(discriminator="kind"),
]


class EvaluationSet(EvaluationModel):
    name: str = "default"
    partition: str | None = None
    selection: CaseSelection = Field(default_factory=AllCases)

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        return _non_empty(name, "evaluation set name")

    @field_validator("partition")
    @classmethod
    def validate_partition(cls, partition: str | None) -> str | None:
        return _optional_non_empty(partition, "partition")

    def budget_key(self, backend_id: str) -> str:
        """Return the stable key used to meter this backend/set combination."""
        _non_empty(backend_id, "backend ID")
        return f"{backend_id}:{self.name}:{self.partition or ''}"


class EvaluationLimits(EvaluationModel):
    timeout_seconds: int = Field(default=600, gt=0)
    case_timeout_seconds: int = Field(default=180, gt=0)
    max_concurrency: int = Field(default=100, gt=0)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)


class EvaluationRequest(EvaluationModel):
    candidate: Candidate
    evaluation_set: EvaluationSet
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
        """Hash the canonical JSON form used for repeat-measurement grouping."""
        payload = _canonical_json(self.model_dump(mode="json"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CommandEvaluationInput(EvaluationModel):
    """Versioned, backend-neutral file input passed to command harnesses."""

    schema_version: Literal["1"] = "1"
    request: EvaluationRequest


class EvaluationArtifact(EvaluationModel):
    path: str
    media_type: str | None = None
    description: str | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        _non_empty(path, "artifact path")
        if "\\" in path:
            raise ValueError("artifact paths must use POSIX separators")
        if path.startswith("/") or PurePosixPath(path).is_absolute():
            raise ValueError("artifact paths must be relative")
        parts = path.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("artifact paths must not contain empty, '.' or '..' segments")
        return path

    @field_validator("media_type")
    @classmethod
    def validate_media_type(cls, value: str | None) -> str | None:
        return _optional_non_empty(value, "media type")

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        return _optional_non_empty(value, "artifact description")


class CaseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class CaseError(EvaluationModel):
    message: str
    code: str | None = None
    phase: str | None = None
    attempt: int | None = Field(default=None, gt=0)
    retryable: bool | None = None
    terminal: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def validate_message(cls, message: str) -> str:
        return _non_empty(message, "error message")

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
    def validate_case_id(cls, case_id: str) -> str:
        return _non_empty(case_id, "case ID")

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, metrics: dict[str, float]) -> dict[str, float]:
        return _validate_finite_metrics(metrics)

    @field_validator("feedback")
    @classmethod
    def validate_feedback(cls, value: str | None) -> str | None:
        return _optional_non_empty(value, "feedback")

    @model_validator(mode="after")
    def validate_status_errors(self) -> CaseResult:
        has_terminal_error = any(error.terminal for error in self.errors)
        if self.status == CaseStatus.SUCCESS and has_terminal_error:
            raise ValueError("successful cases must not have terminal errors")
        if self.status == CaseStatus.ERROR and not has_terminal_error:
            raise ValueError("errored cases require at least one terminal error")
        if self.status == CaseStatus.SKIPPED and has_terminal_error:
            raise ValueError("skipped cases must not have terminal errors")
        return self


class EvaluationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


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
    schema_version: Literal["1"] = "1"
    status: EvaluationStatus
    metrics: dict[str, float] = Field(default_factory=dict)
    cases: list[CaseResult] = Field(default_factory=list)
    diagnostics: list[EvaluationDiagnostic] = Field(default_factory=list)
    artifacts: list[EvaluationArtifact] = Field(default_factory=list)
    error: str | None = None

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, metrics: dict[str, float]) -> dict[str, float]:
        return _validate_finite_metrics(metrics)

    @field_validator("error")
    @classmethod
    def validate_error(cls, value: str | None) -> str | None:
        return _optional_non_empty(value, "evaluation error")

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> EvaluationReport:
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
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
    def validate_config_digest(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
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
            config.model_dump(mode="json") if isinstance(config, BaseModel) else dict(config)
        )
        payload = {"name": name, "version": version, "config": config_value}
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
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
    def validate_metric(cls, metric: str) -> str:
        return _non_empty(metric, "metric name")


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
    def validate_reason(cls, reason: str) -> str:
        return _non_empty(reason, "constraint violation reason")


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
    def validate_feasible_result(self) -> ObjectiveResult:
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
    objective_spec: ObjectiveSpec | None = None
    objective: ObjectiveResult | None = None
    created_at: datetime
    completed_at: datetime

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _non_empty(value, "evaluation record ID")

    @field_validator("backend_id")
    @classmethod
    def validate_backend_id(cls, value: str) -> str:
        return _non_empty(value, "backend ID")

    @model_validator(mode="after")
    def validate_record(self) -> EvaluationRecord:
        if (self.objective_spec is None) != (self.objective is None):
            raise ValueError("objective_spec and objective must both be present or absent")
        try:
            completed_before_created = self.completed_at < self.created_at
        except TypeError as error:
            raise ValueError("record timestamps must use compatible timezones") from error
        if completed_before_created:
            raise ValueError("completed_at must not be before created_at")
        return self


class DisclosureLevel(str, Enum):
    FULL = "full"
    AGGREGATE = "aggregate"
    NONE = "none"


class EvaluationSummary(EvaluationModel):
    evaluation_id: str
    candidate_commit: str
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
    def validate_metrics(cls, metrics: dict[str, float]) -> dict[str, float]:
        return _validate_finite_metrics(metrics)

    @model_validator(mode="after")
    def validate_case_counts(self) -> EvaluationSummary:
        count_sum = self.successful_cases + self.errored_cases + self.skipped_cases
        if count_sum != self.total_cases:
            raise ValueError("case status counts must sum to total_cases")
        return self


class EvaluationAcknowledgement(EvaluationModel):
    evaluation_id: str
    status: EvaluationStatus


class EvaluationAuthorization(EvaluationModel):
    may_evaluate: bool
    meter_budget: bool = True
    disclosure: DisclosureLevel = DisclosureLevel.FULL
    reason: str | None = None

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
    def validate_remaining_values(self) -> EvaluationBudget:
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
