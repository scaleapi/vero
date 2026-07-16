"""Provider-neutral task inputs, outputs, and execution context."""

from __future__ import annotations

import asyncio
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

TaskT = TypeVar("TaskT")
ParametersT = TypeVar("ParametersT", bound="TaskParameters")


class TaskParameters(BaseModel):
    """Strict base class for typed task-specific parameters."""

    model_config = ConfigDict(extra="forbid")


class RetryPolicy(BaseModel):
    """Per-case retry policy from VeRO's backend-neutral request."""

    model_config = ConfigDict(extra="forbid")

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
    def validate_policy(self) -> RetryPolicy:
        if self.maximum_delay_seconds < self.initial_delay_seconds:
            raise ValueError("maximum retry delay cannot be less than initial delay")
        if any(not value.strip() for value in self.retry_exception_names):
            raise ValueError("retry exception names must not be empty")
        if len(set(self.retry_exception_names)) != len(self.retry_exception_names):
            raise ValueError("retry exception names must be unique")
        if any(value < 100 or value > 599 for value in self.retry_status_codes):
            raise ValueError("retry status codes must be between 100 and 599")
        if len(set(self.retry_status_codes)) != len(self.retry_status_codes):
            raise ValueError("retry status codes must be unique")
        for pattern in self.retry_message_patterns:
            if not pattern.strip():
                raise ValueError("retry message patterns must not be empty")
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"invalid retry message pattern {pattern!r}: {error}"
                ) from error
        return self

    def should_retry(self, error: Exception) -> bool:
        if self.retry_on_timeout and isinstance(
            error, (TimeoutError, asyncio.TimeoutError)
        ):
            return True
        exception_names = {
            name
            for error_type in type(error).__mro__
            for name in (
                error_type.__name__,
                f"{error_type.__module__}.{error_type.__qualname__}",
            )
        }
        if any(name in exception_names for name in self.retry_exception_names):
            return True
        status_code = getattr(error, "status_code", getattr(error, "status", None))
        if status_code in self.retry_status_codes:
            return True
        message = str(error)
        return any(
            re.search(pattern, message, flags=re.IGNORECASE)
            for pattern in self.retry_message_patterns
        )

    def delay_after(self, attempt: int) -> float:
        return min(
            self.maximum_delay_seconds,
            self.initial_delay_seconds * self.multiplier ** (attempt - 1),
        )


class TaskAttemptError(BaseModel):
    """One failed inference or evaluation attempt."""

    model_config = ConfigDict(extra="forbid")

    message: str
    phase: Literal["inference", "evaluation"]
    attempt: int = Field(ge=1)
    retryable: bool
    terminal: bool
    traceback: str | None = None


class TaskContext(BaseModel):
    """Evaluation context visible to task inference and scoring functions."""

    model_config = ConfigDict(extra="forbid")

    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    max_concurrency: int = Field(default=100, ge=1)
    case_timeout_seconds: float = Field(default=180.0, gt=0.0)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    seed: int | None = None

    @property
    def task_params(self) -> dict[str, JsonValue]:
        return self.parameters

    def parse_task_params(self, model: type[ParametersT]) -> ParametersT:
        return model.model_validate(self.parameters)


@dataclass
class TaskOutput:
    """In-process output of inference for one evaluation case."""

    output: Any = None
    error: Exception | None = None
    execution_trace: Sequence[Any] | None = None
    attempt_errors: list[TaskAttemptError] = field(default_factory=list)


class TaskResult(BaseModel):
    """Serializable scoring result for one evaluation case."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    output: Any = None
    error: str | None = None
    execution_trace: Sequence[Any] | None = None
    score: float | None = None
    feedback: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    eval_error: str | None = None
    evaluation_trace: Sequence[Any] | None = None
    error_traceback: str | None = None
    evaluation_error_traceback: str | None = None
    attempt_errors: list[TaskAttemptError] = Field(default_factory=list)

    @classmethod
    def from_task_output(
        cls,
        task_output: TaskOutput,
        **values: Any,
    ) -> TaskResult:
        if task_output.error is not None:
            values["error"] = str(task_output.error)
            values["error_traceback"] = "".join(
                traceback.format_exception(
                    type(task_output.error),
                    task_output.error,
                    task_output.error.__traceback__,
                )
            )
        values["output"] = task_output.output
        values["execution_trace"] = task_output.execution_trace
        values.setdefault("attempt_errors", list(task_output.attempt_errors))
        return cls(**values)

    def is_error(self) -> bool:
        return self.error is not None or self.eval_error is not None
