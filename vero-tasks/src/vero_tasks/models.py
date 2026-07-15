"""Provider-neutral task inputs, outputs, and execution context."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue

TaskT = TypeVar("TaskT")
ParametersT = TypeVar("ParametersT", bound="TaskParameters")


class TaskParameters(BaseModel):
    """Strict base class for typed task-specific parameters."""

    model_config = ConfigDict(extra="forbid")


class TaskContext(BaseModel):
    """Evaluation context visible to task inference and scoring functions."""

    model_config = ConfigDict(extra="forbid")

    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    max_concurrency: int = Field(default=100, ge=1)
    case_timeout_seconds: float = Field(default=180.0, gt=0.0)
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
        return cls(**values)

    def is_error(self) -> bool:
        return self.error is not None or self.eval_error is not None
