from __future__ import annotations

from typing import TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from vero.core.db.run import ExperimentRun
from vero.core.utils import RetryConfig

T = TypeVar("T", bound="TaskParameters")


class TaskParameters(BaseModel):
    """Base class for typed parameters accepted by a Python VeroTask."""

    model_config = ConfigDict(extra="forbid")


class BaseEvaluationParameters(BaseModel):
    """Execution controls for the Python VeroTask subprocess protocol."""

    max_concurrency: int = 100
    error_rate_threshold: float = 0.1
    timeout: int = 60 * 10
    sample_timeout: int = 180
    task_params: dict[str, JsonValue] = Field(default_factory=dict)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    use_threading: bool = False

    def parse_task_params(self, model_cls: type[T]) -> T:
        """Validate raw task parameters against a task-owned model."""
        return model_cls.model_validate(self.task_params)


class EvaluationParameters(BaseEvaluationParameters):
    """Deprecated request envelope for the Python VeroTask subprocess.

    General backends receive ``EvaluationRequest``. This type is retained only
    for the historical task runner and third-party task packages.

    Attributes:
        result_id: Unique identifier for this evaluation result.
        run: The details of the experiment run.
        dataset_id: ID of the dataset in the session's dataset mapping.
        task: Task name to execute from vero_tasks module.
        session_id: Session ID to scope cache to a session directory.
    """

    result_id: str = Field(default_factory=lambda: str(uuid4()))
    run: ExperimentRun
    dataset_id: str | None = None
    task: str | None = None
    session_id: str
