from __future__ import annotations

from typing import TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from vero.core.db.run import ExperimentRun
from vero.core.utils import RetryConfig

T = TypeVar("T", bound="TaskParameters")


class TaskParameters(BaseModel):
    """Base class for typed task parameters.

    Subclass this in your agent project to define the parameters your task accepts.
    Unknown keys will raise a validation error (extra="forbid"), catching typos early.

    Example::

        class MyAgentParams(TaskParameters):
            model: str = "gpt-4.1-mini"
            temperature: float = 0.0
            num_trials: int = 1

        # In your task function:
        params = evaluation_parameters.parse_task_params(MyAgentParams)
        params.model  # typed, autocomplete works
    """

    model_config = ConfigDict(extra="forbid")


class BaseEvaluationParameters(BaseModel):
    """Base parameters for evaluation. Typically constant for a given evaluation setup.

    Attributes:
        max_concurrency: Maximum allowed number of concurrent async tasks.
        error_rate_threshold: Task error rate threshold.
        timeout: Overall timeout for the evaluation subprocess in seconds.
        sample_timeout: Timeout for a single sample/task in seconds (used inside the eval harness).
        task_params: Task-specific parameters passed to the evaluation.
        retry_config: Retry configuration for transient failures.
        use_threading: Run coroutines in separate threads. Useful for coroutines that block the event loop.
    """

    max_concurrency: int = 100
    error_rate_threshold: float = 0.1
    timeout: int = 60 * 10
    sample_timeout: int = 180
    task_params: dict[str, JsonValue] = Field(default_factory=dict)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    use_threading: bool = False


class EvaluationParameters(BaseEvaluationParameters):
    """All parameters for running an evaluation.

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

    def parse_task_params(self, model_cls: type[T]) -> T:
        """Parse task_params into a typed TaskParameters subclass.

        Validates the raw dict against the model, raising on unknown keys
        if the model uses extra="forbid" (the default for TaskParameters).

        Args:
            model_cls: A TaskParameters subclass defining the expected schema.

        Returns:
            An instance of model_cls populated from task_params.

        Raises:
            pydantic.ValidationError: If task_params contains unknown keys or invalid types.
        """
        return model_cls.model_validate(self.task_params)
