from __future__ import annotations

import json
import logging
import traceback
from enum import Enum
from typing import TYPE_CHECKING, Any, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vero.core.constants import default_maximum_score, default_minimum_score
from vero.core.db.dataset import DatasetSample
from vero.core.db.pytest import PyTestReport

if TYPE_CHECKING:
    from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


class TaskOutput(BaseModel):
    """Serializable output of inference on a single task.

    Persisted between the inference and scoring stages, so it must be
    JSON-serializable. An ``Exception`` passed to ``error`` is coerced to its
    string form, with the traceback captured into ``error_traceback``.

    Attributes:
        output: The output of the agent on the task.
        error: An error string (e.g. ``str(exception)``).
        error_traceback: Full traceback string if inference raised.
        execution_trace: An optional list of spans describing the inference process.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: Any = None
    error: str | None = None
    error_traceback: str | None = None
    execution_trace: Sequence[Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_exception_error(cls, data: Any) -> Any:
        """Accept an Exception in ``error`` and convert it to str + traceback."""
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, BaseException):
                data = dict(data)
                data["error"] = str(err)
                if not data.get("error_traceback"):
                    data["error_traceback"] = "".join(
                        traceback.format_exception(type(err), err, err.__traceback__)
                    )
        return data


class TaskResult(BaseModel):
    """Serializable evaluation result for a single task. Used across processes for long-term storage of evaluation results.

    Attributes:
        output: The output of the inference process.
        error: The error message as a string.
        execution_trace: An execution trace of the inference process.
        score: The score of the sample.
        feedback: A feedback message from the evaluation process.
        metrics: A dictionary of metric names to scores.
        eval_error: The evaluation error message as a string.
        eval_trace: An execution trace of the evaluation process.
        error_traceback: The full error traceback as a string.
    """

    output: Any = None
    error: str | None = None
    execution_trace: Sequence[Any] | None = None
    score: float | None = None
    feedback: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    eval_error: str | None = None
    eval_trace: Sequence[Any] | None = None
    error_traceback: str | None = None

    @classmethod
    def from_task_output(cls, task_output: TaskOutput, **kwargs: Any) -> TaskResult:
        """Create a TaskResult from a (serializable) TaskOutput."""
        kwargs["output"] = task_output.output
        kwargs["execution_trace"] = task_output.execution_trace
        if task_output.error is not None:
            kwargs.setdefault("error", task_output.error)
            kwargs.setdefault("error_traceback", task_output.error_traceback)
        return cls(**kwargs)


class SampleResult(TaskResult):
    """Evaluation result for a single sample.

    Attributes:
        id: Unique identifier for this sample result.
        dataset_sample: The dataset sample associated with this result.
        commit: The commit hash of the candidate.
        result_id: The experiment/evaluation result ID.
        input: The raw task input data.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    dataset_sample: DatasetSample
    commit: str | None = None
    result_id: str | None = None
    input: dict[str, Any] | None = None

    @classmethod
    def from_task_result(
        cls, dataset_sample: DatasetSample, task_result: TaskResult, **kwargs: Any
    ) -> SampleResult:
        """Create a SampleResult from a TaskResult."""
        return cls(dataset_sample=dataset_sample, **task_result.model_dump(), **kwargs)

    def is_error(self) -> bool:
        """Returns True if the sample result resulted in an error."""
        return (
            self.error is not None
            or self.eval_error is not None
            or self.score is None
            or self.error_traceback is not None
        )

    def is_scored(self) -> bool:
        """True once the scoring stage has run for this sample (score or eval_error set)."""
        return self.score is not None or self.eval_error is not None

    def as_pandas_series(self, exclude: set[str] | None = None) -> Series:
        """Return the sample result in a pandas representation."""
        import pandas as pd

        data = self.model_dump(exclude=exclude)

        if "execution_trace" in data:
            try:
                data["execution_trace"] = json.dumps(data["execution_trace"])
            except TypeError as e:
                if "not JSON serializable" in str(e):
                    data["execution_trace"] = f"{data['execution_trace']}"
                else:
                    logger.error(
                        f"Failed to serialize execution trace: {e}. The execution trace will be excluded from the pandas series."
                    )
                    data["execution_trace"] = None

        data["is_error"] = self.is_error()
        return pd.json_normalize(data, sep="_").iloc[0]


class ExperimentResultStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ExperimentResult(BaseModel):
    """The result of an experiment run, including evaluation results and the pytest report.

    Attributes:
        id: Unique identifier for this experiment result.
        run_id: The ID of the experiment run.
        status: The status of the experiment result.
        sample_results: A mapping of sample IDs to their results.
        pytest_report: The pytest report for this experiment result.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    status: ExperimentResultStatus
    sample_results: dict[int, SampleResult] = Field(default_factory=dict, repr=False)
    pytest_report: PyTestReport | None = Field(default=None, repr=False)

    @classmethod
    def create_with_status(
        cls,
        error_rate: float,
        run_id: str,
        sample_results: dict[int, SampleResult],
        pytest_report: PyTestReport | None = None,
        id: str | None = None,
    ) -> ExperimentResult:
        """Create an experiment result instance and set the status based on the error rate."""
        kwargs: dict[str, Any] = {
            "run_id": run_id,
            "sample_results": sample_results,
            "pytest_report": pytest_report,
            "status": ExperimentResultStatus.UNKNOWN,
        }
        if id is not None:
            kwargs["id"] = id
        experiment_result = cls(**kwargs)
        if experiment_result.error_rate() >= error_rate:
            experiment_result.status = ExperimentResultStatus.FAILED
        else:
            experiment_result.status = ExperimentResultStatus.SUCCESS
        return experiment_result

    def get_sample_result(self, sample_id: int) -> SampleResult | None:
        """Get a sample result by dataset sample_id."""
        return self.sample_results.get(sample_id)

    @property
    def sample_ids(self) -> list[int]:
        """Get the list of sample_ids in this result."""
        return sorted(self.sample_results.keys())

    def score(self, fill_score: float | None = default_minimum_score) -> float | None:
        """Compute the score of the experiment result.

        Args:
            fill_score (float | None): Score to fill in for sample results with no score

        Returns:
            The score of the experiment result
        """
        import numpy as np

        # if the result has no samples, return the fill score
        if not self.sample_results:
            return fill_score

        # sample results is a dict with at least one element
        if fill_score is not None:
            scores = [
                result.score if result.score is not None else fill_score
                for result in self.sample_results.values()
            ]
        else:
            scores = [
                result.score
                for result in self.sample_results.values()
                if result.score is not None
            ]

        # if empty after filtering, return None
        if not scores:
            return None

        return float(np.mean(scores))

    def confidence_interval(
        self,
        interval: float = 0.95,
        n_bootstrap: int = 100_000,
        fill_score: float | None = default_minimum_score,
        seed: int | None = None,
        max_array_numel: int = 10_000,
    ) -> tuple[float | None, float | None]:
        """Compute the confidence interval of the experiment result.

        Args:
            interval: The confidence interval to compute
            n_bootstrap: The number of bootstrap samples to draw
            fill_score: The score to fill in for missing scores
            seed: The seed to use for the random number generator
            max_array_numel: The maximum number of elements in the array to use for the bootstrap samples

        Returns:
            The lower and upper confidence interval
        """

        import numpy as np

        # if the result has no samples, return the fill score
        if not self.sample_results:
            return (None, None)

        # sample results is a dict with at least one element
        if fill_score is not None:
            scores = [
                result.score if result.score is not None else fill_score
                for result in self.sample_results.values()
            ]
        else:
            scores = [
                result.score
                for result in self.sample_results.values()
                if result.score is not None
            ]

        # if empty after filtering, return None
        if not scores:
            return (None, None)

        rng = np.random.default_rng(seed=seed)
        bootstrap_scores = []

        max_array_numel = max(max_array_numel, len(scores))
        batch_size = max_array_numel // len(scores)
        num_batches = int(np.ceil(n_bootstrap / batch_size))
        scores = np.array(scores, dtype=np.float32)

        for _ in range(num_batches):
            resampled_scores = rng.choice(
                scores, size=(len(scores), batch_size), replace=True
            )
            resampled_scores = resampled_scores.mean(axis=0)
            bootstrap_scores.extend(resampled_scores.tolist())

        bootstrap_scores = np.array(bootstrap_scores)
        lower = np.percentile(bootstrap_scores, (1 - interval) / 2 * 100)
        upper = np.percentile(bootstrap_scores, (1 + interval) / 2 * 100)

        return (float(lower), float(upper))

    def error_rate(self) -> float:
        """Compute the error rate of the experiment result.

        Returns:
            The error rate of the experiment result
        """

        if not self.sample_results:
            return 1.0

        error_count = sum(result.is_error() for result in self.sample_results.values())
        return error_count / len(self.sample_results)

    def sample_results_df(self, exclude: set[str] | None = None) -> "DataFrame | None":
        """Convert sample results to a DataFrame indexed by sample_id.

        Args:
            exclude: List of fields to exclude from the DataFrame

        Returns:
            DataFrame with sample_id as a column, or None if no results
        """
        import pandas as pd

        if not self.sample_results:
            return None

        rows = []
        for sample_id, sample_result in self.sample_results.items():
            row = sample_result.as_pandas_series(exclude=exclude)
            row["sample_id"] = sample_id
            rows.append(row)

        df = pd.DataFrame(rows)
        # Sort by sample_id for consistent ordering
        df = df.sort_values("sample_id").reset_index(drop=True)
        return df

    def sample_results_statistics(
        self,
        nan_score_fill_value: float | None = default_minimum_score,
        convert_lists_to_strings: bool = False,
        as_dict: bool = False,
    ) -> Series | dict | None:
        """Describe the sample results statistics.

        Note: All sample indices in the output (error_sample_ids, min_score_sample_ids,
        max_score_sample_ids) are dataset sample_ids, not positional indices.
        """

        import pandas as pd

        sample_results_df = self.sample_results_df()

        if sample_results_df is None:
            return None

        def fill_scores(
            scores: pd.Series, fill_value: float | None, mask: pd.Series
        ) -> pd.Series:
            """Fill scores with a value, ignoring errors."""
            return scores.fillna(fill_value).where(~mask, other=fill_value)

        def safe_float(x) -> float | None:
            """Convert a value to a float, returning None if the value is NaN."""
            try:
                if pd.isna(x):
                    return None
                return float(x)
            except Exception:
                return None

        def format_ids_as_ranges(ids: list[int]) -> str:
            """Format a list of integers as a compact range string. E.g. [1,2,3,5,7,8] -> "1-3,5,7-8"."""
            if not ids:
                return ""
            sorted_ids = sorted(ids)
            ranges = []
            start = end = sorted_ids[0]
            for val in sorted_ids[1:]:
                if val == end + 1:
                    end = val
                else:
                    ranges.append(f"{start}-{end}" if end > start else str(start))
                    start = end = val
            ranges.append(f"{start}-{end}" if end > start else str(start))
            return ",".join(ranges)

        raw_scores: pd.Series = sample_results_df["score"]
        is_error: pd.Series = sample_results_df["is_error"]
        scores_optimistic = fill_scores(raw_scores, default_maximum_score, is_error)
        scores_pessimistic = fill_scores(raw_scores, default_minimum_score, is_error)

        if nan_score_fill_value is not None:
            scores = fill_scores(raw_scores, nan_score_fill_value, is_error)
        else:
            scores = raw_scores

        sum_raw_score = raw_scores.sum()
        mean_raw_score = raw_scores.mean()
        max_score = scores.max()
        min_score = scores.min()
        mean_score = scores.mean()
        std_score = scores.std()
        mean_score_optimistic = scores_optimistic.mean()
        mean_score_pessimistic = scores_pessimistic.mean()
        lower, upper = self.confidence_interval()
        error_count = is_error.sum()
        error_rate = is_error.mean()
        error_sample_ids = sample_results_df.loc[is_error, "sample_id"].tolist()

        max_score_sample_ids = []
        if pd.notna(max_score):
            max_score_sample_ids = sample_results_df.loc[
                scores == max_score, "sample_id"
            ].tolist()

        min_score_sample_ids = []
        if pd.notna(min_score):
            min_score_sample_ids = sample_results_df.loc[
                scores == min_score, "sample_id"
            ].tolist()

        if convert_lists_to_strings:
            min_score_sample_ids = format_ids_as_ranges(min_score_sample_ids)
            max_score_sample_ids = format_ids_as_ranges(max_score_sample_ids)
            error_sample_ids = format_ids_as_ranges(error_sample_ids)

        numerical_stats = {
            "num_results": len(sample_results_df),
            "error_count": error_count,
            "error_rate": error_rate,
            "sum_of_non_null_scores": sum_raw_score,
            "mean_of_non_null_scores": mean_raw_score,
            "optimistic_mean_score": mean_score_optimistic,  # Errors filled with max (benefit of doubt)
            "pessimistic_mean_score": mean_score_pessimistic,  # Errors filled with min (penalized)
            "mean_score": mean_score,
            "min_score": min_score,
            "max_score": max_score,
            "std_score": std_score,
            "bootstrap_lower_confidence_interval": lower,
            "bootstrap_upper_confidence_interval": upper,
        }

        list_stats = {
            "min_score_sample_ids": min_score_sample_ids,
            "max_score_sample_ids": max_score_sample_ids,
            "error_sample_ids": error_sample_ids,
        }

        if as_dict:
            for key in numerical_stats:
                numerical_stats[key] = safe_float(numerical_stats[key])

            stats = {
                **numerical_stats,
                **list_stats,
            }

            return stats

        stats = {
            **numerical_stats,
            **list_stats,
        }

        return pd.Series(stats)

    @property
    def pytest_report_series(self) -> "Series | None":
        if not self.pytest_report:
            return None
        return self.pytest_report.as_pandas_series()
