from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import JsonValue


@dataclass(frozen=True)
class OptimizationTask:
    """One benchmark target and the evaluation set used to optimize it."""

    project_path: str | Path
    dataset_path: str | Path
    task: str
    module: str | None = None
    partition: str = "test"
    evaluation_budget: int = 8
    total_case_budget: int | None = None
    max_cases_per_evaluation: int | None = None
    score_threshold: float | None = None
    parameters: dict[str, JsonValue] = field(default_factory=dict)
    metric: str = "score"
    direction: str = "maximize"

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("benchmark task name must not be empty")
        if self.evaluation_budget < 1:
            raise ValueError("evaluation_budget must include at least the baseline")
        if self.total_case_budget is not None and self.total_case_budget < 1:
            raise ValueError("total_case_budget must be positive")
        if (
            self.max_cases_per_evaluation is not None
            and self.max_cases_per_evaluation < 1
        ):
            raise ValueError("max_cases_per_evaluation must be positive")

    @property
    def resolved_module(self) -> str:
        if self.module is not None:
            return self.module
        project_name = Path(self.project_path).name
        prefixes = {
            "generic-agent": "generic_agent",
            "web_search_agent": "web_search_agent",
            "tau-bench": "tau_bench",
            "KIRA": "terminus_kira",
            "pharma_summarizer": "pharma_summarizer",
        }
        try:
            package = prefixes[project_name]
        except KeyError as error:
            raise ValueError(
                f"task module must be explicit for target project {project_name!r}"
            ) from error
        return f"{package}.vero_tasks.{self.task}"
