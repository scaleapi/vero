from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vero.tools.experiment_runner import SplitBudget


@dataclass
class OptimizationTask:
    """Specification of the target task the optimizer will operate on."""

    project_path: str | Path
    dataset_path: str | Path
    task: str
    # Explicit budget list — passes through to Policy.budget directly.
    # Use this for non-standard splits (e.g. test-only datasets).
    budget: list[SplitBudget] | None = None
    # Convenience fields — build SplitBudget for train/validation splits.
    # Ignored if `budget` is set.
    train_budget: int | None = None
    validation_budget: int | None = None
    train_sample_budget: int | None = None
    validation_sample_budget: int | None = None
    batch_size: int | None = None
    score_threshold: float | None = None
    resource_namespace: str | None = None
    eval_split: str = "test"
