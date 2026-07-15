from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from vero.core.dataset import SplitAccess
from vero.workspace import Workspace

if TYPE_CHECKING:
    from vero.evaluation import (
        BudgetLedger,
        EvaluationDatabase,
        EvaluationEngine,
        EvaluationLimits,
        EvaluationSet,
        ObjectiveSpec,
    )
    from vero.core.db import ExperimentDatabase
    from vero.policy import Policy


class BestVersion(BaseModel):
    """Result of get_best_version."""

    commit: str | None = None
    split: str | None = None
    score: float | None = None
    summary: str | None = None
    objective_metric: str | None = None
    evaluation_id: str | None = None


@dataclass
class Session:
    """Lightweight context that agents and tools bind to.

    All fields except session_id and project_path are optional.
    Tools bind defensively — they use what's available and skip what's not.
    For testing, create a minimal Session with just the fields you need.
    """

    session_id: str
    project_path: Path
    vero_home: Path | None = None
    instructions: str | None = None
    workspace: Workspace | None = None
    policy: Policy | None = None
    engine: EvaluationEngine | None = None
    database: EvaluationDatabase | None = None
    budget_ledger: BudgetLedger | None = None
    backend_id: str | None = None
    evaluation_set: EvaluationSet | None = None
    objective: ObjectiveSpec | None = None
    limits: EvaluationLimits | None = None
    dataset_id: str | None = None
    split_accesses: list[SplitAccess] | None = None
    task: str | None = None
    skills: dict[str, Path] = field(default_factory=dict)
    base_version: str | None = None
    base_branch: str | None = None

    @property
    def db(self) -> ExperimentDatabase | None:
        """Return a deprecated schema-v1 view without storing parallel state."""
        if self.database is None:
            return None
        from vero.evaluation import evaluation_database_to_experiment_database

        return evaluation_database_to_experiment_database(self.database)
