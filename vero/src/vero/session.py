from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from vero.core.dataset import SplitAccess
from vero.core.db import ExperimentDatabase
from vero.core.evaluation import BaseEvaluationParameters
from vero.evaluator import Evaluator
from vero.tools.experiment_runner import SplitBudget  # noqa: E402 — direct import avoids tools/__init__.py
from vero.workspace import Workspace


class BestVersion(BaseModel):
    """Result of get_best_version."""

    commit: str | None = None
    split: str | None = None
    score: float | None = None
    summary: str | None = None


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
    dataset_id: str | None = None
    evaluator: Evaluator | None = None
    db: ExperimentDatabase | None = None
    split_accesses: list[SplitAccess] | None = None
    budget: list[SplitBudget] | None = None
    evaluation_parameters: BaseEvaluationParameters | None = None
    task: str | None = None
    skills: dict[str, Path] = field(default_factory=dict)
    base_version: str | None = None
    base_branch: str | None = None
