"""`BuildConfig` — the `vero harbor build -c build.yaml` schema.

Everything the compiler needs to emit a Harbor optimization task. Mode A (vero
runs inference + scoring) and Mode B (nested `harbor run`) share one topology;
the differences are which extras the sidecar bakes and which secrets it needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SplitAccessSpec(BaseModel):
    split: str
    access: Literal["viewable", "non_viewable", "no_access"]


class BudgetSpec(BaseModel):
    split: str
    total_run_budget: int | None = None
    total_sample_budget: int | None = None


class TargetSpec(BaseModel):
    """A scoring target the verifier evaluates the selected commit on."""

    split: str
    reward_key: str = "reward"
    sample_ids: list[int] | None = None


class BuildConfig(BaseModel):
    """Inputs to `vero harbor build`."""

    # identity
    name: str = Field(description="Harbor task name, 'org/name' format.")
    description: str = ""

    # the target repo the optimizer edits (baseline in main + sidecar)
    agent_repo: str

    # mode A (scoring in vero): task name + dataset (+ optional separate task project)
    mode: Literal["A", "B"] = "A"
    task: str | None = None
    task_project: str | None = None
    task_module: str | None = None
    dataset: str | None = Field(
        default=None, description="Path to a saved DatasetDict (Mode A)."
    )

    # mode B (scoring in nested harbor): HarborConfig kwargs (task_source filled by the
    # compiler from inner_task), the {split: [task_names]} partition, and the inner
    # Harbor task dir baked sidecar-only (the protected benchmark, mirrors Mode A's dataset).
    harbor: dict | None = None
    partition: dict[str, list[str]] | None = None
    inner_task: str | None = None

    # tiers / budget / reward
    splits: list[SplitAccessSpec]
    budgets: list[BudgetSpec] = Field(default_factory=list)
    reward_mode: Literal["submit", "auto_best"] = "auto_best"
    selection_split: str = "validation"
    targets: list[TargetSpec] = Field(default_factory=list)
    submit_enabled: bool = False
    # Also admin-score the unmodified baseline on every target at finalize and
    # write it to <admin_volume>/baseline.json, so a candidate that generalizes
    # WORSE than the untouched repo is visible as a regression.
    score_baseline: bool = False

    # write-access: paths in the target repo the optimizer may NOT edit
    # (the scorer, by default). Applied as unix perms in main before the agent runs.
    read_only_paths: list[str] = Field(default_factory=list)

    # secrets resolved from the host and injected into the SIDECAR only
    secrets: list[str] = Field(default_factory=lambda: ["OPENAI_API_KEY"])

    # image bases
    base_image_main: str = "ghcr.io/astral-sh/uv:python3.12-bookworm"
    base_image_sidecar: str = "ghcr.io/astral-sh/uv:python3.12-bookworm"

    # eval params baked into the ServeConfig
    timeout: int = 1800
    sample_timeout: int = 300
    max_concurrency: int = 8

    @classmethod
    def from_file(cls, path: Path | str) -> BuildConfig:
        path = Path(path).resolve()
        data = yaml.safe_load(path.read_text())
        # Resolve relative local-path fields against the build.yaml's directory, so a
        # config is portable regardless of the working directory it's built from.
        base = path.parent
        for field in ("agent_repo", "dataset", "inner_task"):
            val = data.get(field)
            if isinstance(val, str) and not Path(val).is_absolute():
                data[field] = str((base / val).resolve())
        return cls.model_validate(data)
