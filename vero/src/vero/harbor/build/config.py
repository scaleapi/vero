"""`BuildConfig` — the `vero harbor build -c build.yaml` schema.

Everything the compiler needs to emit a Harbor optimization task. Mode A (vero
runs inference + scoring) and Mode B (nested `harbor run`) share one topology;
the differences are which extras the sidecar bakes and which secrets it needs.

The two modes are DISTINCT types discriminated on `mode`: a Mode-A config that
sets a Mode-B-only field (or vice versa) is a load-time ValidationError, not a
silently-ignored no-op. `extra="forbid"` on the shared base plus the per-mode
subclasses means the wrong-mode key is simply unknown to the resolved variant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


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


class _BuildConfigBase(BaseModel):
    """Fields shared by both modes. Not instantiated directly."""

    # Reject unknown top-level keys so a mistyped lever fails loudly at load
    # time instead of silently disabling the feature: pydantic's default is to
    # ignore extras, which would turn `feeback_transcripts: true` into a no-op.
    # Combined with the Mode-A / Mode-B split below, a wrong-mode field is also
    # "unknown" to the resolved variant, so it fails the same way.
    model_config = ConfigDict(extra="forbid")

    # identity
    name: str = Field(description="Harbor task name, 'org/name' format.")
    description: str = ""

    # the target repo the optimizer edits (baseline in main + sidecar)
    agent_repo: str

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
    max_concurrency: int = 8

    # Wall-clock budget for the VERIFIER phase (Harbor's [verifier] timeout_sec).
    # Finalize is not one eval: it runs up to rescore_top_k shortlist re-scores
    # + 1 floor eval + len(targets) target evals + len(targets) x
    # baseline_score_attempts baseline evals, each a full nested run in Mode B.
    # Sizing this at one eval's duration kills finalize mid-flight and the trial
    # ships NO reward.json. Defaults to `timeout` when unset; size it as
    # (rescore_top_k + 1 + 3 x len(targets)) x a single eval's duration + slack.
    verifier_timeout: int | None = None


class BuildConfigA(_BuildConfigBase):
    """Mode A: vero runs inference + scoring against a saved dataset."""

    mode: Literal["A"] = "A"

    # task name + dataset (+ optional separate task project)
    task: str | None = None
    task_project: str | None = None
    task_module: str | None = None
    dataset: str | None = Field(
        default=None, description="Path to a saved DatasetDict (Mode A)."
    )
    # Per-sample vero-scoring cap. Mode-A only: Mode B's nested `harbor run` uses
    # each task's OWN harbor-configured timeouts, capped only by `timeout`.
    sample_timeout: int = 300


class BuildConfigB(_BuildConfigBase):
    """Mode B: scoring runs in a nested `harbor run`."""

    mode: Literal["B"]

    # HarborConfig kwargs (task_source filled by the compiler from inner_task), the
    # {split: [task_names]} partition, and the inner Harbor task dir baked
    # sidecar-only (the protected benchmark, mirrors Mode A's dataset).
    harbor: dict | None = None
    partition: dict[str, list[str]] | None = None
    inner_task: str | None = None
    # Lever 1: each FAILED sample (reward 0) of an eval carries the tail of its
    # trial transcript in the per-sample `feedback` field. Rides the per-sample
    # result files the sidecar writes ONLY for viewable splits, so it can never
    # surface for non_viewable / no_access tiers.
    feedback_transcripts: bool = False
    feedback_max_bytes: int = 3000
    # Lever 2: the compiled instruction teaches multi-fidelity screening (triage
    # rough ideas on subset evals via num_samples / sample_ids, confirm survivors
    # on the full split). Renders only when the sidecar in the same tree actually
    # accepts subset evals; see the compiler's ctx gate.
    instruct_multifidelity: bool = False
    # Lever 3: each sample's output carries an `attempts` list, one
    # {reward, exception} entry per attempt. Same viewable-only exposure as
    # feedback_transcripts.
    expose_attempt_detail: bool = False


# Discriminated union on `mode`: `vero harbor build` resolves to exactly one
# variant, and a wrong-mode field (e.g. `feedback_transcripts` under mode A) is
# unknown to that variant and rejected by extra="forbid" at load time.
BuildConfig = Annotated[
    BuildConfigA | BuildConfigB, Field(discriminator="mode")
]

# A discriminated union is not a class, so validation goes through a TypeAdapter.
_BuildConfigAdapter: TypeAdapter[BuildConfigA | BuildConfigB] = TypeAdapter(BuildConfig)


def load_build_config(path: Path | str) -> BuildConfigA | BuildConfigB:
    """Load and validate a build.yaml into the correct Mode-A / Mode-B variant.

    Replaces the old ``BuildConfig.from_file`` classmethod: the discriminated
    union is not a class, so the loader lives at module level. Relative
    local-path fields are resolved against the build.yaml's directory, so a
    config is portable regardless of the working directory it's built from.
    """
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    # `mode` defaults to "A" when omitted, preserving the pre-split behavior. The
    # discriminated union needs the tag explicitly, so inject it before validating.
    if isinstance(data, dict):
        data.setdefault("mode", "A")
    base = path.parent
    for field in ("agent_repo", "dataset", "inner_task"):
        val = data.get(field)
        if isinstance(val, str) and not Path(val).is_absolute():
            data[field] = str((base / val).resolve())
    return _BuildConfigAdapter.validate_python(data)
