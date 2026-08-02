"""Configuration schema for compiling VeRO optimization tasks for Harbor.

HarborBuildConfig is one flat YAML document, but its fields fall into six groups
that have little to do with each other. Each group is a private base class below,
carrying its own fields, its own docstring, and the validators that concern only
itself. HarborBuildConfig inherits all six, so the YAML stays flat — a build
config written before this split parses identically — while the grammar is
readable one group at a time.

The grouping also removes a duplicate list. The fields that only mean something
for a nested ``harbor run`` used to be enumerated twice: once as declarations and
once in a hand-maintained frozenset. Now they are exactly the members of
_HarborEvaluationFields, and _HARBOR_ONLY_FIELDS is derived from it.

Rules that span groups stay on HarborBuildConfig itself, as named validators.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from vero.evaluation import MetricSelector, ObjectiveSpec
from vero.harbor.build.specs import (
    AgentAccessSpec,
    CommandBackendSpec,
    InferenceGatewaySpec,
    VerificationTargetSpec,
    WandbSpec,
    WorkspaceOverlaySpec,
)
from vero.models import StrictModel


def _require_non_empty(value: str, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value


class _TaskIdentityFields(StrictModel):
    """What is being optimized, and the images and case sets it is built from.

    Attributes:
        name: Task name written into task.toml.
        description: Human-readable task description.
        agent_repo: Absolute path to the editable target. Copied twice, into the
            immutable agent-baseline and the editable agent-seed.
        harbor_requirement: Pinned harbor requirement for the task image. Needed
            for either evaluation backend, since the outer optimizer is a Harbor
            agent in both cases.
        partitions: Partition name to the Harbor task names it holds. Emitted as
            one cases/<partition>.jsonl per partition.
        task_manifest: Optional path to an existing JSON task manifest. When
            given, every task named in partitions must appear in it.
        instruction_template: Optional absolute path to a Jinja template that
            replaces the built-in `instruction.md.j2` for this build. The
            template's own directory is searched first and the built-in
            directory second, so a benchmark-specific template can
            `{% extends "instruction.md.j2" %}` and override only the blocks it
            needs rather than restating the workflow and rules. Exists because
            `description` is the wrong lever when a task contradicts the shared
            framing -- the shell-seed variants are told to *build* a program,
            while the built-in opening line tells them to *improve* one.
        base_image_main: Base image for the main container.
        base_image_sidecar: Base image for the sidecar container.
    """

    name: str
    description: str = ""
    agent_repo: str
    harbor_requirement: str
    partitions: dict[str, list[str]]
    task_manifest: str | None = None
    instruction_template: str | None = None
    base_image_main: str = "ghcr.io/astral-sh/uv:python3.12-bookworm"
    base_image_sidecar: str = "ghcr.io/astral-sh/uv:python3.12-bookworm"

    @field_validator("name", "agent_repo", "base_image_main", "base_image_sidecar")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return _require_non_empty(value, "Harbor build identity")

    @field_validator("agent_repo")
    @classmethod
    def validate_agent_repo(cls, value: str) -> str:
        if not Path(value).is_absolute() or not Path(value).is_dir():
            raise ValueError("agent_repo must be an existing absolute directory")
        return value

    @field_validator("harbor_requirement")
    @classmethod
    def validate_pinned_harbor(cls, value: str) -> str:
        exact = re.search(
            r"(?:^|\s)harbor(?:\[[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*\])?"
            r"\s*==\s*[^*\s,;]+",
            value,
        )
        pinned_git = re.search(r"@[0-9a-f]{7,64}(?:#.*)?$", value)
        if exact is None and pinned_git is None:
            raise ValueError(
                "harbor_requirement must pin an exact version or Git commit"
            )
        return value

    @field_validator("task_manifest")
    @classmethod
    def validate_task_manifest_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip() or not Path(value).is_file():
            raise ValueError("task_manifest must be an existing JSON file")
        return value

    @field_validator("partitions")
    @classmethod
    def validate_partitions(
        cls,
        value: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        if not value:
            raise ValueError("at least one Harbor partition is required")
        for partition, tasks in value.items():
            if re.fullmatch(r"[A-Za-z0-9_.-]+", partition) is None:
                raise ValueError(
                    f"partition {partition!r} must use letters, digits, '.', '_', or '-'"
                )
            if not tasks or any(not task.strip() for task in tasks):
                raise ValueError(f"partition {partition!r} must contain task names")
            if len(tasks) != len(set(tasks)):
                raise ValueError(f"partition {partition!r} contains duplicate tasks")
        return value


# Flags whose value vero owns, in both spellings. harbor takes the last value
# for a key and build-declared args are appended after vero's own, so listing
# only the short form let a build declare `--agent other` and silently replace
# the agent the caller asked for -- a substitution that invalidates a result
# without failing anything.
#
# Matched exactly, never by prefix: `--agent-timeout-multiplier`,
# `--agent-kwarg` and `--environment-build-timeout-multiplier` are all
# legitimate and all begin with a controlled name. Note harbor spells the long
# form of `-e` as `--env`; there is no `--environment` option.
_OUTER_CONTROLLED_FLAGS = frozenset(
    {"-a", "--agent", "-e", "--env", "-m", "--model", "-p", "--path"}
)

# A nested evaluation run is driven entirely by vero, so it reserves the task
# selection and concurrency flags as well. `-o` is included because the long
# form it pairs with, `--jobs-dir`, was already reserved without it.
_EVALUATION_CONTROLLED_FLAGS = _OUTER_CONTROLLED_FLAGS | frozenset(
    {
        "-d",
        "--dataset",
        "-i",
        "--include-task-name",
        "-n",
        "--n-concurrent",
        "-o",
        "--jobs-dir",
        "--agent-import-path",
        "--max-retries",
        "--n-attempts",
    }
)


class _HarborEvaluationFields(StrictModel):
    """Knobs for the nested ``harbor run`` that scores a candidate.

    These, and only these, are the fields that mean nothing to a command
    evaluation backend. Membership of this class *is* the definition:
    _HARBOR_ONLY_FIELDS below is derived from it, and HarborBuildConfig rejects a
    command build that sets any of them rather than ignoring it in silence.

    Attributes:
        task_source: Local path to the task definitions, or a registry reference
            pinned as name@version.
        agent_import_path: Import path of the target agent class Harbor loads.
        model: Default model the target agent runs on.
        environment_name: Harbor environment, e.g. "modal".
        harbor_python_version: Python version for the task image.
        default_index: Package index used for installs.
        n_attempts: Attempts per case in the nested evaluation run.
        max_retries: Per-case retries for the nested `harbor run`, forwarded as
            --max-retries. Harbor already retries non-excluded exceptions with
            backoff; this tunes how hard, so a transient upstream rate-limit
            storm during a mandatory evaluation does not fail otherwise-good
            candidates.
        retry_wait_multiplier: Backoff multiplier. This and the two delays below
            are forwarded in a --config JobConfig snippet, since harbor's CLI
            exposes no backoff flags.
        retry_min_wait_seconds: Minimum backoff delay.
        retry_max_wait_seconds: Maximum backoff delay.
        infrastructure_max_attempts: Attempts for infrastructure-classed
            failures.
        infrastructure_retry_delay_seconds: Delay between those attempts.
        reward_key: Metric key the backend treats as the reward. A command
            harness reports its own reward, so this is Harbor-only.
        aggregate_attempts: Combine repeat attempts by "best" or "mean".
        feedback_transcripts: Include target transcripts in the agent's feedback.
        feedback_max_bytes: Byte cap per feedback transcript.
        expose_attempt_detail: Report per-attempt detail, not just the aggregate.
        extra_harbor_args: Extra flags for the evaluation sub-run. Rejected if
            they override a flag the compiler controls.
        optimizer_harbor_args: Extra flags for the outer harbor run that hosts
            the optimizer trial. Distinct from extra_harbor_args, which tunes
            the nested evaluation sub-run. Rejected if they override a flag
            `vero harbor run` controls.
        task_agent_timeout_seconds: Wall clock declared for the target agent.
            Grouped here rather than with the other timeouts because it bounds
            the target, which only a Harbor backend runs.
        task_environment: Extra environment for the evaluation sub-run, available
            to the task's ${VAR} substitutions. Must not name secrets.
        task_services_use_upstream: Give task-owned model services (user
            simulators, graders) running inside the task containers the real
            upstream via OPENAI_*, while the candidate keeps the metered,
            allow-listed gateway on VERO_AGENT_INFERENCE_*. Needed for
            benchmarks like tau3 whose environment makes its own model calls.
    """

    # Optional because a command evaluation_backend has neither; HarborBuildConfig
    # requires both for a harbor backend.
    task_source: str | None = None
    agent_import_path: str | None = None

    model: str | None = None
    environment_name: str = "modal"
    harbor_python_version: str = "3.12"
    default_index: str = "https://pypi.org/simple"
    n_attempts: int = Field(default=1, ge=1)
    max_retries: int = Field(default=2, ge=0)
    retry_wait_multiplier: float = Field(default=2.0, ge=1.0)
    retry_min_wait_seconds: float = Field(default=4.0, ge=0.0)
    retry_max_wait_seconds: float = Field(default=60.0, ge=0.0)
    infrastructure_max_attempts: int = Field(default=3, ge=1)
    infrastructure_retry_delay_seconds: float = Field(default=5.0, ge=0)
    reward_key: str | None = None
    aggregate_attempts: Literal["best", "mean"] = "mean"
    feedback_transcripts: bool = False
    feedback_max_bytes: int = Field(default=3000, ge=0)
    expose_attempt_detail: bool = False
    extra_harbor_args: list[str] = Field(default_factory=list)
    # Extra flags for the OUTER `harbor run` that hosts the optimizer trial
    # (`vero harbor run`). Distinct from `extra_harbor_args`: that one tunes the
    # nested eval sub-run, this one tunes the environment the optimizer itself
    # lives in. A build declares here what its optimizer trial needs to survive,
    # e.g. `--ek modal_vm_runtime=true` for a long trial whose teardown keeps
    # losing the DinD gRPC stream.
    optimizer_harbor_args: list[str] = Field(default_factory=list)
    task_agent_timeout_seconds: float = Field(default=600.0, gt=0)
    task_environment: dict[str, str] = Field(default_factory=dict)
    task_services_use_upstream: bool = False

    @field_validator("environment_name", "harbor_python_version", "default_index")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return _require_non_empty(value, "Harbor build identity")

    @field_validator("model", "reward_key", "agent_import_path", "task_source")
    @classmethod
    def validate_optional_identity(cls, value: str | None) -> str | None:
        if value is not None:
            _require_non_empty(value, "optional Harbor identity")
        return value

    @field_validator("extra_harbor_args")
    @classmethod
    def validate_extra_harbor_args(cls, value: list[str]) -> list[str]:
        conflicts = [
            argument
            for argument in value
            if argument.split("=", 1)[0] in _EVALUATION_CONTROLLED_FLAGS
        ]
        if conflicts:
            raise ValueError(
                "extra_harbor_args override controlled flags: " + ", ".join(conflicts)
            )
        return value

    @field_validator("optimizer_harbor_args")
    @classmethod
    def validate_optimizer_harbor_args(cls, value: list[str]) -> list[str]:
        conflicts = [
            argument
            for argument in value
            if argument.split("=", 1)[0] in _OUTER_CONTROLLED_FLAGS
        ]
        if conflicts:
            raise ValueError(
                "optimizer_harbor_args override controlled flags: "
                + ", ".join(conflicts)
            )
        return value


# The fields a command build must not set, derived from the class above so the
# two cannot drift. Setting one is a mistake worth reporting: it would be
# silently ignored.
_HARBOR_ONLY_FIELDS = frozenset(_HarborEvaluationFields.model_fields)


class _SearchAndSelectionFields(StrictModel):
    """What the optimizer may measure, what counts as better, and what ships.

    Attributes:
        agent_access: One AgentAccessSpec per partition the optimizer may reach.
            A partition with no entry is held out by the absence of a policy.
        selection_partition: Partition search optimizes against, normally
            validation.
        targets: Trusted final scoring passes; see VerificationTargetSpec.
        evaluation_set_name: Name given to the evaluation set.
        objective: Metric and direction that define fitness.
        reward_mode: "submit" scores the candidate the agent submits;
            "auto_best" scores the best measured candidate.
        baseline_floor: Never ship a candidate scoring below the baseline.
        baseline_selection_score: Pin the baseline's selection score rather than
            measuring it.
        score_baseline: Whether to score the baseline at all.
        rescore_top_k: How many leading candidates the trusted side re-scores.
        rescore_attempts: Re-score repeats per candidate.
        selection_coverage_threshold: Fraction of the selection set an agent
            evaluation must cover to be eligible for auto_best ranking. Below
            this it is too partial to trust for selection.
    """

    agent_access: list[AgentAccessSpec]
    selection_partition: str
    targets: list[VerificationTargetSpec]
    evaluation_set_name: str = "harbor"
    objective: ObjectiveSpec = Field(
        default_factory=lambda: ObjectiveSpec(
            selector=MetricSelector(metric="score"),
            direction="maximize",
        )
    )
    reward_mode: Literal["submit", "auto_best"] = "auto_best"
    baseline_floor: bool = False
    baseline_selection_score: float | None = None
    score_baseline: bool = True
    rescore_top_k: int = Field(default=3, ge=1)
    rescore_attempts: int = Field(default=1, ge=1)
    selection_coverage_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    @field_validator("evaluation_set_name")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return _require_non_empty(value, "Harbor build identity")


class _EvaluationLimitFields(StrictModel):
    """Wall clocks and concurrency for the work the sidecar runs.

    Attributes:
        timeout_seconds: Wall clock for one evaluation.
        case_timeout_seconds: Wall clock for one case.
        max_concurrency: Cases evaluated concurrently.
        error_rate_threshold: Case error fraction above which an evaluation is
            abandoned.
        build_timeout_seconds: Wall clock for building the task's images.
        verifier_timeout_seconds: Wall clock for the trusted verifier. Falls back
            to timeout_seconds.
        evaluation_drain_timeout_seconds: Grace period for finalization to wait
            on already-running agent evaluations before cancelling them.
            Defaults to 600s. Deliberately independent of timeout_seconds, which
            is sized to be unreachable: inheriting it would let one hung sub-run
            stall the held-out score for hours. Expiry is safe — cancellation
            persists terminal records and refunds budgets.
    """

    timeout_seconds: float = Field(default=1800.0, gt=0)
    case_timeout_seconds: float = Field(default=180.0, gt=0)
    max_concurrency: int = Field(default=8, ge=1)
    error_rate_threshold: float | None = Field(default=0.1, gt=0, le=1)
    build_timeout_seconds: int = Field(default=1800, ge=1)
    verifier_timeout_seconds: int | None = Field(default=None, ge=1)
    evaluation_drain_timeout_seconds: float | None = Field(default=None, gt=0)


class _TaskEnvironmentFields(StrictModel):
    """Credentials, model access, and reporting for the containers.

    Attributes:
        secrets: Environment variable names routed into the task. Their presence
            on the build host is checked at compile time unless
            VERO_SKIP_SECRET_CHECK is set.
        inference_gateway: Gateway credential source and per-scope policies; see
            InferenceGatewaySpec. Omit for no metered model access.
        wandb: Trusted-side Weights & Biases reporting from the evaluation
            sidecar. Requires WANDB_API_KEY in secrets, routed to the sidecar and
            never to the agent.
        agent_env: Environment injected into the optimizer agent's own shell
            (setup, install, and run) as harbor --ae KEY=VALUE. Distinct from
            extra_harbor_args, which only reaches the evaluation sub-run, and
            task_environment, which is that sub-run's environment. Use it for
            things like UV_TOOL_BIN_DIR, so `uv tool install` targets a writable
            directory on a non-root sandbox.
        harness_user: Unprivileged OS user that runs the untrusted candidate
            harness in the sidecar, isolating it from trusted state. None
            disables that isolation.
    """

    secrets: list[str] = Field(default_factory=list)
    inference_gateway: InferenceGatewaySpec | None = None
    wandb: WandbSpec | None = None
    agent_env: dict[str, str] = Field(default_factory=dict)
    harness_user: str | None = "harness"

    @field_validator("secrets")
    @classmethod
    def validate_secrets(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("secret environment names must be unique")
        for name in value:
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                raise ValueError(f"invalid secret environment name: {name!r}")
        return value


class _AgentWorkspaceFields(StrictModel):
    """What the optimizer finds in its workspace and is told about the task.

    Attributes:
        read_only_paths: Candidate-relative paths the optimizer may read but not
            edit. Must be safe relative paths.
        workspace_overlays: Host files baked into the optimizer's workspace.
        include_evals_skill: Bake vero's packaged evals skill into the workspace,
            so any coding agent learns the CLI and the .evals layout.
        instruct_multifidelity: Tell the optimizer it allocates its own case
            budget and may evaluate subsets. Requires disclose_budget.
        instruct_exhaust_budget: Tell the optimizer that unspent budget is
            wasted. Requires disclose_budget.
        disclose_budget: Master switch for budget disclosure, not enforcement.
            When False the instruction renders no budget language and the
            sidecar's status omits remaining budgets, so the agent cannot reason
            about its budget at all. The two instruct_ flags above apply only
            when this is True.
    """

    read_only_paths: list[str] = Field(default_factory=list)
    workspace_overlays: list[WorkspaceOverlaySpec] = Field(default_factory=list)
    include_evals_skill: bool = True
    instruct_multifidelity: bool = True
    instruct_exhaust_budget: bool = True
    disclose_budget: bool = True

    @field_validator("read_only_paths")
    @classmethod
    def validate_read_only_paths(cls, value: list[str]) -> list[str]:
        for item in value:
            path = Path(item)
            if (
                not item.strip()
                or path.is_absolute()
                or ".." in path.parts
                or re.fullmatch(r"[A-Za-z0-9_./-]+", item) is None
            ):
                raise ValueError(
                    "read_only_paths must be safe relative candidate paths"
                )
        return value


def _read_manifest(manifest_path: str) -> dict:
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("task_manifest must contain valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("task_manifest must be a JSON object")
    return manifest


def _manifest_source_matches(
    manifest: dict,
    manifest_path: str,
    task_source: str,
) -> bool:
    """Whether a manifest's recorded task_source is the build's task_source."""
    recorded = manifest.get("task_source")
    if recorded == task_source:
        return True
    # A vendored local source is recorded relative to the manifest, while the
    # loader resolves the build's copy to an absolute path once the directory
    # exists; compare resolved locations.
    return (
        isinstance(recorded, str)
        and (Path(manifest_path).parent / recorded).resolve()
        == Path(task_source).resolve()
    )


def _manifest_task_names(manifest: dict) -> list[str]:
    """The task names a manifest declares, rejecting a malformed tasks list."""
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("task_manifest tasks must be a JSON array")
    names: list[str] = []
    for item in tasks:
        name = item.get("name") if isinstance(item, dict) else item
        if not isinstance(name, str) or not name.strip():
            raise ValueError("task_manifest tasks must be names or objects with a name")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("task_manifest contains duplicate task names")
    return names


class HarborBuildConfig(
    _TaskIdentityFields,
    _HarborEvaluationFields,
    _SearchAndSelectionFields,
    _EvaluationLimitFields,
    _TaskEnvironmentFields,
    _AgentWorkspaceFields,
):
    """Everything needed to emit an isolated Harbor optimization task.

    This is the whole grammar of a benchmark's build.yaml: load_harbor_build_config
    parses and validates one, and compile_harbor_task lowers it into a task
    directory. Unknown keys are rejected, so a typo fails the build instead of
    quietly taking a default.

    Most fields come from the six groups this inherits, each documented on its
    own class: _TaskIdentityFields, _HarborEvaluationFields,
    _SearchAndSelectionFields, _EvaluationLimitFields, _TaskEnvironmentFields,
    and _AgentWorkspaceFields. Declared here are only the choice of inner
    evaluation backend and the rules that span groups — each of the latter is a
    named validator below, so a rejected build points at the rule it broke.

    Attributes:
        evaluation_backend: How a candidate is scored. "harbor" drives a target
            agent with a nested `harbor run`; "command" runs a program instead,
            for a target that is not an agent. The outer optimizer is a Harbor
            agent either way.
        command_backend: The scoring program, required when evaluation_backend is
            "command" and rejected otherwise.
    """

    evaluation_backend: Literal["harbor", "command"] = "harbor"
    command_backend: CommandBackendSpec | None = None

    @model_validator(mode="after")
    def validate_harness_isolation(self) -> HarborBuildConfig:
        if self.harness_user is not None and self.task_services_use_upstream:
            raise ValueError(
                "harness_user (harness isolation) is incompatible with "
                "task_services_use_upstream: the raw upstream credential would "
                "reach the isolated harness through its environment. Set "
                "harness_user: null to build task-owned upstream services."
            )
        return self

    @model_validator(mode="after")
    def validate_task_source_is_pinned(self) -> HarborBuildConfig:
        if self.task_source is None or Path(self.task_source).exists():
            return self
        # An explicit version means a registry reference, and it is pinned, so it
        # is valid whether or not it resolves on this filesystem. Checked first
        # because registry names contain "/" too ("gaia/gaia@sha256:..."), so a
        # path-shape test run ahead of this would reject every pinned dataset.
        if "@" in self.task_source:
            return self
        # An unpinned value that looks like a path is almost always vendored data
        # that has not been fetched, not a registry reference missing its version.
        # Saying the latter sends the reader hunting for a version pin to add,
        # which is why a fresh checkout with no tasks/ directory cost real
        # debugging time: the loader leaves an unresolvable relative path
        # untouched, and this validator then reads it as a registry name.
        if self.task_source.startswith((".", "/")) or self.task_source.endswith(
            ("tasks", "tasks/")
        ):
            raise ValueError(
                f"task_source {self.task_source!r} looks like a path but does not "
                "exist (resolved relative to the build file). Vendored task data "
                "is gitignored, so a fresh checkout has to fetch it first -- see "
                "each benchmark's scripts/ directory, e.g. officeqa's "
                "vendor_tasks.sh or browsecomp-plus's build_tasks.py"
            )
        raise ValueError("registry task_source must include an explicit version")

    @model_validator(mode="after")
    def validate_backend_coherence(self) -> HarborBuildConfig:
        """Each backend gets its own fields and only its own."""
        if self.evaluation_backend == "command":
            if self.command_backend is None:
                raise ValueError(
                    "command evaluation_backend requires a command_backend"
                )
            # Reported rather than ignored: these would silently do nothing.
            ignored = sorted(_HARBOR_ONLY_FIELDS & self.model_fields_set)
            if ignored:
                raise ValueError(
                    "these fields only apply to a harbor evaluation_backend: "
                    + ", ".join(ignored)
                )
        else:
            if self.command_backend is not None:
                raise ValueError("command_backend requires evaluation_backend: command")
            missing = sorted(
                name
                for name in ("agent_import_path", "task_source")
                if getattr(self, name) is None
            )
            if missing:
                raise ValueError(
                    "harbor evaluation_backend requires: " + ", ".join(missing)
                )
        return self

    @model_validator(mode="after")
    def validate_partition_references(self) -> HarborBuildConfig:
        """Every partition named elsewhere must exist, and be reachable."""
        known = set(self.partitions)
        if self.selection_partition not in known:
            raise ValueError("selection_partition is not present in partitions")
        access_names = [access.partition for access in self.agent_access]
        if len(access_names) != len(set(access_names)):
            raise ValueError("agent_access partitions must be unique")
        unknown_access = sorted(set(access_names) - known)
        if unknown_access:
            raise ValueError(
                f"agent_access references unknown partitions: {unknown_access}"
            )
        if self.selection_partition not in access_names:
            raise ValueError("selection_partition must be agent-evaluable")
        if not self.targets:
            raise ValueError("at least one verification target is required")
        unknown_targets = sorted({target.partition for target in self.targets} - known)
        if unknown_targets:
            raise ValueError(f"targets reference unknown partitions: {unknown_targets}")
        reward_keys = [target.reward_key for target in self.targets]
        if len(reward_keys) != len(set(reward_keys)):
            raise ValueError("target reward keys must be unique")
        return self

    @model_validator(mode="after")
    def validate_target_attempt_overrides(self) -> HarborBuildConfig:
        """A per-target n_attempts/aggregate_attempts override is only honorable
        on a held-out target: backends are per-partition, so overriding an
        agent-evaluable partition (search or selection) would silently change
        those runs too, and a command backend has no attempts to override."""
        overriding = [
            target
            for target in self.targets
            if target.n_attempts is not None or target.aggregate_attempts is not None
        ]
        if not overriding:
            return self
        if self.evaluation_backend == "command":
            raise ValueError(
                "per-target n_attempts/aggregate_attempts only apply to a harbor "
                "evaluation_backend"
            )
        shared = {access.partition for access in self.agent_access} | {
            self.selection_partition
        }
        bad = sorted({t.partition for t in overriding} & shared)
        if bad:
            raise ValueError(
                "per-target n_attempts/aggregate_attempts cannot override an "
                "agent-evaluable partition (its backend is shared with search/"
                f"selection): {', '.join(bad)}"
            )
        return self

    @model_validator(mode="after")
    def validate_task_manifest_agreement(self) -> HarborBuildConfig:
        """The manifest and the partitions must describe the same task set."""
        if self.task_manifest is None or self.task_source is None:
            return self
        manifest = _read_manifest(self.task_manifest)
        if not _manifest_source_matches(manifest, self.task_manifest, self.task_source):
            raise ValueError(
                "task_manifest task_source does not match build task_source"
            )
        declared = set(_manifest_task_names(manifest))
        selected = {
            task
            for partition_tasks in self.partitions.values()
            for task in partition_tasks
        }
        unknown = sorted(selected - declared)
        if unknown:
            raise ValueError(
                "partitions reference tasks absent from task_manifest: "
                + ", ".join(unknown)
            )
        return self

    @model_validator(mode="after")
    def validate_gateway_credentials_are_not_secrets(self) -> HarborBuildConfig:
        """The upstream credential is the gateway's alone.

        Declaring it as a task secret would deliver it to the containers the
        gateway exists to keep it away from.
        """
        if self.inference_gateway is None:
            return self
        gateway_environment = {self.inference_gateway.upstream_api_key_env}
        if self.inference_gateway.upstream_base_url_env is not None:
            gateway_environment.add(self.inference_gateway.upstream_base_url_env)
        overlap = sorted(set(self.secrets) & gateway_environment)
        if overlap:
            raise ValueError(
                "inference gateway credentials must not also be sidecar secrets: "
                + ", ".join(overlap)
            )
        return self

    @model_validator(mode="after")
    def validate_requested_models_are_allowed(self) -> HarborBuildConfig:
        """Every model that will be requested must be allowed by its scope.

        Otherwise the mismatch surfaces as a gateway 403 at run time — and for a
        verification target, only after search has already spent its budget. A
        command backend requests no models, so there is nothing to reconcile.
        """
        if self.inference_gateway is None or self.evaluation_backend != "harbor":
            return self
        evaluation_models = self.inference_gateway.evaluation.allowed_models
        if self.model is not None and self.model not in evaluation_models:
            raise ValueError(
                f"model {self.model!r} is not in the inference gateway's "
                f"evaluation allowed_models ({', '.join(evaluation_models)})"
            )
        # Finalization runs the target agent too, so it must allow the model each
        # target scores with: its own override, else the task model.
        finalization_models = (
            self.inference_gateway.finalization or self.inference_gateway.evaluation
        ).allowed_models
        for target in self.targets:
            scoring_model = target.model or self.model
            if scoring_model is not None and scoring_model not in finalization_models:
                raise ValueError(
                    f"target {target.partition!r} scores with model "
                    f"{scoring_model!r}, which is not in the inference gateway's "
                    f"finalization allowed_models "
                    f"({', '.join(finalization_models)})"
                )
        return self
