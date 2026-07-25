"""Configuration schema for compiling VeRO optimization tasks for Harbor."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from vero.evaluation import (
    DisclosureLevel,
    EvaluationAccessPolicy,
    MetricSelector,
    ObjectiveSpec,
)
from vero.models import StrictModel


class AgentAccessSpec(StrictModel):
    """The optimizer agent's access to one named evaluation partition.

    A build declares one spec per partition; to_access_policy translates it into
    the runtime EvaluationAccessPolicy the sidecar enforces. This governs what
    the optimizer may see and spend while searching, as opposed to how the
    trusted side finally scores a candidate (see VerificationTargetSpec).

    Attributes:
        partition: Partition this access applies to, e.g. "validation".
        disclosure: How much of a result the agent sees — aggregate score vs.
            per-case detail.
        expose_case_resources: Whether each case's input files are materialized
            into the agent's workspace.
        min_aggregate_cases: Smallest number of cases an aggregate score may
            cover, so the agent cannot request a subset small enough to reveal an
            individual case's result. Only consulted when disclosure is AGGREGATE.
        total_runs: Optional cap on evaluation runs against this partition.
        total_cases: Optional cap on cases scored against this partition.
    """

    partition: str
    disclosure: DisclosureLevel = DisclosureLevel.AGGREGATE
    expose_case_resources: bool = False
    min_aggregate_cases: int = Field(default=5, ge=1)
    total_runs: int | None = Field(default=None, ge=0)
    total_cases: int | None = Field(default=None, ge=0)

    @field_validator("partition")
    @classmethod
    def validate_partition(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("agent access partition must not be empty")
        return value

    def to_access_policy(self) -> EvaluationAccessPolicy:
        """The single typed translation from build spec to runtime policy."""
        return EvaluationAccessPolicy(
            disclosure=self.disclosure,
            expose_case_resources=self.expose_case_resources,
            min_aggregate_cases=self.min_aggregate_cases,
        )


class VerificationTargetSpec(StrictModel):
    """One trusted final scoring pass the verifier runs on a chosen candidate.

    Distinct from the evaluations the optimizer runs during search: after search
    picks a best candidate on the selection partition, the trusted verifier
    scores it against these targets (normally the held-out test partition) to
    produce the final reward. A build declares one spec per pass.

    Attributes:
        partition: Partition to score on, e.g. "test".
        reward_key: Metric key in the report that is the reward.
        model: Optional model override for this pass; passed through as
            harbor_model_override so final scoring can differ from search.
        parameters: Extra backend parameters for the pass.
        failure_value: Score assigned when the pass fails.
        baseline_reward: Pin the seed's reward on this target to skip scoring
            the immutable baseline every run.
        max_attempts: Number of scoring attempts before accepting failure.
    """

    partition: str
    reward_key: str = "reward"
    model: str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    failure_value: float = 0.0
    baseline_reward: float | None = None
    max_attempts: int = Field(default=1, ge=1)

    @field_validator("partition", "reward_key")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("verification target identity must not be empty")
        return value

    @field_validator("failure_value")
    @classmethod
    def validate_failure_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("target failure_value must be finite")
        return value


class InferenceBudgetSpec(StrictModel):
    """Routing policy and optional limits for one inference-gateway scope.

    The compiler lowers each spec into an InferenceScopeConfig, adding the
    SHA-256 digest of a freshly minted per-scope token. The gateway then checks
    the presented token, the requested model, and the running budget on every
    proxied request; limits are held server-side in a ledger written through to
    disk, so they survive a gateway restart.

    Attributes:
        allowed_models: Models this scope may request. A request naming anything
            else is refused with 403 model_denied.
        max_requests: Cap on proxied requests; unlimited when omitted.
        max_tokens: Cap on cumulative tokens; unlimited when omitted. Checked
            before a request starts, so a single request can overshoot it.
        max_concurrency: Requests this scope may have in flight at once.
    """

    allowed_models: list[str]
    max_requests: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_concurrency: int = Field(default=8, ge=1)

    @field_validator("allowed_models")
    @classmethod
    def validate_models(cls, value: list[str]) -> list[str]:
        if not value or any(not model.strip() for model in value):
            raise ValueError("allowed_models must contain non-empty model names")
        if len(value) != len(set(value)):
            raise ValueError("allowed_models must be unique")
        return value


class InferenceGatewaySpec(StrictModel):
    """Credential source and independent producer/evaluator policies.

    All model access in a compiled task goes through the gateway, which holds
    the one upstream credential and hands each consumer a separate token. The
    optimizer never sees the evaluation or finalization tokens, so it cannot
    spend from those pools.

    The per-scope allow-lists keep the target on its fixed evaluation model, but
    only against a non-adversarial optimizer: the scopes share a gateway host and
    are split by URL path, so an optimizer that smuggles its own token into the
    candidate can reach the producer scope from the eval sandbox. See the proxy
    handler in vero.harbor.inference for the full note.

    Attributes:
        upstream_api_key_env: Environment variable on the build host holding the
            real upstream API key.
        upstream_base_url_env: Environment variable holding the upstream base
            URL; None to always use default_upstream_base_url.
        default_upstream_base_url: Upstream used when no base-URL variable is set.
        producer: Policy for the optimizer's own model calls.
        evaluation: Policy for the target agent under evaluation.
        finalization: Reserved policy for trusted finalization (admin re-score
            and verification targets). Defaults to a copy of evaluation, giving
            the mandatory re-score a pool that search evaluations cannot starve.
        log_requests: Capture every proxied or denied request as JSONL on the
            gateway state volume. Never agent-visible; the trusted sidecar
            mirrors it to Weights & Biases and the session archive.
        request_log_body_bytes: Head+tail budget per logged body, in bytes.
        request_log_attribution: Experimental. Stamp log records with
            provider-agnostic conversation threads for post-hoc per-trial
            accounting.
    """

    upstream_api_key_env: str = "OPENAI_API_KEY"
    upstream_base_url_env: str | None = "OPENAI_BASE_URL"
    default_upstream_base_url: str = "https://api.openai.com/v1"
    producer: InferenceBudgetSpec
    evaluation: InferenceBudgetSpec
    finalization: InferenceBudgetSpec | None = None
    log_requests: bool = True
    request_log_body_bytes: int = Field(default=16384, ge=0)
    request_log_attribution: bool = False

    @field_validator("upstream_api_key_env", "upstream_base_url_env")
    @classmethod
    def validate_environment_name(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
            raise ValueError("gateway environment names must be valid identifiers")
        return value

    @field_validator("default_upstream_base_url")
    @classmethod
    def validate_upstream_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("default_upstream_base_url must be HTTP(S)")
        return value.rstrip("/")


class WorkspaceOverlaySpec(StrictModel):
    """A host file or directory to copy into the compiled task's agent workspace.

    General-purpose filesystem injection: bake anything (agent definitions,
    skills, config, data) into the optimizer's /work/agent at build time.

    Attributes:
        source: Host path, resolved relative to the build YAML. Must exist. A
            directory contributes its contents; a file lands as dest/<name>.
        dest: Where the contents land, relative to the workspace root. Must be a
            safe relative path; "." is the root itself.
    """

    source: str
    dest: str = "."

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        if not value.strip() or not Path(value).exists():
            raise ValueError("overlay source must be an existing file or directory")
        return value

    @field_validator("dest")
    @classmethod
    def validate_dest(cls, value: str) -> str:
        candidate = value.strip()
        path = Path(candidate)
        if candidate == ".":
            return candidate
        if (
            not candidate
            or path.is_absolute()
            or ".." in path.parts
            or re.fullmatch(r"[A-Za-z0-9_./-]+", candidate) is None
        ):
            raise ValueError(
                "overlay dest must be a safe relative path within the workspace"
            )
        return candidate


class WandbSpec(StrictModel):
    """Weights & Biases reporting settings, emitted into the sidecar's serve.json.

    Reporting runs on the trusted side only. The field names match the sidecar's
    SidecarWandbConfig, which consumes them verbatim.

    Attributes:
        project: Weights & Biases project to report into.
        entity: Owning user or team; the account default when omitted.
        name: Display name for the run.
        group: Group to file the run under.
        tags: Tags applied to the run.
        mode: Client mode, e.g. "online" or "offline".
        notes: Free-text note attached to the run.
        run_id: Resume into this existing run instead of creating one.
        log_traces: Upload each evaluation's trace artifacts. Off by default
            because they are large.
    """

    project: str
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    mode: str | None = None
    notes: str | None = None
    run_id: str | None = None
    log_traces: bool = False


class CommandBackendSpec(StrictModel):
    """Inner evaluation that scores a candidate by running a program.

    Use this when the outer loop is still a Harbor coding agent but the target is
    not an agent — a solver, an index build, a data pipeline — so there is nothing
    to drive with a nested `harbor run`. The compiler copies harness_source into
    the sidecar and points the backend's harness_root at it.

    Attributes:
        harness_source: Host directory holding the scoring program, resolved
            relative to the build YAML. Must exist.
        command: Argument vector to run. Supports the placeholders {harness},
            {workspace}, {request}, {report}, and {artifacts}.
        working_directory: Directory to run the command from.
        environment: Environment variables set for the command.
        passthrough_environment: Names forwarded from the sidecar's environment.
        staged_inputs: Extra files staged into the command's workspace.
        agent_context_inputs: Per-case files published into the agent context.
    """

    harness_source: str
    command: list[str]
    working_directory: str = "."
    environment: dict[str, str] = Field(default_factory=dict)
    passthrough_environment: list[str] = Field(default_factory=list)
    staged_inputs: dict[str, str] = Field(default_factory=dict)
    agent_context_inputs: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("harness_source")
    @classmethod
    def validate_harness_source(cls, value: str) -> str:
        if not value.strip() or not Path(value).is_dir():
            raise ValueError("harness_source must be an existing directory")
        return value

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: list[str]) -> list[str]:
        if not value or any(not part.strip() for part in value):
            raise ValueError("command must contain non-empty arguments")
        return value


# Fields that only mean something for a nested `harbor run`. Setting one on a
# command build is a mistake worth reporting: it would be silently ignored.
_HARBOR_ONLY_FIELDS = frozenset(
    {
        "agent_import_path",
        "model",
        "environment_name",
        "harbor_python_version",
        "default_index",
        "n_attempts",
        "max_retries",
        "retry_wait_multiplier",
        "retry_min_wait_seconds",
        "retry_max_wait_seconds",
        "infrastructure_max_attempts",
        "infrastructure_retry_delay_seconds",
        "aggregate_attempts",
        "feedback_transcripts",
        "feedback_max_bytes",
        "expose_attempt_detail",
        "extra_harbor_args",
        "task_agent_timeout_seconds",
        "task_environment",
        "task_services_use_upstream",
    }
)


class HarborBuildConfig(StrictModel):
    """Everything needed to emit an isolated Harbor optimization task.

    This is the whole grammar of a benchmark's build.yaml: load_harbor_build_config
    parses and validates one, and compile_harbor_task lowers it into a task
    directory. Unknown keys are rejected, so a typo fails the build instead of
    quietly taking a default.

    Two cross-field rules are checked after parsing. harness_user cannot be
    combined with task_services_use_upstream, because the raw upstream credential
    would then reach the isolated harness. And a task_source that is not a local
    path must name an explicit registry version.

    Attributes:
        name: Task name written into task.toml.
        description: Human-readable task description.
        agent_repo: Absolute path to the editable target. Copied twice, into the
            immutable agent-baseline and the editable agent-seed.
        task_source: Local path to the task definitions, or a registry reference
            pinned as name@version.
        agent_import_path: Import path of the target agent class Harbor loads.
            Required for a harbor evaluation_backend, unused for a command one.
        evaluation_backend: How a candidate is scored. "harbor" drives a target
            agent with a nested `harbor run`; "command" runs a program instead,
            for a target that is not an agent. The outer optimizer is a Harbor
            agent either way.
        command_backend: The scoring program, required when evaluation_backend is
            "command" and rejected otherwise.
        harbor_requirement: Pinned harbor requirement for the task image.
        partitions: Partition name to the Harbor task names it holds. Emitted as
            one cases/<partition>.jsonl per partition.
        task_manifest: Optional path to an existing JSON task manifest.
        agent_access: One AgentAccessSpec per partition the optimizer may reach.
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
        model: Default model for the run.
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
        reward_key: Metric key the backend treats as the reward.
        aggregate_attempts: Combine repeat attempts by "best" or "mean".
        feedback_transcripts: Include target transcripts in the agent's feedback.
        feedback_max_bytes: Byte cap per feedback transcript.
        expose_attempt_detail: Report per-attempt detail, not just the aggregate.
        extra_harbor_args: Extra flags for the evaluation sub-run. Rejected if
            they override a flag the compiler controls.
        agent_env: Environment injected into the optimizer agent's own shell
            (setup, install, and run) as harbor --ae KEY=VALUE. Distinct from
            extra_harbor_args, which only reaches the evaluation sub-run, and
            task_environment, which is that sub-run's environment. Use it for
            things like UV_TOOL_BIN_DIR, so `uv tool install` targets a writable
            directory on a non-root sandbox.
        timeout_seconds: Wall clock for one evaluation.
        case_timeout_seconds: Wall clock for one case.
        task_agent_timeout_seconds: Wall clock declared for the target agent.
        max_concurrency: Cases evaluated concurrently.
        error_rate_threshold: Case error fraction above which an evaluation is
            abandoned.
        verifier_timeout_seconds: Wall clock for the trusted verifier. Falls back
            to timeout_seconds.
        evaluation_drain_timeout_seconds: Grace period for in-flight evaluations
            to finish. Falls back to timeout_seconds.
        secrets: Environment variable names routed into the task. Their presence
            on the build host is checked at compile time unless
            VERO_SKIP_SECRET_CHECK is set.
        inference_gateway: Gateway credential source and per-scope policies; see
            InferenceGatewaySpec. Omit for no metered model access.
        wandb: Trusted-side Weights & Biases reporting from the evaluation
            sidecar. Requires WANDB_API_KEY in secrets, routed to the sidecar and
            never to the agent.
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
        task_services_use_upstream: Give task-owned model services (user
            simulators, graders) running inside the task containers the real
            upstream via OPENAI_*, while the candidate keeps the metered,
            allow-listed gateway on VERO_AGENT_INFERENCE_*. Needed for
            benchmarks like tau3 whose environment makes its own model calls.
        harness_user: Unprivileged OS user that runs the untrusted candidate
            harness in the sidecar, isolating it from trusted state. None
            disables that isolation.
        task_environment: Extra environment for the evaluation sub-run, available
            to the task's ${VAR} substitutions. Must not name secrets.
        base_image_main: Base image for the main container.
        base_image_sidecar: Base image for the sidecar container.
    """

    name: str
    description: str = ""
    agent_repo: str
    task_source: str
    # Optional because a command evaluation_backend has no agent class to load;
    # validate_references requires it for a harbor backend.
    agent_import_path: str | None = None
    harbor_requirement: str
    partitions: dict[str, list[str]]
    task_manifest: str | None = None
    agent_access: list[AgentAccessSpec]
    selection_partition: str
    targets: list[VerificationTargetSpec]
    # TODO: the Harbor-inner fields below (model, retries, feedback, harbor args)
    # only apply when evaluation_backend is "harbor", and the command equivalents
    # live in command_backend. Nesting each group under its own discriminated
    # sub-spec would make that structural instead of validated, at the cost of
    # rewriting every benchmark's build.yaml. Deferred deliberately.
    evaluation_backend: Literal["harbor", "command"] = "harbor"
    command_backend: CommandBackendSpec | None = None

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
    agent_env: dict[str, str] = Field(default_factory=dict)

    timeout_seconds: float = Field(default=1800.0, gt=0)
    case_timeout_seconds: float = Field(default=180.0, gt=0)
    task_agent_timeout_seconds: float = Field(default=600.0, gt=0)
    max_concurrency: int = Field(default=8, ge=1)
    error_rate_threshold: float | None = Field(default=0.1, gt=0, le=1)
    verifier_timeout_seconds: int | None = Field(default=None, ge=1)
    evaluation_drain_timeout_seconds: float | None = Field(default=None, gt=0)

    secrets: list[str] = Field(default_factory=list)
    inference_gateway: InferenceGatewaySpec | None = None
    wandb: WandbSpec | None = None
    read_only_paths: list[str] = Field(default_factory=list)
    workspace_overlays: list[WorkspaceOverlaySpec] = Field(default_factory=list)
    include_evals_skill: bool = True
    instruct_multifidelity: bool = True
    instruct_exhaust_budget: bool = True
    disclose_budget: bool = True
    task_services_use_upstream: bool = False
    harness_user: str | None = "harness"
    task_environment: dict[str, str] = Field(default_factory=dict)
    base_image_main: str = "ghcr.io/astral-sh/uv:python3.12-bookworm"
    base_image_sidecar: str = "ghcr.io/astral-sh/uv:python3.12-bookworm"

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

    @field_validator(
        "name",
        "agent_repo",
        "task_source",
        "evaluation_set_name",
        "environment_name",
        "harbor_python_version",
        "default_index",
        "base_image_main",
        "base_image_sidecar",
    )
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Harbor build identity must not be empty")
        return value

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

    @field_validator("model", "reward_key", "agent_import_path")
    @classmethod
    def validate_optional_identity(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional Harbor identity must not be empty")
        return value

    @field_validator("task_manifest")
    @classmethod
    def validate_task_manifest_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip() or not Path(value).is_file():
            raise ValueError("task_manifest must be an existing JSON file")
        return value

    @field_validator("extra_harbor_args")
    @classmethod
    def validate_extra_harbor_args(cls, value: list[str]) -> list[str]:
        controlled = {
            "-a",
            "-d",
            "-e",
            "-i",
            "-m",
            "-n",
            "-p",
            "--agent-import-path",
            "--jobs-dir",
            "--max-retries",
            "--n-attempts",
        }
        conflicts = [
            argument for argument in value if argument.split("=", 1)[0] in controlled
        ]
        if conflicts:
            raise ValueError(
                "extra_harbor_args override controlled flags: " + ", ".join(conflicts)
            )
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

    @field_validator("secrets")
    @classmethod
    def validate_secrets(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("secret environment names must be unique")
        for name in value:
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                raise ValueError(f"invalid secret environment name: {name!r}")
        return value

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

    @model_validator(mode="after")
    def validate_references(self) -> HarborBuildConfig:
        if not Path(self.task_source).exists() and "@" not in self.task_source:
            raise ValueError("registry task_source must include an explicit version")
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
                raise ValueError(
                    "command_backend requires evaluation_backend: command"
                )
            if self.agent_import_path is None:
                raise ValueError(
                    "harbor evaluation_backend requires an agent_import_path"
                )
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
        if self.task_manifest is not None:
            try:
                manifest = json.loads(
                    Path(self.task_manifest).read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as error:
                raise ValueError("task_manifest must contain valid JSON") from error
            if not isinstance(manifest, dict):
                raise ValueError("task_manifest must be a JSON object")
            manifest_source = manifest.get("task_source")
            if manifest_source != self.task_source and not (
                # A vendored local source is recorded relative to the manifest,
                # while the loader resolves the build's copy to an absolute
                # path once the directory exists; compare resolved locations.
                isinstance(manifest_source, str)
                and (Path(self.task_manifest).parent / manifest_source).resolve()
                == Path(self.task_source).resolve()
            ):
                raise ValueError(
                    "task_manifest task_source does not match build task_source"
                )
            manifest_tasks = manifest.get("tasks")
            if not isinstance(manifest_tasks, list):
                raise ValueError("task_manifest tasks must be a JSON array")
            names: list[str] = []
            for item in manifest_tasks:
                name = item.get("name") if isinstance(item, dict) else item
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        "task_manifest tasks must be names or objects with a name"
                    )
                names.append(name)
            if len(names) != len(set(names)):
                raise ValueError("task_manifest contains duplicate task names")
            selected = {
                task
                for partition_tasks in self.partitions.values()
                for task in partition_tasks
            }
            unknown = sorted(selected - set(names))
            if unknown:
                raise ValueError(
                    "partitions reference tasks absent from task_manifest: "
                    + ", ".join(unknown)
                )
        if self.inference_gateway is not None:
            gateway_environment = {self.inference_gateway.upstream_api_key_env}
            if self.inference_gateway.upstream_base_url_env is not None:
                gateway_environment.add(self.inference_gateway.upstream_base_url_env)
            overlap = sorted(set(self.secrets) & gateway_environment)
            if overlap:
                raise ValueError(
                    "inference gateway credentials must not also be sidecar secrets: "
                    + ", ".join(overlap)
                )
        # Every model that will actually be requested has to be allowed by the
        # scope that will actually serve it. Otherwise the mismatch surfaces as a
        # gateway 403 at run time — and for a verification target, only after
        # search has already spent its budget. A command backend requests no
        # models, so there is nothing to reconcile.
        if self.inference_gateway is not None and self.evaluation_backend == "harbor":
            evaluation_models = self.inference_gateway.evaluation.allowed_models
            if self.model is not None and self.model not in evaluation_models:
                raise ValueError(
                    f"model {self.model!r} is not in the inference gateway's "
                    f"evaluation allowed_models ({', '.join(evaluation_models)})"
                )
            # Finalization runs the target agent too, so it must allow the model
            # each target scores with: its own override, else the task model.
            finalization_models = (
                self.inference_gateway.finalization or self.inference_gateway.evaluation
            ).allowed_models
            for target in self.targets:
                scoring_model = target.model or self.model
                if scoring_model is not None and scoring_model not in (
                    finalization_models
                ):
                    raise ValueError(
                        f"target {target.partition!r} scores with model "
                        f"{scoring_model!r}, which is not in the inference gateway's "
                        f"finalization allowed_models "
                        f"({', '.join(finalization_models)})"
                    )
        return self


_BUILD_PARAM = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::(-|\?)([^}]*))?\}")


def _substitute_build_param(text: str, context: dict[str, str]) -> str:
    """Resolve ``${NAME}`` / ``${NAME:-default}`` / ``${NAME:?message}`` in one scalar."""

    def replace(match: re.Match[str]) -> str:
        name, operator, argument = match.group(1), match.group(2), match.group(3)
        resolved = context.get(name)
        if resolved:
            return resolved
        if operator == "-":
            return argument or ""
        if operator == "?":
            raise ValueError(
                f"required build parameter {name!r} is unset: "
                f"{argument or 'no message provided'}"
            )
        raise ValueError(
            f"build parameter {name!r} is unset; pass --param {name}=VALUE "
            "or set the environment variable"
        )

    return _BUILD_PARAM.sub(replace, text)


def _resolve_build_params(value: object, context: dict[str, str]) -> object:
    """Recursively resolve ``${...}`` placeholders in string scalars of a YAML value."""
    if isinstance(value, str):
        return _substitute_build_param(value, context)
    if isinstance(value, dict):
        return {key: _resolve_build_params(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_build_params(item, context) for item in value]
    return value


def load_harbor_build_config(
    path: Path | str,
    *,
    params: dict[str, str] | None = None,
) -> HarborBuildConfig:
    """Load YAML and resolve local paths relative to the configuration file.

    ``${NAME}`` placeholders in the YAML are substituted at load time from
    ``params`` (explicit, e.g. ``--param NAME=VALUE``) layered over the process
    environment, so run-time knobs (optimizer model, inner sandbox provider,
    concurrency, ...) can be varied without rebuilding the task. Use
    ``${NAME:-default}`` for a fallback and ``${NAME:?message}`` to require a
    value. Fields left un-templated stay fixed (the reproducible measurement
    substrate).
    """
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "install scale-vero[harbor] to load Harbor builds"
        ) from error

    config_path = Path(path).expanduser().resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Harbor build config must be a YAML object")
    context = {**os.environ, **(params or {})}
    value = _resolve_build_params(value, context)
    base = config_path.parent
    partition_files = value.pop("partition_files", None)
    if partition_files is not None:
        if "partitions" in value:
            raise ValueError("use either partitions or partition_files, not both")
        if not isinstance(partition_files, dict) or not partition_files:
            raise ValueError("partition_files must be a non-empty YAML object")
        partitions: dict[str, list[str]] = {}
        for partition, filename in partition_files.items():
            if not isinstance(partition, str) or not isinstance(filename, str):
                raise ValueError("partition_files must map names to JSON files")
            partition_path = Path(filename).expanduser()
            if not partition_path.is_absolute():
                partition_path = base / partition_path
            try:
                tasks = json.loads(partition_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"partition file {partition_path} must contain valid JSON"
                ) from error
            if not isinstance(tasks, list) or any(
                not isinstance(task, str) for task in tasks
            ):
                raise ValueError(
                    f"partition file {partition_path} must be a JSON array of task names"
                )
            partitions[partition] = tasks
        value["partitions"] = partitions
    agent_repo = value.get("agent_repo")
    if isinstance(agent_repo, str) and not Path(agent_repo).is_absolute():
        value["agent_repo"] = str((base / agent_repo).resolve())
    task_source = value.get("task_source")
    if isinstance(task_source, str):
        local_source = base / task_source
        if not Path(task_source).is_absolute() and local_source.exists():
            value["task_source"] = str(local_source.resolve())
    task_manifest = value.get("task_manifest")
    if isinstance(task_manifest, str) and not Path(task_manifest).is_absolute():
        value["task_manifest"] = str((base / task_manifest).resolve())
    overlays = value.get("workspace_overlays")
    if isinstance(overlays, list):
        for entry in overlays:
            source = entry.get("source") if isinstance(entry, dict) else None
            if isinstance(source, str) and not Path(source).is_absolute():
                entry["source"] = str((base / source).resolve())
    return HarborBuildConfig.model_validate(value)
