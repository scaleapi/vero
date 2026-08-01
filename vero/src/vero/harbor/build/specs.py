"""Leaf models a build.yaml composes.

Each spec here is self-contained: it validates its own fields and knows nothing
about the build that holds it. HarborBuildConfig in config.py assembles them and
adds the rules that span more than one.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, field_validator

from vero.evaluation import DisclosureLevel, EvaluationAccessPolicy
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
        n_attempts: Optional per-target override of the global attempts-per-case;
            None inherits the global. Set >1 to score each held-out case several
            times and combine them (with aggregate_attempts), e.g. 3 to average a
            noisy final eval over 3 scorings. Only meaningful on a harbor target
            whose partition is not agent-evaluable (see the config validator).
        aggregate_attempts: Optional per-target override of how repeat attempts
            combine ("best"/"mean"); None inherits the global ("mean").
    """

    partition: str
    reward_key: str = "reward"
    model: str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    failure_value: float = 0.0
    baseline_reward: float | None = None
    max_attempts: int = Field(default=1, ge=1)
    n_attempts: int | None = Field(default=None, ge=1)
    aggregate_attempts: Literal["best", "mean"] | None = None

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
        model_aliases: Optional rewrite of the requested model, applied *after*
            the allow-list check on the way upstream. Use it when the model that
            must serve a request cannot be named by the caller -- codex, for
            instance, reduces a model id to its last path component, so it can
            never ask for a provider-qualified deployment. Declared here rather
            than inferred, so the substitution is auditable in the build config
            and in the request log.
        max_requests: Cap on proxied requests; unlimited when omitted.
        max_tokens: Cap on cumulative tokens; unlimited when omitted. Checked
            before a request starts, so a single request can overshoot it.
        max_concurrency: Requests this scope may have in flight at once.
    """

    allowed_models: list[str]
    model_aliases: dict[str, str] = Field(default_factory=dict)
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

    @field_validator("model_aliases")
    @classmethod
    def validate_aliases(cls, value: dict[str, str]) -> dict[str, str]:
        # Self-aliases are dropped rather than rejected, and keys outside
        # allowed_models are permitted. Both concessions exist because one build
        # config is shared by every cell of a grid, with allowed_models templated
        # per launch: a map that pins the right deployment for one optimizer
        # necessarily carries keys the other launches never request. Rejecting
        # those would make the field unusable exactly where it is needed.
        #
        # Nothing is lost by allowing them. An alias can only fire for a model
        # the allow-list already admitted, so an unreachable key is inert, and a
        # self-alias is a no-op. The failure this field could cause -- a request
        # served by a model other than the one named -- is bounded by the
        # allow-list and made visible by the `aliased_from` field in the request
        # log, not by validation here.
        cleaned = {}
        for requested, upstream in value.items():
            if not requested.strip() or not upstream.strip():
                raise ValueError("model_aliases names must not be empty")
            if requested != upstream:
                cleaned[requested] = upstream
        return cleaned


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
    handler in vero.gateway.inference for the full note.

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
    skills, config, data) into the optimizer's workspace at build time.

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
            CommandBackend runs the harness with PATH=os.defpath, so a harness
            that needs the build image's interpreter must list PATH here.
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
