"""Configuration schema for compiling VeRO optimization tasks for Harbor."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator

from vero.evaluation import (
    DisclosureLevel,
    EvaluationModel,
    MetricSelector,
    ObjectiveSpec,
)


class AgentAccessSpec(EvaluationModel):
    partition: str
    disclosure: DisclosureLevel = DisclosureLevel.AGGREGATE
    expose_case_resources: bool = False
    min_aggregate_cases: int = Field(default=1, ge=1)
    total_runs: int | None = Field(default=None, ge=0)
    total_cases: int | None = Field(default=None, ge=0)

    @field_validator("partition")
    @classmethod
    def validate_partition(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("agent access partition must not be empty")
        return value


class VerificationTargetSpec(EvaluationModel):
    partition: str
    reward_key: str = "reward"
    model: str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    failure_value: float = 0.0
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


class InferenceBudgetSpec(EvaluationModel):
    """Routing policy and optional limits for one inference-gateway scope."""

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


class InferenceGatewaySpec(EvaluationModel):
    """Credential source and independent producer/evaluator policies."""

    upstream_api_key_env: str = "OPENAI_API_KEY"
    upstream_base_url_env: str | None = "OPENAI_BASE_URL"
    default_upstream_base_url: str = "https://api.openai.com/v1"
    producer: InferenceBudgetSpec
    evaluation: InferenceBudgetSpec

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


class HarborBuildConfig(EvaluationModel):
    """Everything needed to emit an isolated Harbor optimization task."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    agent_repo: str
    task_source: str
    agent_import_path: str
    harbor_requirement: str
    partitions: dict[str, list[str]]
    task_manifest: str | None = None
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
    baseline_floor: bool = True
    score_baseline: bool = True
    rescore_top_k: int = Field(default=3, ge=1)
    rescore_attempts: int = Field(default=1, ge=1)

    model: str | None = None
    environment_name: str = "modal"
    harbor_python_version: str = "3.12"
    default_index: str = "https://pypi.org/simple"
    n_attempts: int = Field(default=1, ge=1)
    max_retries: int = Field(default=2, ge=0)
    infrastructure_max_attempts: int = Field(default=3, ge=1)
    infrastructure_retry_delay_seconds: float = Field(default=5.0, ge=0)
    reward_key: str | None = None
    aggregate_attempts: Literal["best", "mean"] = "best"
    feedback_transcripts: bool = False
    feedback_max_bytes: int = Field(default=3000, ge=0)
    expose_attempt_detail: bool = False
    extra_harbor_args: list[str] = Field(default_factory=list)

    timeout_seconds: float = Field(default=1800.0, gt=0)
    max_concurrency: int = Field(default=8, ge=1)
    error_rate_threshold: float | None = Field(default=0.1, gt=0, le=1)
    verifier_timeout_seconds: int | None = Field(default=None, ge=1)
    evaluation_drain_timeout_seconds: float | None = Field(default=None, gt=0)

    secrets: list[str] = Field(default_factory=list)
    inference_gateway: InferenceGatewaySpec | None = None
    read_only_paths: list[str] = Field(default_factory=list)
    instruct_multifidelity: bool = True
    instruct_exhaust_budget: bool = True
    base_image_main: str = "ghcr.io/astral-sh/uv:python3.12-bookworm"
    base_image_sidecar: str = "ghcr.io/astral-sh/uv:python3.12-bookworm"

    @field_validator(
        "name",
        "agent_repo",
        "task_source",
        "agent_import_path",
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

    @field_validator("model", "reward_key")
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
            if manifest_source != self.task_source:
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
        return self


def load_harbor_build_config(path: Path | str) -> HarborBuildConfig:
    """Load YAML and resolve local paths relative to the configuration file."""
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
    return HarborBuildConfig.model_validate(value)
