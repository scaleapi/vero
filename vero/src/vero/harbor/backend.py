"""Evaluate a candidate program by running it over Harbor tasks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import mimetypes
import os
import re
import shutil
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from vero.evaluation.backend import EvaluationContext
from vero.evaluation.error_taxonomy import (
    NO_REWARD_SIGNAL,
    ErrorCategory,
    classify_case,
    policy,
)
from vero.evaluation.models import (
    AllCases,
    BackendProvenance,
    CaseError,
    CaseIds,
    CaseRange,
    CaseResult,
    CaseStatus,
    DiagnosticSeverity,
    EvaluationArtifact,
    EvaluationCost,
    EvaluationDiagnostic,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
)
from vero.evaluation.security import sanitize_evaluation_report, sanitize_text
from vero.harbor.isolation import harness_grant_commands, harness_reachability_probe
from vero.models import StrictModel
from vero.sandbox import CommandResult, Sandbox
from vero.staging import SandboxStagingArea

logger = logging.getLogger(__name__)


def _default_uv() -> str:
    executable = shutil.which("uv")
    if executable is None:
        raise ValueError("uv is required to configure a Harbor backend")
    return str(Path(executable).resolve())


class HarborCase(StrictModel):
    """One canonical case mapped to one Harbor task name."""

    id: str
    task_name: str
    result_task_name: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id", "task_name")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Harbor case identity must not be empty")
        return value

    @field_validator("result_task_name")
    @classmethod
    def validate_result_identity(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Harbor result task identity must not be empty")
        return value

    @property
    def expected_result_task_name(self) -> str:
        return self.result_task_name or self.task_name


# Dedicated env vars carrying the candidate agent's gateway credentials when
# task_services_use_upstream reroutes OPENAI_* to the real upstream. A custom
# agent reads these to keep its own inference on the metered/allow-listed gateway.
# This name is a contract shared with agents (they cannot import this module).
AGENT_INFERENCE_API_KEY_ENV = "VERO_AGENT_INFERENCE_API_KEY"
AGENT_INFERENCE_BASE_URL_ENV = "VERO_AGENT_INFERENCE_BASE_URL"


class HarborBackendConfig(StrictModel):
    """Trusted configuration for nested ``harbor run`` evaluation."""

    task_source: str
    agent_import_path: str
    cases_path: str
    harbor_requirement: str
    evaluation_set_name: str = "harbor"
    partition: str | None = None
    model: str | None = None
    environment_name: str = "modal"
    python_version: str = "3.12"
    case_timeout_seconds: float = Field(default=180.0, gt=0)
    task_agent_timeout_seconds: float = Field(default=600.0, gt=0)
    n_attempts: int = Field(default=1, ge=1)
    max_retries: int = Field(default=2, ge=0)
    retry_wait_multiplier: float = Field(default=2.0, ge=1.0)
    retry_min_wait_seconds: float = Field(default=4.0, ge=0.0)
    retry_max_wait_seconds: float = Field(default=60.0, ge=0.0)
    # Whole-sub-run infrastructure retry. Default 1 (no retry): retry > 1 only
    # applies to trusted finalization evaluations (see HarborBackend.evaluate),
    # never to competitive agent search, where re-running for best coverage is
    # a best-of-N amplifier a candidate can trigger.
    infrastructure_max_attempts: int = Field(default=1, ge=1)
    infrastructure_retry_delay_seconds: float = Field(default=5.0, ge=0)
    reward_key: str | None = None
    # Average attempts by default rather than taking the best: best-of-n is an
    # optimistic estimator that biases selection above the single-shot final
    # score. Selection and final both run through this same aggregation.
    aggregate_attempts: Literal["best", "mean"] = "mean"
    failure_score: float = 0.0
    feedback_transcripts: bool = False
    feedback_max_bytes: int = Field(default=3000, ge=0)
    expose_attempt_detail: bool = False
    uv_executable: str = Field(default_factory=_default_uv)
    default_index: str = "https://pypi.org/simple"
    environment: dict[str, str] = Field(default_factory=dict)
    passthrough_environment: list[str] = Field(default_factory=list)
    inference_gateway_url: str | None = None
    inference_gateway_token: str | None = None
    # Optional reserved scope for trusted finalization (admin re-score / targets),
    # so the optimizer's search evaluations cannot exhaust the budget the mandatory
    # re-score needs. When unset, finalization falls back to the evaluation scope.
    inference_gateway_finalization_token: str | None = None
    # When True, task-owned evaluation services (e.g. LLM user-simulators or graders
    # that run *inside* the task containers and cannot reach the compose-internal
    # gateway) receive the real upstream credentials via OPENAI_*, while the
    # candidate agent still routes through the metered/allow-listed gateway via
    # VERO_AGENT_INFERENCE_*. upstream_*_env name the host env vars holding the
    # upstream credentials (read from the sidecar process environment).
    task_services_use_upstream: bool = False
    upstream_api_key_env: str | None = None
    upstream_base_url_env: str | None = None
    # Unprivileged OS user to execute the untrusted candidate harness as, so it
    # cannot read the trusted state (held-out records, budgets, other
    # candidates, admin token) that the sidecar process owns. None (the default,
    # used for local dev and tests) runs the harness in-process with no drop;
    # the compiled sidecar sets this to isolate.
    harness_user: str | None = None
    case_resources_cache_path: str | None = None
    # The gateway's durable usage ledger; when set, each evaluation's report
    # carries its own inference token totals as metrics (attribution keyed by
    # evaluation id).
    inference_usage_path: str | None = None
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("cases_path", "case_resources_cache_path")
    @classmethod
    def validate_absolute_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip() or not Path(value).is_absolute():
            raise ValueError("Harbor backend file paths must be absolute")
        return value

    @field_validator(
        "task_source",
        "agent_import_path",
        "harbor_requirement",
        "evaluation_set_name",
        "environment_name",
        "python_version",
        "default_index",
    )
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Harbor backend identity must not be empty")
        return value

    @field_validator("uv_executable")
    @classmethod
    def validate_uv_executable(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("uv_executable must not be empty")
        return value

    @field_validator("harbor_requirement")
    @classmethod
    def validate_pinned_harbor_requirement(cls, value: str) -> str:
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

    @field_validator(
        "partition",
        "model",
        "reward_key",
        "inference_gateway_url",
        "inference_gateway_token",
    )
    @classmethod
    def validate_optional_identity(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional Harbor identity must not be empty")
        return value

    @field_validator("failure_score")
    @classmethod
    def validate_failure_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("failure_score must be finite")
        return value

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: dict[str, str]) -> dict[str, str]:
        for name in value:
            if not name or "=" in name:
                raise ValueError(f"invalid environment variable name: {name!r}")
        return value

    @field_validator("passthrough_environment")
    @classmethod
    def validate_passthrough_environment(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("passthrough environment names must be unique")
        for name in value:
            if not name or "=" in name:
                raise ValueError(f"invalid environment variable name: {name!r}")
        return value

    @field_validator("extra_args")
    @classmethod
    def validate_extra_args(cls, value: list[str]) -> list[str]:
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
            "--agent-timeout-multiplier",
        }
        conflicts = [
            argument for argument in value if argument.split("=", 1)[0] in controlled
        ]
        if conflicts:
            raise ValueError(
                "extra_args cannot override backend-controlled Harbor flags: "
                + ", ".join(conflicts)
            )
        return value

        return value

    @model_validator(mode="after")
    def validate_filesystem_and_environment(self) -> HarborBackendConfig:
        if not Path(self.cases_path).is_file():
            raise ValueError("Harbor cases_path must be an existing file")
        overlap = set(self.environment) & set(self.passthrough_environment)
        if overlap:
            raise ValueError(
                "environment and passthrough_environment overlap for: "
                + ", ".join(sorted(overlap))
            )
        if (self.inference_gateway_url is None) != (
            self.inference_gateway_token is None
        ):
            raise ValueError(
                "inference_gateway_url and inference_gateway_token must be set together"
            )
        if (
            self.inference_gateway_url is not None
            and not self.inference_gateway_url.startswith(("http://", "https://"))
        ):
            raise ValueError("inference_gateway_url must be HTTP(S)")
        gateway_names = {"OPENAI_API_KEY", "OPENAI_BASE_URL"}
        if self.task_services_use_upstream:
            gateway_names |= {AGENT_INFERENCE_API_KEY_ENV, AGENT_INFERENCE_BASE_URL_ENV}
        gateway_overlap = gateway_names & (
            set(self.environment) | set(self.passthrough_environment)
        )
        if self.inference_gateway_url is not None and gateway_overlap:
            raise ValueError(
                "gateway-managed environment variables must not also be configured: "
                + ", ".join(sorted(gateway_overlap))
            )
        if self.task_services_use_upstream:
            if self.inference_gateway_url is None:
                raise ValueError(
                    "task_services_use_upstream requires an inference gateway"
                )
            if not self.upstream_api_key_env:
                raise ValueError(
                    "task_services_use_upstream requires upstream_api_key_env"
                )
        if self.harness_user is not None and self.task_services_use_upstream:
            # uid isolation does not hide env vars, so the raw upstream key would
            # still reach the isolated harness through OPENAI_*. Refuse the
            # combination until task-service credentials are delivered to the
            # task containers off the harness environment.
            raise ValueError(
                "harness_user cannot be combined with task_services_use_upstream: "
                "the raw upstream credential would still reach the isolated "
                "harness through its environment"
            )
        return self


class HarborBackend:
    """Run Harbor as an external evaluator and collate its trial records."""

    name = "harbor"
    version = "2"

    def __init__(self, config: HarborBackendConfig):
        self.config = config
        self._cases = self._load_cases()
        case_ids = [case.id for case in self._cases]
        task_names = [case.task_name for case in self._cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Harbor case IDs must be unique")
        if len(task_names) != len(set(task_names)):
            raise ValueError(
                "Harbor task names must be unique within an evaluation set"
            )

    @property
    def provenance(self) -> BackendProvenance:
        return BackendProvenance.from_config(
            name=self.name,
            version=self.version,
            config=self.config,
        )

    def _load_cases(self) -> list[HarborCase]:
        path = Path(self.config.cases_path)
        if path.suffix == ".jsonl":
            raw = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw = raw.get("cases")
        if not isinstance(raw, list) or not raw:
            raise ValueError("Harbor case file must contain a non-empty case list")
        return [HarborCase.model_validate(value) for value in raw]

    def _validate_evaluation_set(self, evaluation_set: EvaluationSet) -> None:
        if evaluation_set.name != self.config.evaluation_set_name:
            raise ValueError(
                f"Harbor backend owns evaluation set "
                f"{self.config.evaluation_set_name!r}, not {evaluation_set.name!r}"
            )
        if evaluation_set.partition != self.config.partition:
            raise ValueError(
                f"Harbor backend owns partition {self.config.partition!r}, "
                f"not {evaluation_set.partition!r}"
            )
        selection = evaluation_set.selection
        if isinstance(selection, CaseRange) and selection.stop > len(self._cases):
            raise ValueError(
                f"case range stops at {selection.stop}, but the Harbor evaluation "
                f"set contains {len(self._cases)} cases"
            )
        if isinstance(selection, CaseIds):
            known = {case.id for case in self._cases}
            unknown = sorted(set(selection.ids) - known)
            if unknown:
                raise ValueError(f"unknown Harbor case IDs: {unknown}")

    def _selected_cases(self, evaluation_set: EvaluationSet) -> list[HarborCase]:
        self._validate_evaluation_set(evaluation_set)
        selection = evaluation_set.selection
        if isinstance(selection, AllCases):
            return list(self._cases)
        if isinstance(selection, CaseRange):
            return self._cases[selection.start : selection.stop]
        if isinstance(selection, CaseIds):
            by_id = {case.id: case for case in self._cases}
            return [by_id[case_id] for case_id in selection.ids]
        raise AssertionError(f"unsupported Harbor case selection: {selection}")

    async def resolve_cost(self, evaluation_set: EvaluationSet) -> EvaluationCost:
        return EvaluationCost(cases=len(self._selected_cases(evaluation_set)))

    async def export_case_resources(
        self,
        *,
        evaluation_set: EvaluationSet,
        destination: str,
        sandbox: Sandbox,
    ) -> None:
        """Expose complete task resources for an explicitly authorized partition."""

        cases = self._selected_cases(evaluation_set)
        configured_cache = self.config.case_resources_cache_path
        if configured_cache is None:
            with tempfile.TemporaryDirectory(prefix="vero-harbor-cases-") as temporary:
                root = Path(temporary)
                await self._materialize_case_resources(root, cases, evaluation_set)
                await sandbox.upload(str(root), destination)
            return

        cache = Path(configured_cache)
        if not (cache / "index.json").is_file():
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(
                    dir=cache.parent,
                    prefix=f".{cache.name}.",
                )
            )
            try:
                await self._materialize_case_resources(
                    temporary,
                    cases,
                    evaluation_set,
                )
                if cache.exists():
                    shutil.rmtree(cache)
                os.replace(temporary, cache)
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        self._make_agent_readable(cache)
        await sandbox.upload(str(cache), destination)

    async def _materialize_case_resources(
        self,
        root: Path,
        cases: list[HarborCase],
        evaluation_set: EvaluationSet,
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        source = Path(self.config.task_source).expanduser()
        if source.exists():
            case_paths = self._materialize_local_case_resources(root, source, cases)
            dataset_files_path = None
        else:
            case_paths, dataset_files_path = await self._materialize_package_resources(
                root,
                cases,
            )
        (root / "index.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "evaluation_set": evaluation_set.model_dump(mode="json"),
                    "task_source": self.config.task_source,
                    "task_source_exposed": True,
                    "dataset_files_path": dataset_files_path,
                    "cases": [
                        {
                            "case_id": case.id,
                            "task_name": case.task_name,
                            "path": case_paths[case.id],
                        }
                        for case in cases
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._make_agent_readable(root)

    @staticmethod
    def _make_agent_readable(root: Path) -> None:
        """Allow the non-root producer to traverse its read-only context mount."""

        for path in (root, *root.rglob("*")):
            if path.is_symlink():
                continue
            mode = path.stat().st_mode & 0o777
            if path.is_dir():
                path.chmod((mode | 0o055) & ~0o022)
            elif path.is_file():
                path.chmod((mode | 0o044) & ~0o022)

    def _materialize_local_case_resources(
        self,
        root: Path,
        source: Path,
        cases: list[HarborCase],
    ) -> dict[str, str]:
        tasks = root / "tasks"
        tasks.mkdir()
        paths: dict[str, str] = {}
        resolved_source = source.resolve()
        for case in cases:
            task = (resolved_source / case.task_name).resolve()
            if task.parent != resolved_source or not task.is_dir():
                raise ValueError(
                    f"local Harbor case {case.task_name!r} is not a direct task directory"
                )
            destination = tasks / hashlib.sha256(case.id.encode()).hexdigest()
            shutil.copytree(task, destination)
            paths[case.id] = destination.relative_to(root).as_posix()
        dataset_files = root / "dataset-files"
        dataset_files.mkdir()
        for path in resolved_source.iterdir():
            if path.is_file() and not path.is_symlink():
                shutil.copy2(path, dataset_files / path.name)
        if not any(dataset_files.iterdir()):
            dataset_files.rmdir()
        return paths

    async def _materialize_package_resources(
        self,
        root: Path,
        cases: list[HarborCase],
    ) -> tuple[dict[str, str], str | None]:
        try:
            from harbor.registry.client.package import PackageDatasetClient
            from harbor.tasks.client import TaskClient
        except ImportError as error:
            raise RuntimeError(
                "the pinned Harbor package must be installed to expose remote case resources"
            ) from error

        client = PackageDatasetClient()
        metadata = await client.get_dataset_metadata(self.config.task_source)
        by_name = {task_id.get_name(): task_id for task_id in metadata.task_ids}
        missing = sorted(
            case.task_name for case in cases if case.task_name not in by_name
        )
        if missing:
            raise ValueError(
                "authorized cases are absent from the pinned Harbor dataset: "
                + ", ".join(missing)
            )
        task_ids = [by_name[case.task_name] for case in cases]
        tasks_root = root / "tasks"
        result = await TaskClient().download_tasks(
            task_ids=task_ids,
            output_dir=tasks_root,
            export=True,
        )
        paths = {
            case.id: download.path.relative_to(root).as_posix()
            for case, download in zip(cases, result.results, strict=True)
        }
        dataset_root = root / "dataset-files"
        files = await client.download_dataset_files(
            metadata,
            output_dir=dataset_root,
        )
        return paths, "dataset-files" if files else None

    def _secrets(self) -> list[str]:
        values = list(self.config.environment.values())
        values.extend(
            os.environ[name]
            for name in self.config.passthrough_environment
            if name in os.environ
        )
        if self.config.inference_gateway_token is not None:
            values.append(self.config.inference_gateway_token)
        if self.config.inference_gateway_finalization_token is not None:
            values.append(self.config.inference_gateway_finalization_token)
        return values

    def sanitize_error(self, message: str) -> str:
        return sanitize_text(message, self._secrets())

    def validate_request(self, request: EvaluationRequest) -> None:
        self._validate_evaluation_set(request.evaluation_set)
        if request.limits.retry.max_attempts > 1:
            raise ValueError(
                "Harbor does not support generic per-case retries; configure "
                "HarborBackendConfig.max_retries instead"
            )
        if request.limits.case_timeout_seconds != self.config.case_timeout_seconds:
            raise ValueError(
                "Harbor case timeout is fixed by the backend at "
                f"{self.config.case_timeout_seconds:g} seconds"
            )
        if request.seed is not None:
            raise ValueError("Harbor does not support the generic evaluation seed")
        payload = request.model_dump_json()
        if any(secret in payload for secret in self._secrets() if len(secret) >= 4):
            raise ValueError(
                "evaluation parameters must not contain configured secret values; "
                "pass secrets through the backend environment"
            )

    def _environment(
        self, evaluation_id: str, *, finalization: bool = False
    ) -> dict[str, str]:
        environment = {"PATH": os.defpath, "LANG": "C.UTF-8"}
        for name in ("TMPDIR", "TMP", "TEMP", "SYSTEMROOT"):
            if name in os.environ:
                environment[name] = os.environ[name]
        environment.update(self.config.environment)
        for name in self.config.passthrough_environment:
            if name in os.environ:
                environment[name] = os.environ[name]
        if self.config.harness_user is not None:
            # The isolated harness runs as an unprivileged user with no access
            # to root's home; point uv and git at a home it owns.
            home = f"/home/{self.config.harness_user}"
            environment["HOME"] = home
            environment["UV_CACHE_DIR"] = f"{home}/.cache/uv"
        if self.config.inference_gateway_url is not None:
            # Route trusted finalization evals to the reserved scope when configured;
            # everything else (and the fallback) uses the shared evaluation scope.
            use_finalization = (
                finalization
                and self.config.inference_gateway_finalization_token is not None
            )
            if finalization and not use_finalization:
                # The trusted final re-score asked for the reserved pool but none
                # was provisioned, so it silently shares the search budget and can
                # be starved. Surface it loudly rather than degrading quietly.
                logger.warning(
                    "finalization evaluation requested the reserved inference "
                    "scope but no finalization token is configured; falling back "
                    "to the shared 'evaluation' scope, which search evaluations "
                    "can exhaust"
                )
            scope = "finalization" if use_finalization else "evaluation"
            gateway_token = (
                self.config.inference_gateway_finalization_token
                if use_finalization
                else self.config.inference_gateway_token
            ) or ""
            gateway_url = (
                f"{self.config.inference_gateway_url.rstrip('/')}/scopes/{scope}/"
                f"{evaluation_id}/v1"
            )
            if self.config.task_services_use_upstream:
                # Task-owned eval services (user-sims, LLM graders) run inside the
                # task containers and can't reach the compose-internal gateway; hand
                # them the real upstream via OPENAI_*. The candidate agent keeps its
                # metered/allow-listed gateway on dedicated VERO_AGENT_INFERENCE_*.
                environment["OPENAI_API_KEY"] = os.environ.get(
                    self.config.upstream_api_key_env or "", ""
                )
                if self.config.upstream_base_url_env:
                    environment["OPENAI_BASE_URL"] = os.environ.get(
                        self.config.upstream_base_url_env, ""
                    )
                environment[AGENT_INFERENCE_API_KEY_ENV] = gateway_token
                environment[AGENT_INFERENCE_BASE_URL_ENV] = gateway_url
            else:
                environment["OPENAI_API_KEY"] = gateway_token
                environment["OPENAI_BASE_URL"] = gateway_url
            # Task packages that template `${OPENAI_API_BASE}` (litellm-style
            # name, e.g. swe-atlas-qna's rubric judge) get the same endpoint.
            if "OPENAI_BASE_URL" in environment:
                environment["OPENAI_API_BASE"] = environment["OPENAI_BASE_URL"]
        return environment

    def _source_args(self, task_source: str, *, local: bool) -> list[str]:
        return ["-p", task_source] if local else ["-d", task_source]

    def _retry_config_json(self) -> str:
        """Partial harbor JobConfig carrying only the retry policy.

        Harbor's ``run`` CLI exposes ``--max-retries``/``--retry-include``/
        ``--retry-exclude`` but no backoff flags, so the backoff is delivered via a
        ``--config`` snippet. The CLI loads ``--config`` as the base JobConfig and
        then applies our flags on top (``--max-retries`` wins for the count), so a
        retry-only partial is sufficient and safe.
        """
        return json.dumps(
            {
                "retry": {
                    "max_retries": self.config.max_retries,
                    "wait_multiplier": self.config.retry_wait_multiplier,
                    "min_wait_sec": self.config.retry_min_wait_seconds,
                    "max_wait_sec": self.config.retry_max_wait_seconds,
                }
            }
        )

    def _command(
        self,
        *,
        workspace: str,
        request: EvaluationRequest,
        cases: list[HarborCase],
        jobs_dir: str,
        task_source: str,
        local_task_source: bool,
        retry_config_path: str,
    ) -> list[str]:
        command = [
            self.config.uv_executable,
            "run",
            "--python",
            self.config.python_version,
            "--no-config",
            "--no-env-file",
            "--default-index",
            self.config.default_index,
            "--index-strategy",
            "first-index",
            "--project",
            workspace,
            "--with",
            self.config.harbor_requirement,
            "harbor",
            "run",
            # Non-interactive: a task that declares env vars (e.g. tau3's user-sim
            # TAU2_* and grader TAU2_NL_ASSERTIONS_MODEL) makes `harbor run` prompt
            # "Proceed? (Y/n)"; with no stdin the sub-run aborts (0 trials). GAIA
            # declares none, so it never hit this.
            "--yes",
            *self._source_args(task_source, local=local_task_source),
            "--agent-import-path",
            self.config.agent_import_path,
            "-e",
            self.config.environment_name,
            "-n",
            str(request.limits.max_concurrency),
            "--n-attempts",
            str(self.config.n_attempts),
            "--config",
            retry_config_path,
            "--max-retries",
            str(self.config.max_retries),
            "--agent-timeout-multiplier",
            str(
                self.config.case_timeout_seconds
                / self.config.task_agent_timeout_seconds
            ),
        ]
        model = request.parameters.get("harbor_model_override", self.config.model)
        if model is not None:
            command.extend(["-m", str(model)])
        for case in cases:
            command.extend(["-i", case.task_name])
        command.extend(["--jobs-dir", jobs_dir, *self.config.extra_args])
        return command

    def _trial_groups(self, jobs_dir: Path) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if not jobs_dir.exists():
            return groups
        jobs_root = jobs_dir.resolve()
        for result_path in jobs_dir.rglob("result.json"):
            if result_path.is_symlink():
                continue
            try:
                resolved = result_path.resolve()
                resolved.relative_to(jobs_root)
                value = json.loads(resolved.read_text(encoding="utf-8"))
                modified = resolved.stat().st_mtime
            except (json.JSONDecodeError, OSError, ValueError):
                continue
            task_name = value.get("task_name") if isinstance(value, dict) else None
            if not task_name:
                continue
            value["_trial_dir"] = str(resolved.parent)
            value["_mtime"] = modified
            groups[str(task_name)].append(value)
        for attempts in groups.values():
            attempts.sort(
                key=lambda value: (
                    value.get("finished_at") is None,
                    value.get("finished_at") or "",
                    value.get("trial_name") or "",
                    value.get("_mtime", 0.0),
                )
            )
        return groups

    def _extract_reward(self, rewards: dict[str, Any]) -> float | None:
        value: Any
        if self.config.reward_key is not None:
            value = rewards.get(self.config.reward_key)
        else:
            value = None
            for key in ("pass", "reward"):
                if key in rewards:
                    value = rewards[key]
                    break
            if value is None and len(rewards) == 1:
                value = next(iter(rewards.values()))
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _attempt_reward(self, attempt: dict[str, Any]) -> float | None:
        rewards = (attempt.get("verifier_result") or {}).get("rewards") or {}
        return self._extract_reward(rewards) if rewards else None

    def _attempt_is_infra(self, attempt: dict[str, Any], *, trusted: bool) -> bool:
        """Whether a dead attempt died to infrastructure (vs the candidate).

        For a competitive (agent) evaluation a candidate-controlled trial's own
        transient-infra exception is not trusted as infrastructure — the
        candidate could emit a timeout/connection error to have its dead attempt
        excluded from the mean. Only trusted evaluations honor that signal.
        """
        info = attempt.get("exception_info") or {}
        if not info:
            return False
        signals = [str(info.get("exception_type") or NO_REWARD_SIGNAL)]
        for detail_key in ("message", "detail", "exception_message"):
            detail = info.get(detail_key)
            if isinstance(detail, str) and detail:
                signals.append(detail)
        category = classify_case(signals)
        if not trusted and category == ErrorCategory.TRANSIENT_INFRA:
            return False
        return not policy(category).is_informative_sample

    def _best_attempt(
        self,
        attempts: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, float | None]:
        scored = [
            (attempt, self._attempt_reward(attempt))
            for attempt in attempts
            if self._attempt_reward(attempt) is not None
        ]
        if not scored:
            return None, None
        return max(
            scored,
            key=lambda item: (
                not bool(item[0].get("exception_info")),
                item[1],
                item[0].get("finished_at") or "",
                item[0].get("_mtime", 0.0),
            ),
        )

    def _transcript_feedback(
        self,
        attempts: list[dict[str, Any]],
    ) -> str | None:
        if not self.config.feedback_transcripts or self.config.feedback_max_bytes == 0:
            return None
        for attempt in attempts:
            trial_dir_value = attempt.get("_trial_dir")
            if not trial_dir_value:
                continue
            trial_dir = Path(trial_dir_value).resolve()
            for relative in ("agent/terminus_2.pane", "agent/trajectory.json"):
                path = trial_dir / relative
                if path.is_symlink():
                    continue
                try:
                    resolved = path.resolve()
                    resolved.relative_to(trial_dir)
                    data = resolved.read_bytes()
                except (OSError, ValueError):
                    continue
                if data:
                    return data[-self.config.feedback_max_bytes :].decode(
                        "utf-8", errors="replace"
                    )
        return None

    def _trial_artifacts(
        self,
        attempts: list[dict[str, Any]],
        artifact_root: Path,
    ) -> list[EvaluationArtifact]:
        """Reference complete Harbor trial records and redact configured credentials."""

        resolved_root = artifact_root.resolve()
        artifacts: list[EvaluationArtifact] = []
        seen: set[str] = set()
        for attempt in attempts:
            trial_dir_value = attempt.get("_trial_dir")
            if not trial_dir_value:
                continue
            trial_root = Path(trial_dir_value)
            if not trial_root.is_dir() or trial_root.is_symlink():
                continue
            for path in sorted(trial_root.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                try:
                    resolved = path.resolve(strict=True)
                    resolved.relative_to(resolved_root)
                except (OSError, ValueError):
                    continue
                relative = resolved.relative_to(resolved_root).as_posix()
                if relative in seen:
                    continue
                seen.add(relative)
                try:
                    payload = resolved.read_bytes()
                    if b"\x00" not in payload:
                        text = payload.decode("utf-8")
                        sanitized = sanitize_text(text, self._secrets())
                        if sanitized != text:
                            resolved.write_text(sanitized, encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    pass
                media_type = mimetypes.guess_type(resolved.name)[0]
                artifacts.append(
                    EvaluationArtifact(
                        path=relative,
                        media_type=media_type,
                        description=(
                            "Harbor trial record: "
                            + path.relative_to(trial_root).as_posix()
                        ),
                    )
                )
        return artifacts

    def _inference_usage_metrics(self, evaluation_id: str) -> dict[str, float]:
        """This evaluation's gateway token totals, from the durable ledger.

        Attribution is keyed by evaluation id (search and finalization scopes
        alike). Best-effort: telemetry must never fail an evaluation.
        """
        if self.config.inference_usage_path is None:
            return {}
        try:
            ledger = json.loads(
                Path(self.config.inference_usage_path).read_text(encoding="utf-8")
            )
            scopes = ledger.get("scopes")
            if not isinstance(scopes, dict):
                return {}
            totals = {
                "requests": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
            found = False
            for scope in scopes.values():
                attribution = (scope.get("attributions") or {}).get(evaluation_id)
                if not isinstance(attribution, dict):
                    continue
                found = True
                for key in totals:
                    value = attribution.get(key)
                    if isinstance(value, (int, float)):
                        totals[key] += int(value)
            if not found:
                return {}
            return {f"inference_{key}": float(value) for key, value in totals.items()}
        except (OSError, ValueError):
            logger.warning("inference usage metrics unavailable", exc_info=True)
            return {}

    @staticmethod
    def _attempt_wall_seconds(attempt: dict[str, Any]) -> float | None:
        started = attempt.get("started_at")
        finished = attempt.get("finished_at")
        if not isinstance(started, str) or not isinstance(finished, str):
            return None
        try:
            delta = datetime.fromisoformat(
                finished.replace("Z", "+00:00")
            ) - datetime.fromisoformat(started.replace("Z", "+00:00"))
        except ValueError:
            return None
        seconds = delta.total_seconds()
        return seconds if seconds >= 0 else None

    def _case_wall_seconds(self, attempts: list[dict[str, Any]]) -> float | None:
        durations = [
            seconds
            for attempt in attempts
            if (seconds := self._attempt_wall_seconds(attempt)) is not None
        ]
        return max(durations) if durations else None

    @staticmethod
    def _agent_reported_tokens(attempts: list[dict[str, Any]]) -> dict[str, float]:
        """Sum the agent-self-reported token counts across a case's attempts.

        Harbor agents record their own usage in ``agent_result`` (stock
        adapters and the baseline agents alike). Self-declared, so telemetry
        grade — the gateway's per-evaluation ``inference_*`` metrics remain
        the trusted envelope.
        """
        names = {
            "n_input_tokens": "agent_reported_input_tokens",
            "n_cache_tokens": "agent_reported_cached_input_tokens",
            "n_output_tokens": "agent_reported_output_tokens",
        }
        totals: dict[str, float] = {}
        for attempt in attempts:
            result = attempt.get("agent_result")
            if not isinstance(result, dict):
                continue
            for source, metric in names.items():
                value = result.get(source)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    totals[metric] = totals.get(metric, 0.0) + float(value)
        return totals

    def _case_result(
        self,
        case: HarborCase,
        attempts: list[dict[str, Any]],
        *,
        artifact_root: Path,
        trusted: bool = False,
    ) -> tuple[CaseResult, float]:
        trial_artifacts = self._trial_artifacts(attempts, artifact_root)
        wall_seconds = self._case_wall_seconds(attempts)
        reported_tokens = self._agent_reported_tokens(attempts)
        attempt_detail = [
            {
                "reward": self._attempt_reward(attempt),
                "exception": (attempt.get("exception_info") or {}).get(
                    "exception_type"
                ),
            }
            for attempt in attempts
        ]
        output: dict[str, JsonValue] = {
            "task_name": case.task_name,
            "result_task_name": case.expected_result_task_name,
        }
        if self.config.expose_attempt_detail:
            output["attempts"] = attempt_detail

        if self.config.aggregate_attempts == "mean" and attempts:
            rewards = [self._attempt_reward(attempt) for attempt in attempts]
            if any(reward is not None for reward in rewards):
                measured = [
                    self.config.failure_score if reward is None else reward
                    for reward in rewards
                ]
                score = sum(measured) / len(measured)
                output["attempt_scores"] = measured
                output["aggregate"] = "mean"
                # Split the zero-filled dead attempts so an outage-diluted mean
                # is distinguishable from a genuinely clean low mean: n_dead_infra
                # are dead-to-infrastructure; n_clean are the rest (scored or a
                # real candidate failure).
                n_dead_infra = sum(
                    1
                    for attempt, reward in zip(attempts, rewards)
                    if reward is None
                    and self._attempt_is_infra(attempt, trusted=trusted)
                )
                return (
                    CaseResult(
                        case_id=case.id,
                        status=CaseStatus.SUCCESS,
                        metrics={
                            "score": score,
                            "n_attempts": float(len(attempts)),
                            "n_scored": float(
                                sum(reward is not None for reward in rewards)
                            ),
                            "n_dead_infra": float(n_dead_infra),
                            "n_clean": float(len(attempts) - n_dead_infra),
                            **(
                                {"wall_seconds": wall_seconds}
                                if wall_seconds is not None
                                else {}
                            ),
                            **reported_tokens,
                        },
                        input={"task_name": case.task_name, **case.metadata},
                        output=output,
                        feedback=(
                            self._transcript_feedback(attempts)
                            if score == self.config.failure_score
                            else None
                        ),
                        artifacts=trial_artifacts,
                    ),
                    score,
                )

        best, score = self._best_attempt(attempts)
        if best is not None and score is not None:
            rewards = (best.get("verifier_result") or {}).get("rewards") or {}
            output.update(
                {
                    "trial_name": best.get("trial_name"),
                    "rewards": rewards,
                    "aggregate": "best",
                }
            )
            numeric_rewards = {
                key: float(value)
                for key, value in rewards.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
            numeric_rewards["score"] = score
            if wall_seconds is not None:
                numeric_rewards["wall_seconds"] = wall_seconds
            numeric_rewards.update(reported_tokens)
            return (
                CaseResult(
                    case_id=case.id,
                    status=CaseStatus.SUCCESS,
                    metrics=numeric_rewards,
                    input={"task_name": case.task_name, **case.metadata},
                    output=output,
                    feedback=(
                        self._transcript_feedback(attempts)
                        if score == self.config.failure_score
                        else None
                    ),
                    artifacts=trial_artifacts,
                ),
                score,
            )

        # No scored attempt. Decide, from the single source of truth, whether
        # this is the candidate's own failure (an informative low score) or an
        # infrastructure loss (excluded from the aggregate score).
        exception_counts: dict[str, int] = {}
        signals: list[str] = []
        for attempt in attempts:
            info = attempt.get("exception_info") or {}
            name = info.get("exception_type")
            key = str(name or NO_REWARD_SIGNAL)
            exception_counts[key] = exception_counts.get(key, 0) + 1
            signals.append(key)
            # The gateway embeds a distinct "budget_exhausted" code in the error
            # body; the in-container client collapses the type to a generic
            # rate-limit, but the message survives, so classify on it too.
            for detail_key in ("message", "detail", "exception_message"):
                detail = info.get(detail_key)
                if isinstance(detail, str) and detail:
                    signals.append(detail)

        if not attempts:
            # Harbor produced no trial for this task at all: infrastructure
            # dropped the case (see the coverage gate), not the candidate. This
            # coverage gap is harness-produced and stays excluded even for
            # competitive evaluations.
            category = ErrorCategory.TRANSIENT_INFRA
        else:
            category = classify_case(signals)
            if not trusted and category == ErrorCategory.TRANSIENT_INFRA:
                # A trial ran and died with a transient-looking exception. For
                # competitive (agent) selection we cannot trust a candidate-
                # controlled process's own exception text to mean
                # "infrastructure": excluding it would drop the case from its
                # own denominator and let a candidate inflate its mean by
                # emitting a timeout/connection error on hard cases. Score it at
                # the failure value instead. Genuine infrastructure is caught
                # out of band (coverage gaps above; gateway-ledger budget/auth,
                # which remain terminating) and via trusted-only retry.
                category = ErrorCategory.TASK_FAILURE
        category_policy = policy(category)
        output["dead_exception_types"] = exception_counts
        output["error_category"] = category.value

        if not attempts:
            message = (
                f"Harbor produced no trial for task {case.task_name!r} "
                "(infrastructure dropped the case)"
            )
        else:
            message = f"No verifier reward for Harbor task {case.task_name!r}"
            if exception_counts:
                causes = ", ".join(
                    f"{name} x{count}"
                    for name, count in sorted(exception_counts.items())
                )
                message += f"; attempts: {causes}"

        if category_policy.is_informative_sample:
            # The candidate produced no answer or crashed on its own: a real,
            # scoreable outcome at the failure value, not infrastructure noise.
            output["aggregate"] = "task_failure"
            return (
                CaseResult(
                    case_id=case.id,
                    status=CaseStatus.SUCCESS,
                    metrics={"score": self.config.failure_score},
                    input={"task_name": case.task_name, **case.metadata},
                    output=output,
                    feedback=self._transcript_feedback(attempts),
                    metadata={"error_category": category.value},
                    artifacts=trial_artifacts,
                ),
                self.config.failure_score,
            )

        # An infrastructure case: excluded from the aggregate, counted toward
        # invalidity, and — for budget/auth — a terminating condition surfaced
        # to the aggregation via its category.
        return (
            CaseResult(
                case_id=case.id,
                status=CaseStatus.ERROR,
                metrics={"score": self.config.failure_score},
                input={"task_name": case.task_name, **case.metadata},
                output=output,
                feedback=self._transcript_feedback(attempts),
                metadata={"error_category": category.value},
                artifacts=trial_artifacts,
                errors=[
                    CaseError(
                        message=message,
                        code=category_policy.diagnostic_code,
                        phase="harbor",
                        terminal=True,
                    )
                ],
            ),
            self.config.failure_score,
        )

    async def evaluate(
        self,
        *,
        context: EvaluationContext,
        request: EvaluationRequest,
    ) -> EvaluationReport:
        self.validate_request(request)
        target_root = context.workspace.sandbox.host_path(
            context.workspace.project_path
        )
        if target_root is not None:
            target_root = target_root.resolve()
            cases_path = Path(self.config.cases_path).resolve()
            if cases_path == target_root or cases_path.is_relative_to(target_root):
                raise ValueError("Harbor cases must live outside the editable target")
        source = Path(self.config.task_source).expanduser()
        local_task_source = source.exists()
        if local_task_source and target_root is not None:
            resolved_source = source.resolve()
            if resolved_source == target_root or resolved_source.is_relative_to(
                target_root
            ):
                raise ValueError(
                    "local Harbor tasks must live outside the editable target"
                )

        cases = self._selected_cases(request.evaluation_set)
        capture_dir = context.artifact_dir / "harbor"
        capture_dir.mkdir(parents=True, exist_ok=True)
        requested_tasks = {case.expected_result_task_name for case in cases}
        attempts: list[tuple[CommandResult, str, str]] = []
        groups: dict[str, list[dict[str, Any]]] = {}
        jobs_dir = capture_dir / "jobs"
        # Whole-sub-run infrastructure retry is a trusted-only affordance: for a
        # competitive (agent) evaluation, re-running the sub-run and keeping the
        # best coverage is a best-of-N amplifier over a candidate that can
        # itself trigger the retry. Trusted finalization re-scores, run by the
        # operator on a controlled environment, keep the configured retries.
        max_infra_attempts = (
            self.config.infrastructure_max_attempts if context.finalization else 1
        )
        for attempt in range(1, max_infra_attempts + 1):
            attempt_jobs_dir = (
                jobs_dir if attempt == 1 else capture_dir / f"retry-{attempt}" / "jobs"
            )
            attempt_jobs_dir.mkdir(parents=True, exist_ok=True)
            async with SandboxStagingArea(
                context.workspace.sandbox,
                prefix=f"vero-harbor-{context.evaluation_id[:8]}-{attempt}-",
            ) as staging:
                remote_jobs_dir = await staging.mkdir("jobs")
                task_source = (
                    (
                        str(source.resolve())
                        if context.workspace.sandbox.capabilities.host_paths
                        else await staging.upload(source.resolve(), "tasks")
                    )
                    if local_task_source
                    else self.config.task_source
                )
                retry_config_path = await staging.write_text(
                    "retry-config.json", self._retry_config_json()
                )
                command = self._command(
                    workspace=context.workspace.project_path,
                    request=request,
                    cases=cases,
                    jobs_dir=remote_jobs_dir,
                    task_source=task_source,
                    local_task_source=local_task_source,
                    retry_config_path=retry_config_path,
                )
                # Isolate the untrusted harness: hand the unprivileged user
                # ownership of exactly the dirs it must read and write (its
                # workspace and the staging tree), then run harbor as that user.
                # Everything trusted (session dir, budgets, other candidates,
                # admin token) is owned by root and unreadable to it.
                if self.config.harness_user is not None:
                    # Hand the harness user its work dirs and make the checkout
                    # reachable (the parent that `mktemp -d` left 0700 root); see
                    # vero.harbor.isolation for the why.
                    for provision_command in harness_grant_commands(
                        self.config.harness_user,
                        chown_paths=[context.workspace.project_path, staging.root],
                        checkout_root=context.workspace.root,
                    ):
                        provision = await context.workspace.sandbox.run(
                            provision_command, timeout=120
                        )
                        if provision.returncode != 0:
                            raise RuntimeError(
                                "failed to provision harness workspace "
                                f"({' '.join(provision_command)}): "
                                f"{self.sanitize_error(provision.stderr)}"
                            )
                    # Fail fast, at the provisioning site, if the dropped user
                    # still can't reach its workspace, instead of a cryptic
                    # "No module named <agent>" several retries downstream.
                    probe = await context.workspace.sandbox.run(
                        harness_reachability_probe(context.workspace.project_path),
                        run_as=self.config.harness_user,
                    )
                    if probe.returncode != 0:
                        raise RuntimeError(
                            f"harness {self.config.harness_user!r} cannot reach its "
                            f"workspace {context.workspace.project_path!r} after "
                            "provisioning; check that every ancestor directory is "
                            "traversable by the dropped user"
                        )
                result = await context.workspace.sandbox.run(
                    command,
                    cwd=context.workspace.project_path,
                    timeout=request.limits.timeout_seconds,
                    env=self._environment(
                        context.evaluation_id, finalization=context.finalization
                    ),
                    run_as=self.config.harness_user,
                )
                await staging.download("jobs", attempt_jobs_dir)
            stdout = self.sanitize_error(result.stdout)
            stderr = self.sanitize_error(result.stderr)
            attempts.append((result, stdout, stderr))
            groups = self._trial_groups(attempt_jobs_dir)
            # Require FULL coverage before accepting the sub-run. Partial
            # coverage used to be accepted silently, scoring the dropped tasks
            # as zeros; now we retry, and any tasks still missing after retries
            # are surfaced as infrastructure cases (excluded from the mean,
            # counted toward invalidity) rather than folded in as zeros.
            if requested_tasks <= set(groups):
                jobs_dir = attempt_jobs_dir
                break
            if attempt < max_infra_attempts:
                await asyncio.sleep(
                    self.config.infrastructure_retry_delay_seconds * attempt
                )

        result, stdout, stderr = attempts[-1]
        (capture_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (capture_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        artifacts = [
            EvaluationArtifact(
                path="harbor/stdout.log",
                media_type="text/plain",
                description="Harbor standard output",
            ),
            EvaluationArtifact(
                path="harbor/stderr.log",
                media_type="text/plain",
                description="Harbor standard error",
            ),
        ]

        matching_tasks = requested_tasks & set(groups)
        if not matching_tasks:
            # Distinguish "sub-run produced nothing" from "produced trials whose
            # task_name doesn't match what we requested" — the two collapse to the
            # same failure otherwise. Surface the found group keys + exit status.
            detail = (
                f"no matching trials for {len(cases)} requested tasks after "
                f"{len(attempts)} attempts; requested e.g. {sorted(requested_tasks)[:2]}, "
                f"harbor produced {len(groups)} trial group(s) e.g. {sorted(groups)[:3]}, "
                f"returncode={result.returncode}"
            )
            message = (stderr.strip() + " || " + detail) if stderr.strip() else detail
            report = EvaluationReport(
                status=EvaluationStatus.FAILED,
                diagnostics=[
                    EvaluationDiagnostic(
                        code="infrastructure_failure",
                        message=message,
                        severity=DiagnosticSeverity.ERROR,
                        phase="harbor",
                    )
                ],
                artifacts=artifacts,
            )
            return sanitize_evaluation_report(report, self._secrets())

        case_results: list[CaseResult] = []
        scores: list[float] = []
        for case in cases:
            case_result, score = self._case_result(
                case,
                groups.get(case.expected_result_task_name, []),
                artifact_root=context.artifact_dir,
                trusted=context.finalization,
            )
            case_results.append(case_result)
            scores.append(score)
            await context.case_store.save(case_result)
        diagnostics = []
        if result.returncode != 0:
            diagnostics.append(
                EvaluationDiagnostic(
                    code=(
                        "harbor_partial_timeout"
                        if result.returncode == -1
                        else "harbor_nonzero_exit"
                    ),
                    message=stderr.strip()
                    or f"Harbor exited with status {result.returncode}; partial trials collated",
                    severity=DiagnosticSeverity.WARNING,
                    phase="harbor",
                )
            )

        # Coverage: any requested task that produced no trial is an
        # infrastructure loss, recorded loudly rather than silently zeroed.
        missing_tasks = sorted(requested_tasks - set(groups))
        if missing_tasks:
            diagnostics.append(
                EvaluationDiagnostic(
                    code="harbor_incomplete_coverage",
                    message=(
                        f"{len(missing_tasks)} of {len(requested_tasks)} requested "
                        f"tasks produced no trial after {len(attempts)} attempt(s): "
                        f"{missing_tasks[:5]}"
                    ),
                    severity=DiagnosticSeverity.ERROR,
                    phase="harbor",
                )
            )

        # Infrastructure cases are excluded from the aggregate score; only
        # informative samples (successes and legitimate task failures) count.
        informative_scores = [
            score
            for case, score in zip(case_results, scores)
            if case.status != CaseStatus.ERROR
        ]
        infra_cases = [
            case for case in case_results if case.status == CaseStatus.ERROR
        ]

        def _category(case: CaseResult) -> ErrorCategory | None:
            raw = case.metadata.get("error_category")
            try:
                return ErrorCategory(raw) if isinstance(raw, str) else None
            except ValueError:
                return None

        # A terminating condition (inference-budget exhaustion or auth failure)
        # anywhere fails the whole evaluation loudly: distinct code, no score.
        terminating = next(
            (
                _category(case)
                for case in infra_cases
                if (category := _category(case)) is not None
                and policy(category).terminating
            ),
            None,
        )
        if terminating is not None:
            report = EvaluationReport(
                status=EvaluationStatus.INVALID,
                metrics={
                    "error_rate": len(infra_cases) / len(case_results),
                },
                cases=case_results,
                diagnostics=[
                    *diagnostics,
                    EvaluationDiagnostic(
                        code=policy(terminating).diagnostic_code,
                        message=(
                            f"terminating condition {terminating.value!r} reached; "
                            "the run cannot continue and must not be retried"
                        ),
                        severity=DiagnosticSeverity.ERROR,
                        phase="harbor",
                    ),
                ],
                artifacts=artifacts,
            )
            return sanitize_evaluation_report(report, self._secrets())

        # No informative sample survived (every case was infrastructure): the
        # aggregate is undefined, so the whole evaluation is invalid. Retryable
        # transient loss keeps the historical infrastructure_failure code so the
        # engine still refunds and lets the optimizer retry.
        if not informative_scores:
            diagnostics.append(
                EvaluationDiagnostic(
                    code="infrastructure_failure",
                    message=(
                        "no informative sample survived; every case was lost to "
                        "infrastructure after Harbor retries were exhausted"
                    ),
                    severity=DiagnosticSeverity.ERROR,
                    phase="harbor",
                )
            )
            report = EvaluationReport(
                status=EvaluationStatus.INVALID,
                metrics={"error_rate": len(infra_cases) / len(case_results)},
                cases=case_results,
                diagnostics=diagnostics,
                artifacts=artifacts,
            )
            return sanitize_evaluation_report(report, self._secrets())

        case_walls = [
            wall
            for case in case_results
            if isinstance((wall := case.metrics.get("wall_seconds")), (int, float))
        ]
        reported_totals: dict[str, float] = {}
        for case in case_results:
            for key, value in case.metrics.items():
                if key.startswith("agent_reported_") and isinstance(
                    value, (int, float)
                ):
                    total_key = key.replace("agent_reported_", "agent_reported_total_")
                    reported_totals[total_key] = (
                        reported_totals.get(total_key, 0.0) + float(value)
                    )
        report = EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={
                "score": sum(informative_scores) / len(informative_scores),
                "error_rate": len(infra_cases) / len(case_results),
                # Spread across informative cases, so a real difference between
                # candidates is distinguishable from evaluation noise.
                "score_stddev": (
                    statistics.pstdev(informative_scores)
                    if len(informative_scores) > 1
                    else 0.0
                ),
                # Cost/latency telemetry: reported alongside accuracy, never
                # part of the score.
                **(
                    {
                        "mean_case_wall_seconds": sum(case_walls) / len(case_walls),
                        "max_case_wall_seconds": max(case_walls),
                    }
                    if case_walls
                    else {}
                ),
                **reported_totals,
                **self._inference_usage_metrics(context.evaluation_id),
            },
            cases=case_results,
            diagnostics=diagnostics,
            artifacts=artifacts,
        )
        return sanitize_evaluation_report(report, self._secrets())
