"""Evaluate a candidate program by running it over Harbor tasks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from vero.evaluation.backend import EvaluationContext
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
    EvaluationLimits,
    EvaluationModel,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
)
from vero.evaluation.security import sanitize_evaluation_report, sanitize_text
from vero.staging import SandboxStagingArea
from vero.sandbox import CommandResult, Sandbox


def _default_uv() -> str:
    executable = shutil.which("uv")
    if executable is None:
        raise ValueError("uv is required to configure a Harbor backend")
    return str(Path(executable).resolve())


class HarborCase(EvaluationModel):
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


class HarborBackendConfig(EvaluationModel):
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
    n_attempts: int = Field(default=1, ge=1)
    max_retries: int = Field(default=2, ge=0)
    infrastructure_max_attempts: int = Field(default=3, ge=1)
    infrastructure_retry_delay_seconds: float = Field(default=5.0, ge=0)
    infrastructure_exception_patterns: list[str] = Field(
        default_factory=lambda: [
            "rate.?limit",
            "timeout",
            "connection",
            "service.?unavailable",
            "internal.?server",
            "overloaded",
            "authentication",
            "permission",
            "quota",
            "insufficient.?credits",
            "billing",
        ]
    )
    reward_key: str | None = None
    aggregate_attempts: Literal["best", "mean"] = "best"
    failure_score: float = 0.0
    feedback_transcripts: bool = False
    feedback_max_bytes: int = Field(default=3000, ge=0)
    expose_attempt_detail: bool = False
    uv_executable: str = Field(default_factory=_default_uv)
    default_index: str = "https://pypi.org/simple"
    environment: dict[str, str] = Field(default_factory=dict)
    passthrough_environment: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("cases_path")
    @classmethod
    def validate_absolute_file(cls, value: str) -> str:
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

    @field_validator("partition", "model", "reward_key")
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

    @field_validator("infrastructure_exception_patterns")
    @classmethod
    def validate_infrastructure_patterns(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("infrastructure exception patterns must not be empty")
        for pattern in value:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as error:
                raise ValueError(
                    f"invalid infrastructure exception pattern {pattern!r}: {error}"
                ) from error
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
        """Expose configured case identities, never the hidden task source."""

        index = []
        for case in self._selected_cases(evaluation_set):
            digest = hashlib.sha256(case.id.encode()).hexdigest()
            filename = f"{digest}.json"
            await sandbox.write_file(
                str(PurePosixPath(destination) / filename),
                case.model_dump_json(indent=2) + "\n",
            )
            index.append({"case_id": case.id, "path": filename})
        await sandbox.write_file(
            str(PurePosixPath(destination) / "index.json"),
            json.dumps(
                {
                    "schema_version": 1,
                    "evaluation_set": evaluation_set.model_dump(mode="json"),
                    "cases": index,
                    "task_source_exposed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )

    def _secrets(self) -> list[str]:
        values = list(self.config.environment.values())
        values.extend(
            os.environ[name]
            for name in self.config.passthrough_environment
            if name in os.environ
        )
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
        default_case_timeout = EvaluationLimits().case_timeout_seconds
        if request.limits.case_timeout_seconds != default_case_timeout:
            raise ValueError(
                "Harbor does not support an absolute per-case timeout override; "
                "configure task timeouts or Harbor timeout multipliers instead"
            )
        if request.seed is not None:
            raise ValueError("Harbor does not support the generic evaluation seed")
        payload = request.model_dump_json()
        if any(secret in payload for secret in self._secrets() if len(secret) >= 4):
            raise ValueError(
                "evaluation parameters must not contain configured secret values; "
                "pass secrets through the backend environment"
            )

    def _environment(self) -> dict[str, str]:
        environment = {"PATH": os.defpath, "LANG": "C.UTF-8"}
        for name in ("TMPDIR", "TMP", "TEMP", "SYSTEMROOT"):
            if name in os.environ:
                environment[name] = os.environ[name]
        environment.update(self.config.environment)
        for name in self.config.passthrough_environment:
            if name in os.environ:
                environment[name] = os.environ[name]
        return environment

    def _source_args(self, task_source: str, *, local: bool) -> list[str]:
        return ["-p", task_source] if local else ["-d", task_source]

    def _command(
        self,
        *,
        workspace: str,
        request: EvaluationRequest,
        cases: list[HarborCase],
        jobs_dir: str,
        task_source: str,
        local_task_source: bool,
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
            *self._source_args(task_source, local=local_task_source),
            "--agent-import-path",
            self.config.agent_import_path,
            "-e",
            self.config.environment_name,
            "-n",
            str(request.limits.max_concurrency),
            "--n-attempts",
            str(self.config.n_attempts),
            "--max-retries",
            str(self.config.max_retries),
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

    def _case_result(
        self,
        case: HarborCase,
        attempts: list[dict[str, Any]],
    ) -> tuple[CaseResult, float]:
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
                        },
                        input={"task_name": case.task_name, **case.metadata},
                        output=output,
                        feedback=(
                            self._transcript_feedback(attempts)
                            if score == self.config.failure_score
                            else None
                        ),
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
                ),
                score,
            )

        exception_counts: dict[str, int] = {}
        for attempt in attempts:
            name = (attempt.get("exception_info") or {}).get("exception_type")
            key = str(name or "no_rewards_recorded")
            exception_counts[key] = exception_counts.get(key, 0) + 1
        message = f"No verifier reward for Harbor task {case.task_name!r}"
        if exception_counts:
            causes = ", ".join(
                f"{name} x{count}" for name, count in sorted(exception_counts.items())
            )
            message += f"; attempts: {causes}"
        output["dead_exception_types"] = exception_counts
        return (
            CaseResult(
                case_id=case.id,
                status=CaseStatus.ERROR,
                metrics={"score": self.config.failure_score},
                input={"task_name": case.task_name, **case.metadata},
                output=output,
                feedback=self._transcript_feedback(attempts),
                errors=[
                    CaseError(
                        message=message,
                        code="harbor_no_reward",
                        phase="harbor",
                        terminal=True,
                    )
                ],
            ),
            self.config.failure_score,
        )

    def _only_infrastructure_failures(
        self,
        case_results: list[CaseResult],
    ) -> bool:
        if not case_results or any(
            case.status != CaseStatus.ERROR for case in case_results
        ):
            return False
        patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.infrastructure_exception_patterns
        ]
        for case in case_results:
            output = case.output if isinstance(case.output, dict) else {}
            names = output.get("dead_exception_types")
            if not isinstance(names, dict) or not names:
                return False
            if any(
                not any(pattern.search(str(name)) for pattern in patterns)
                for name in names
            ):
                return False
        return True

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
        for attempt in range(1, self.config.infrastructure_max_attempts + 1):
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
                command = self._command(
                    workspace=context.workspace.project_path,
                    request=request,
                    cases=cases,
                    jobs_dir=remote_jobs_dir,
                    task_source=task_source,
                    local_task_source=local_task_source,
                )
                result = await context.workspace.sandbox.run(
                    command,
                    cwd=context.workspace.project_path,
                    timeout=request.limits.timeout_seconds,
                    env=self._environment(),
                )
                await staging.download("jobs", attempt_jobs_dir)
            stdout = self.sanitize_error(result.stdout)
            stderr = self.sanitize_error(result.stderr)
            attempts.append((result, stdout, stderr))
            groups = self._trial_groups(attempt_jobs_dir)
            if requested_tasks & set(groups):
                jobs_dir = attempt_jobs_dir
                break
            if attempt < self.config.infrastructure_max_attempts:
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
            message = stderr.strip() or (
                "Harbor infrastructure produced no matching trials for "
                f"{len(cases)} requested tasks after {len(attempts)} attempts"
            )
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
        infrastructure_failure = self._only_infrastructure_failures(case_results)
        if infrastructure_failure:
            diagnostics.append(
                EvaluationDiagnostic(
                    code="infrastructure_failure",
                    message=(
                        "All Harbor cases failed with transient infrastructure "
                        "exceptions after Harbor retries were exhausted"
                    ),
                    severity=DiagnosticSeverity.ERROR,
                    phase="harbor",
                )
            )
        report = EvaluationReport(
            status=(
                EvaluationStatus.FAILED
                if infrastructure_failure
                else EvaluationStatus.SUCCESS
            ),
            metrics={
                "score": sum(scores) / len(scores),
                "error_rate": sum(
                    case.status == CaseStatus.ERROR for case in case_results
                )
                / len(case_results),
            },
            cases=case_results,
            diagnostics=diagnostics,
            artifacts=artifacts,
        )
        return sanitize_evaluation_report(report, self._secrets())
