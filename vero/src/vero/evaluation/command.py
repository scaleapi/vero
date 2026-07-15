"""Language- and framework-neutral command evaluation backend."""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from vero.evaluation.backend import EvaluationContext
from vero.evaluation.models import (
    AllCases,
    BackendProvenance,
    CaseIds,
    CaseRange,
    CommandEvaluationInput,
    DiagnosticSeverity,
    EvaluationArtifact,
    EvaluationCost,
    EvaluationDiagnostic,
    EvaluationModel,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
)
from vero.evaluation.security import sanitize_evaluation_report, sanitize_text

_PLACEHOLDERS = {"workspace", "request", "report", "artifacts"}
_PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")


class CommandBackendConfig(EvaluationModel):
    harness_root: str
    command: list[str]
    working_directory: str = "."
    environment: dict[str, str] = Field(default_factory=dict)
    passthrough_environment: list[str] = Field(default_factory=list)

    @field_validator("harness_root")
    @classmethod
    def validate_harness_root(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("harness_root must not be empty")
        if not Path(value).is_absolute():
            raise ValueError("harness_root must be absolute after config resolution")
        return value

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: list[str]) -> list[str]:
        if not value or any(not argument for argument in value):
            raise ValueError("command and its arguments must not be empty")
        unknown = {
            placeholder
            for argument in value
            for placeholder in _PLACEHOLDER_PATTERN.findall(argument)
            if placeholder not in _PLACEHOLDERS
        }
        if unknown:
            raise ValueError(
                f"unknown command placeholders: {', '.join(sorted(unknown))}"
            )
        return value

    @field_validator("working_directory")
    @classmethod
    def validate_working_directory(cls, value: str) -> str:
        path = Path(value)
        if not value.strip() or path.is_absolute() or ".." in path.parts:
            raise ValueError("working_directory must stay within harness_root")
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
    def validate_passthrough(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("passthrough_environment names must be unique")
        for name in value:
            if not name or "=" in name:
                raise ValueError(f"invalid environment variable name: {name!r}")
        return value

    @model_validator(mode="after")
    def validate_environment_sources(self) -> CommandBackendConfig:
        overlap = set(self.environment) & set(self.passthrough_environment)
        if overlap:
            raise ValueError(
                "environment and passthrough_environment overlap for: "
                + ", ".join(sorted(overlap))
            )
        return self


class CommandBackend:
    """Invoke a trusted external harness through a versioned JSON contract."""

    name = "command"
    version = "1"

    def __init__(self, config: CommandBackendConfig):
        self.config = config

    @property
    def provenance(self) -> BackendProvenance:
        return BackendProvenance.from_config(
            name=self.name,
            version=self.version,
            config=self.config,
        )

    async def resolve_cost(self, evaluation_set: EvaluationSet) -> EvaluationCost:
        selection = evaluation_set.selection
        if isinstance(selection, CaseIds):
            return EvaluationCost(cases=len(selection.ids))
        if isinstance(selection, CaseRange):
            return EvaluationCost(cases=selection.stop - selection.start)
        if isinstance(selection, AllCases):
            return EvaluationCost(cases=None)
        raise AssertionError(f"unsupported case selection: {selection}")

    def _working_directory(self) -> Path:
        root = Path(self.config.harness_root).resolve()
        working_directory = (root / self.config.working_directory).resolve()
        if not working_directory.is_relative_to(root):
            raise ValueError("working_directory escapes harness_root")
        return working_directory

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
        payload = request.model_dump_json()
        if any(secret in payload for secret in self._secrets() if len(secret) >= 4):
            raise ValueError(
                "evaluation parameters must not contain configured secret values; "
                "pass secrets through the backend environment"
            )

    def _expand_command(self, values: dict[str, str]) -> list[str]:
        command: list[str] = []
        for argument in self.config.command:
            expanded = argument
            for placeholder, value in values.items():
                expanded = expanded.replace(f"{{{placeholder}}}", value)
            command.append(expanded)
        return command

    @staticmethod
    def _failure_report(
        *,
        code: str,
        message: str,
        artifacts: list[EvaluationArtifact],
    ) -> EvaluationReport:
        return EvaluationReport(
            status=EvaluationStatus.FAILED,
            diagnostics=[
                EvaluationDiagnostic(
                    code=code,
                    message=message,
                    severity=DiagnosticSeverity.ERROR,
                    phase="command",
                )
            ],
            artifacts=artifacts,
        )

    async def evaluate(
        self,
        *,
        context: EvaluationContext,
        request: EvaluationRequest,
    ) -> EvaluationReport:
        harness_root = Path(self.config.harness_root).resolve()
        target_root = Path(context.workspace.project_path).resolve()
        if harness_root == target_root or harness_root.is_relative_to(target_root):
            raise ValueError("command harness must live outside the editable target")

        backend_dir = context.result_dir / "backend" / self.name
        backend_dir.mkdir(parents=True, exist_ok=True)
        capture_dir = context.artifact_dir / "command"
        capture_dir.mkdir(parents=True, exist_ok=True)
        request_path = backend_dir / "request.json"
        report_path = backend_dir / "report.json"
        request_path.write_text(
            CommandEvaluationInput(request=request).model_dump_json(indent=2),
            encoding="utf-8",
        )

        command = self._expand_command(
            {
                "workspace": context.workspace.project_path,
                "request": str(request_path),
                "report": str(report_path),
                "artifacts": str(context.artifact_dir),
            }
        )
        result = await context.workspace.sandbox.run(
            command,
            cwd=str(self._working_directory()),
            timeout=request.limits.timeout_seconds,
            env=self._environment(),
        )

        stdout = self.sanitize_error(result.stdout)
        stderr = self.sanitize_error(result.stderr)
        (capture_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (capture_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        capture_artifacts = [
            EvaluationArtifact(
                path="command/stdout.log",
                media_type="text/plain",
                description="Command harness standard output",
            ),
            EvaluationArtifact(
                path="command/stderr.log",
                media_type="text/plain",
                description="Command harness standard error",
            ),
        ]

        if result.returncode != 0:
            code = "command_timeout" if result.returncode == -1 else "command_failed"
            message = stderr.strip() or f"evaluation command exited with status {result.returncode}"
            return self._failure_report(
                code=code,
                message=message,
                artifacts=capture_artifacts,
            )
        if not report_path.exists():
            return self._failure_report(
                code="missing_report",
                message="evaluation command did not write a report",
                artifacts=capture_artifacts,
            )
        try:
            report = EvaluationReport.model_validate_json(
                report_path.read_text(encoding="utf-8")
            )
        except Exception as error:
            return self._failure_report(
                code="invalid_report",
                message=self.sanitize_error(
                    f"evaluation command wrote an invalid report: {error}"
                ),
                artifacts=capture_artifacts,
            )
        report = sanitize_evaluation_report(report, self._secrets())
        return report.model_copy(
            update={"artifacts": [*report.artifacts, *capture_artifacts]}
        )
