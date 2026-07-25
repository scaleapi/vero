"""Language- and framework-neutral command evaluation backend."""

from __future__ import annotations

import json
import os
import posixpath
import re
from pathlib import Path, PurePosixPath
from typing import Literal

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
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
)
from vero.evaluation.security import sanitize_evaluation_report, sanitize_text
from vero.models import StrictModel
from vero.sandbox import Sandbox
from vero.staging import SandboxStagingArea

_PLACEHOLDERS = {"workspace", "harness", "request", "report", "artifacts"}
_PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")
_INPUT_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


class CommandBackendConfig(StrictModel):
    # Discriminates this from the other backend configs a deployment may name.
    type: Literal["command"] = "command"
    harness_root: str
    command: list[str]
    working_directory: str = "."
    environment: dict[str, str] = Field(default_factory=dict)
    passthrough_environment: list[str] = Field(default_factory=list)
    staged_inputs: dict[str, str] = Field(default_factory=dict)
    agent_context_inputs: dict[str, list[str]] = Field(default_factory=dict)

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
            if placeholder not in _PLACEHOLDERS and not placeholder.startswith("input:")
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
        invalid = sorted(
            name
            for name in self.staged_inputs
            if not _INPUT_NAME_PATTERN.fullmatch(name)
        )
        if invalid:
            raise ValueError(f"invalid staged input names: {', '.join(invalid)}")
        referenced = {
            placeholder.removeprefix("input:")
            for argument in self.command
            for placeholder in _PLACEHOLDER_PATTERN.findall(argument)
            if placeholder.startswith("input:")
        }
        unknown = sorted(referenced - set(self.staged_inputs))
        if unknown:
            raise ValueError(f"unknown staged command inputs: {', '.join(unknown)}")
        for evaluation, names in self.agent_context_inputs.items():
            if not evaluation.strip():
                raise ValueError("agent_context_inputs evaluation names must not be empty")
            if len(names) != len(set(names)):
                raise ValueError(
                    f"agent_context_inputs for {evaluation!r} must be unique"
                )
        unknown_context = sorted(
            {
                name
                for names in self.agent_context_inputs.values()
                for name in names
            }
            - set(self.staged_inputs)
        )
        if unknown_context:
            raise ValueError(
                "agent_context_inputs reference unknown staged inputs: "
                + ", ".join(unknown_context)
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

    async def export_case_resources(
        self,
        *,
        evaluation_set: EvaluationSet,
        destination: str,
        sandbox: Sandbox,
    ) -> None:
        """Copy only explicitly allowlisted staged inputs into agent context."""

        resources = []
        for name in self.config.agent_context_inputs.get(evaluation_set.name, []):
            source = Path(self.config.staged_inputs[name]).resolve()
            if not source.exists():
                raise ValueError(
                    f"agent context input {name!r} does not exist: {source}"
                )
            target = str(PurePosixPath(destination) / name)
            await sandbox.upload(str(source), target)
            resources.append({"name": name, "path": name})
        await sandbox.write_file(
            str(PurePosixPath(destination) / "index.json"),
            json.dumps(
                {
                    "schema_version": 1,
                    "evaluation_set": evaluation_set.model_dump(mode="json"),
                    "resources": resources,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )

    def _working_directory(self, harness_root: str) -> str:
        return posixpath.normpath(
            posixpath.join(harness_root, self.config.working_directory)
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
        harness_source = Path(self.config.harness_root).resolve()
        target_root = context.workspace.sandbox.host_path(
            context.workspace.project_path
        )
        if target_root is not None:
            target_root = target_root.resolve()
            if harness_source == target_root or harness_source.is_relative_to(
                target_root
            ):
                raise ValueError(
                    "command harness must live outside the editable target"
                )

        capture_dir = context.artifact_dir / "command"
        capture_dir.mkdir(parents=True, exist_ok=True)
        async with SandboxStagingArea(
            context.workspace.sandbox,
            prefix=f"vero-eval-{context.evaluation_id[:8]}-",
        ) as staging:
            harness_root = (
                str(harness_source)
                if context.workspace.sandbox.capabilities.host_paths
                else await staging.upload(harness_source, "harness")
            )
            staged_inputs = {
                f"input:{name}": await staging.upload(source, f"inputs/{name}")
                for name, source in self.config.staged_inputs.items()
            }
            request_path = await staging.write_text(
                "request.json",
                CommandEvaluationInput(request=request).model_dump_json(indent=2),
            )
            report_path = staging.path("report.json")
            artifacts_path = await staging.mkdir("artifacts")

            command = self._expand_command(
                {
                    "workspace": context.workspace.project_path,
                    "harness": harness_root,
                    "request": request_path,
                    "report": report_path,
                    "artifacts": artifacts_path,
                    **staged_inputs,
                }
            )
            result = await context.workspace.sandbox.run(
                command,
                cwd=self._working_directory(harness_root),
                timeout=request.limits.timeout_seconds,
                env=self._environment(),
            )

            if await staging.exists("artifacts"):
                await staging.download("artifacts", context.artifact_dir)

            report_payload = (
                await staging.read_text("report.json")
                if await staging.exists("report.json")
                else None
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
            message = (
                stderr.strip()
                or f"evaluation command exited with status {result.returncode}"
            )
            return self._failure_report(
                code=code,
                message=message,
                artifacts=capture_artifacts,
            )
        if report_payload is None:
            return self._failure_report(
                code="missing_report",
                message="evaluation command did not write a report",
                artifacts=capture_artifacts,
            )
        try:
            report = EvaluationReport.model_validate_json(report_payload)
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
