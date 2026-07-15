"""Trusted command candidate producer."""

from __future__ import annotations

import os
import posixpath
import re
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from vero.evaluation import EvaluationModel
from vero.optimization.models import (
    CandidateChange,
    CandidateProposal,
    OptimizationContext,
)
from vero.optimization.protocols import CandidateEvaluationGateway
from vero.staging import SandboxStagingArea
from vero.workspace import Workspace

_PLACEHOLDERS = {
    "workspace",
    "producer",
    "round",
    "instruction",
    "best_candidate_id",
    "best_version",
    "best_value",
}
_PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")


class CommandCandidateProducerConfig(EvaluationModel):
    root: str
    command: list[str]
    working_directory: str = "."
    environment: dict[str, str] = Field(default_factory=dict)
    passthrough_environment: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=600.0, gt=0.0)
    description: str = "Optimize candidate"

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: str) -> str:
        if not value.strip() or not Path(value).is_absolute():
            raise ValueError("producer root must be absolute after config resolution")
        return value

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: list[str]) -> list[str]:
        if not value or any(not argument for argument in value):
            raise ValueError("producer command and its arguments must not be empty")
        unknown = {
            placeholder
            for argument in value
            for placeholder in _PLACEHOLDER_PATTERN.findall(argument)
            if placeholder not in _PLACEHOLDERS
        }
        if unknown:
            raise ValueError(
                f"unknown producer placeholders: {', '.join(sorted(unknown))}"
            )
        return value

    @field_validator("working_directory")
    @classmethod
    def validate_working_directory(cls, value: str) -> str:
        path = Path(value)
        if not value.strip() or path.is_absolute() or ".." in path.parts:
            raise ValueError("producer working_directory must stay within its root")
        return value

    @field_validator("passthrough_environment")
    @classmethod
    def validate_passthrough(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("producer passthrough environment names must be unique")
        return value

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("producer description must not be empty")
        return value

    @model_validator(mode="after")
    def validate_environment(self) -> CommandCandidateProducerConfig:
        overlap = set(self.environment) & set(self.passthrough_environment)
        if overlap:
            raise ValueError(
                "producer environment sources overlap for: "
                + ", ".join(sorted(overlap))
            )
        return self


class CommandCandidateProducer:
    """Run a trusted command that edits the supplied candidate workspace."""

    def __init__(self, config: CommandCandidateProducerConfig):
        self.config = config

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

    async def produce(
        self,
        *,
        proposal: CandidateProposal,
        context: OptimizationContext,
        workspace: Workspace,
        evaluation: CandidateEvaluationGateway,
    ) -> CandidateChange | None:
        root = Path(self.config.root).resolve()
        target = workspace.sandbox.host_path(workspace.project_path)
        if target is not None:
            target = target.resolve()
            if root == target or root.is_relative_to(target):
                raise ValueError("candidate producer must live outside the editable target")

        best = context.best
        async with SandboxStagingArea(
            workspace.sandbox,
            prefix=f"vero-producer-{proposal.id[:8]}-",
        ) as staging:
            producer_root = (
                str(root)
                if workspace.sandbox.capabilities.host_paths
                else await staging.upload(root, "producer")
            )
            working_directory = posixpath.normpath(
                posixpath.join(producer_root, self.config.working_directory)
            )
            values = {
                "workspace": workspace.project_path,
                "producer": producer_root,
                "round": str(context.round),
                "instruction": proposal.instruction or "",
                "best_candidate_id": best.request.candidate.id if best else "",
                "best_version": best.request.candidate.version if best else "",
                "best_value": (
                    str(best.objective.value)
                    if best is not None
                    and best.objective is not None
                    and best.objective.value is not None
                    else ""
                ),
            }
            command: list[str] = []
            for argument in self.config.command:
                expanded = argument
                for placeholder, value in values.items():
                    expanded = expanded.replace(f"{{{placeholder}}}", value)
                command.append(expanded)

            result = await workspace.sandbox.run(
                command,
                cwd=working_directory,
                timeout=self.config.timeout_seconds,
                env=self._environment(),
            )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr
                or f"candidate producer exited with status {result.returncode}"
            )
        return CandidateChange(description=self.config.description)
