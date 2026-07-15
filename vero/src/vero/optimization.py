"""Dataset-free optimization loop over versioned program candidates."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import Field, JsonValue, field_validator, model_validator

from vero.core.db.candidate import Candidate
from vero.evaluation.engine import EvaluationEngine
from vero.evaluation.models import (
    EvaluationLimits,
    EvaluationModel,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationSet,
    ObjectiveSpec,
)
from vero.workspace import Workspace

_OPTIMIZER_PLACEHOLDERS = {
    "workspace",
    "iteration",
    "best_commit",
    "best_value",
}
_PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")


@dataclass(frozen=True)
class OptimizationContext:
    workspace: Workspace
    iteration: int
    baseline: EvaluationRecord
    evaluations: tuple[EvaluationRecord, ...]
    best: EvaluationRecord | None


@runtime_checkable
class CandidateProducer(Protocol):
    async def propose(self, context: OptimizationContext) -> str | None:
        """Edit the workspace and return a commit message, or ``None`` to stop."""
        ...


class CommandCandidateProducerConfig(EvaluationModel):
    root: str
    command: list[str]
    working_directory: str = "."
    environment: dict[str, str] = Field(default_factory=dict)
    passthrough_environment: list[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=600, gt=0)
    commit_message: str = "Optimize candidate"

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("optimizer root must be absolute after config resolution")
        return value

    @field_validator("command")
    @classmethod
    def validate_command(cls, command: list[str]) -> list[str]:
        if not command or any(not argument for argument in command):
            raise ValueError("optimizer command and its arguments must not be empty")
        unknown = {
            placeholder
            for argument in command
            for placeholder in _PLACEHOLDER_PATTERN.findall(argument)
            if placeholder not in _OPTIMIZER_PLACEHOLDERS
        }
        if unknown:
            raise ValueError(
                f"unknown optimizer placeholders: {', '.join(sorted(unknown))}"
            )
        return command

    @field_validator("working_directory")
    @classmethod
    def validate_working_directory(cls, value: str) -> str:
        path = Path(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise ValueError("optimizer working_directory must stay within its root")
        return value

    @field_validator("passthrough_environment")
    @classmethod
    def validate_passthrough(cls, names: list[str]) -> list[str]:
        if len(names) != len(set(names)):
            raise ValueError("optimizer passthrough environment names must be unique")
        return names

    @model_validator(mode="after")
    def validate_environment(self) -> CommandCandidateProducerConfig:
        overlap = set(self.environment) & set(self.passthrough_environment)
        if overlap:
            raise ValueError(
                "optimizer environment sources overlap for: "
                + ", ".join(sorted(overlap))
            )
        if not self.commit_message.strip():
            raise ValueError("optimizer commit_message must not be empty")
        return self


class CommandCandidateProducer:
    """Run a trusted external optimizer command that edits the target workspace."""

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

    async def propose(self, context: OptimizationContext) -> str | None:
        root = Path(self.config.root).resolve()
        working_directory = (root / self.config.working_directory).resolve()
        if not working_directory.is_relative_to(root):
            raise ValueError("optimizer working directory escapes its trusted root")
        best = context.best
        values = {
            "workspace": context.workspace.project_path,
            "iteration": str(context.iteration),
            "best_commit": best.request.candidate.commit if best else "",
            "best_value": (
                str(best.objective.value)
                if best is not None
                and best.objective is not None
                and best.objective.value is not None
                else ""
            ),
        }
        command = []
        for argument in self.config.command:
            expanded = argument
            for placeholder, value in values.items():
                expanded = expanded.replace(f"{{{placeholder}}}", value)
            command.append(expanded)
        result = await context.workspace.sandbox.run(
            command,
            cwd=str(working_directory),
            timeout=self.config.timeout_seconds,
            env=self._environment(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr
                or f"optimizer command exited with status {result.returncode}"
            )
        return self.config.commit_message


class AgentCandidateProducer:
    """Adapt the existing single coding-agent step into candidate production."""

    def __init__(
        self,
        agent,
        *,
        prompt: str | None,
        max_turns: int,
        on_event: Callable[[Any], None] | None = None,
        commit_message: str = "Optimize candidate",
    ):
        self.agent = agent
        self.prompt = prompt
        self.max_turns = max_turns
        self.on_event = on_event
        self.commit_message = commit_message

    async def propose(self, context: OptimizationContext) -> str | None:
        await self.agent.step(
            self.prompt,
            self.max_turns,
            on_event=self.on_event,
        )
        return self.commit_message


@dataclass(frozen=True)
class OptimizationRun:
    baseline: EvaluationRecord
    evaluations: tuple[EvaluationRecord, ...]
    best: EvaluationRecord | None


@dataclass
class ProgramPolicy:
    """Single-producer optimization policy over generic evaluation primitives."""

    workspace: Workspace
    engine: EvaluationEngine
    backend_id: str
    evaluation_set: EvaluationSet
    objective: ObjectiveSpec
    optimizer: CandidateProducer | None = None
    parameters: dict[str, JsonValue] = field(default_factory=dict)
    limits: EvaluationLimits = field(default_factory=EvaluationLimits)
    seed: int | None = None
    max_candidates: int = 1
    base_version: str | None = None

    def best_evaluation(
        self,
        *,
        exclude_candidate: Candidate | tuple[str, str] | None = None,
    ) -> EvaluationRecord | None:
        return self.engine.database.get_best(
            self.objective,
            backend_ids={self.backend_id},
            evaluation_sets=[self.evaluation_set],
            exclude_candidate=exclude_candidate,
        )

    def _candidate(
        self,
        commit: str,
        *,
        parent_commit: str | None = None,
        message: str | None = None,
    ) -> Candidate:
        existing = self.engine.database.candidates.get((self.workspace.name, commit))
        if existing is not None:
            return existing
        return Candidate(
            commit=commit,
            repo_name=self.workspace.name,
            parent_commit=parent_commit,
            message=message,
            created_at=datetime.now(UTC),
        )

    async def evaluate_candidate(
        self,
        commit: str,
        *,
        parent_commit: str | None = None,
        message: str | None = None,
    ) -> EvaluationRecord:
        request = EvaluationRequest(
            candidate=self._candidate(
                commit,
                parent_commit=parent_commit,
                message=message,
            ),
            evaluation_set=self.evaluation_set,
            parameters=self.parameters,
            limits=self.limits,
            seed=self.seed,
        )
        return await self.engine.evaluate_record(
            backend_id=self.backend_id,
            request=request,
            objective_spec=self.objective,
        )

    async def run(self, *, skip_initial_evaluation: bool = False) -> OptimizationRun:
        if self.max_candidates < 0:
            raise ValueError("max_candidates must be non-negative")
        if isinstance(self.optimizer, CommandCandidateProducer):
            optimizer_root = Path(self.optimizer.config.root).resolve()
            target_root = Path(self.workspace.project_path).resolve()
            if optimizer_root == target_root or optimizer_root.is_relative_to(target_root):
                raise ValueError(
                    "optimizer configuration must live outside the editable target"
                )
        base_version = self.base_version or await self.workspace.current_version()
        existing_baseline = [
            record
            for record in self.engine.database.evaluations.values()
            if record.request.candidate.commit == base_version
            and record.backend_id == self.backend_id
            and record.request.evaluation_set == self.evaluation_set
            and record.objective_spec == self.objective
        ]
        if skip_initial_evaluation:
            if not existing_baseline:
                raise ValueError(
                    "skip_initial_evaluation requires an existing compatible baseline"
                )
            baseline = existing_baseline[-1]
        else:
            baseline = await self.evaluate_candidate(base_version)

        evaluations: list[EvaluationRecord] = [baseline]
        if self.optimizer is None:
            return OptimizationRun(
                baseline=baseline,
                evaluations=tuple(evaluations),
                best=self.best_evaluation(),
            )

        previous_version = await self.workspace.current_version()
        for iteration in range(self.max_candidates):
            best = self.best_evaluation()
            message = await self.optimizer.propose(
                OptimizationContext(
                    workspace=self.workspace,
                    iteration=iteration,
                    baseline=baseline,
                    evaluations=tuple(evaluations),
                    best=best,
                )
            )
            if message is None:
                break
            if await self.workspace.is_dirty():
                commit = await self.workspace.save(message)
            else:
                commit = await self.workspace.current_version()
            if commit == previous_version:
                break
            record = await self.evaluate_candidate(
                commit,
                parent_commit=previous_version,
                message=message,
            )
            evaluations.append(record)
            previous_version = commit

        return OptimizationRun(
            baseline=baseline,
            evaluations=tuple(evaluations),
            best=self.best_evaluation(),
        )
