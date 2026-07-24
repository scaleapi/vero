"""Provider-neutral coding-agent contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from vero.candidate import Candidate
from vero.optimization import (
    CandidateEvaluationGateway,
    CandidateProposal,
)
from vero.runtime.context import AGENT_CONTEXT_DIRECTORY
from vero.workspace import Workspace


@dataclass(frozen=True)
class AgentRequirements:
    """Workspace capabilities required by a coding-agent adapter."""

    host_visible_workspace: bool = False


@dataclass(frozen=True)
class AgentContext:
    """Capabilities visible to a coding agent working on one proposal."""

    session_id: str
    workspace: Workspace
    proposal: CandidateProposal
    parent: Candidate
    evaluation: CandidateEvaluationGateway

    @property
    def project_path(self) -> Path:
        path = self.workspace.sandbox.host_path(self.workspace.project_path)
        if path is None:
            raise RuntimeError("coding agent requires a host-visible workspace path")
        return path

    @property
    def sandbox_project_path(self) -> str:
        return self.workspace.project_path

    @property
    def context_path(self) -> str:
        return str(PurePosixPath(self.workspace.project_path) / AGENT_CONTEXT_DIRECTORY)

    @property
    def relative_context_path(self) -> str:
        return AGENT_CONTEXT_DIRECTORY

    @property
    def instructions(self) -> str:
        context = (
            f"Read-only evaluation context has been placed in "
            f"`{AGENT_CONTEXT_DIRECTORY}/`. Inspect its README, tasks, prior "
            "candidates, and evaluation results when useful. Do not modify or "
            "commit that directory."
        )
        if self.proposal.instruction:
            return f"{self.proposal.instruction}\n\n{context}"
        return context

    @property
    def base_version(self) -> str:
        return self.parent.version


class AgentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = "Apply coding-agent changes"
    state: JsonValue | None = None
    trace: JsonValue | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("agent result description must not be empty")
        return value


@runtime_checkable
class CodingAgent(Protocol):
    async def run(
        self,
        *,
        context: AgentContext,
        prompt: str | None,
        max_turns: int,
        on_event: Callable[[Any], Any] | None = None,
    ) -> AgentRunResult | None: ...
