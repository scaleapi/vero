"""Provider-neutral coding-agent contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from vero.optimization import (
    CandidateEvaluationGateway,
    CandidateProposal,
    OptimizationContext,
)
from vero.runtime import ArtifactStore
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
    optimization: OptimizationContext
    evaluation: CandidateEvaluationGateway
    artifacts: ArtifactStore | None = None

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
    def instructions(self) -> str | None:
        return self.proposal.instruction

    @property
    def base_version(self) -> str:
        parent_id = self.proposal.parent_id
        if parent_id is not None:
            parent = self.optimization.candidates.get(parent_id)
            if parent is not None:
                return parent.version
        return self.optimization.baseline.request.candidate.version


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
