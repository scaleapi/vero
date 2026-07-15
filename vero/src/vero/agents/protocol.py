"""Provider-neutral coding-agent contract."""

from __future__ import annotations

from dataclasses import dataclass
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
class AgentContext:
    """Capabilities visible to a coding agent working on one proposal."""

    session_id: str
    workspace: Workspace
    proposal: CandidateProposal
    optimization: OptimizationContext
    evaluation: CandidateEvaluationGateway
    artifacts: ArtifactStore | None = None


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
