"""Adapt a coding agent into the optimization candidate-producer protocol."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from vero.agents.protocol import AgentContext, AgentRequirements, CodingAgent
from vero.optimization import (
    CandidateChange,
    CandidateEvaluationGateway,
    CandidateProposal,
    OptimizationContext,
)
from vero.runtime import ArtifactStore
from vero.workspace import Workspace


class AgentCandidateProducer:
    def __init__(
        self,
        agent: CodingAgent,
        *,
        prompt: str | None = None,
        max_turns: int = 200,
        artifacts: ArtifactStore | None = None,
        on_event: Callable[[Any], Any] | None = None,
    ):
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.agent = agent
        self.prompt = prompt
        self.max_turns = max_turns
        self.artifacts = artifacts
        self.on_event = on_event

    def validate_workspace(self, workspace: Workspace) -> None:
        requirements = getattr(self.agent, "requirements", AgentRequirements())
        if (
            requirements.host_visible_workspace
            and workspace.sandbox.host_path(workspace.project_path) is None
        ):
            raise ValueError(
                f"{type(self.agent).__name__} requires a host-visible workspace; "
                f"sandbox {type(workspace.sandbox).__name__} does not expose one"
            )

    def bind_artifacts(
        self,
        artifacts: ArtifactStore,
        *,
        producer_id: str = "default",
        restore: bool = True,
    ) -> None:
        """Attach durable storage and restore the latest supported agent state."""

        self.artifacts = artifacts
        if not restore:
            return
        state_path = self._producer_state_path(producer_id)
        if not artifacts.path(state_path).exists():
            return
        deserialize = getattr(self.agent, "deserialize_state", None)
        if callable(deserialize):
            deserialize(artifacts.read_json(state_path))

    @staticmethod
    def _producer_state_path(producer_id: str) -> str:
        digest = hashlib.sha256(producer_id.encode()).hexdigest()[:16]
        return f"agents/producers/{digest}/state.json"

    async def produce(
        self,
        *,
        proposal: CandidateProposal,
        context: OptimizationContext,
        workspace: Workspace,
        evaluation: CandidateEvaluationGateway,
    ) -> CandidateChange | None:
        self.validate_workspace(workspace)
        result = await self.agent.run(
            context=AgentContext(
                session_id=context.session_id,
                workspace=workspace,
                proposal=proposal,
                optimization=context,
                evaluation=evaluation,
                artifacts=self.artifacts,
            ),
            prompt=proposal.instruction or self.prompt,
            max_turns=self.max_turns,
            on_event=self.on_event,
        )
        if result is None:
            return None

        if self.artifacts is not None:
            digest = hashlib.sha256(proposal.id.encode()).hexdigest()[:16]
            if result.state is not None:
                self.artifacts.write_json(f"agents/{digest}/state.json", result.state)
                self.artifacts.write_json(
                    self._producer_state_path(proposal.producer_id), result.state
                )
            if result.trace is not None:
                self.artifacts.write_json(f"agents/{digest}/trace.json", result.trace)
        return CandidateChange(
            description=result.description,
            metadata={"agent": type(self.agent).__name__, **result.metadata},
        )
