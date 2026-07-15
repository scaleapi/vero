"""Adapt a coding agent into the optimization candidate-producer protocol."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from vero.agents.protocol import AgentContext, CodingAgent
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

    async def produce(
        self,
        *,
        proposal: CandidateProposal,
        context: OptimizationContext,
        workspace: Workspace,
        evaluation: CandidateEvaluationGateway,
    ) -> CandidateChange | None:
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
            if result.trace is not None:
                self.artifacts.write_json(f"agents/{digest}/trace.json", result.trace)
        return CandidateChange(
            description=result.description,
            metadata={"agent": type(self.agent).__name__, **result.metadata},
        )
