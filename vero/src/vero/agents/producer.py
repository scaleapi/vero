"""Adapt a coding agent into the optimization candidate-producer protocol."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable

from vero.agents.protocol import AgentContext, AgentRequirements, CodingAgent
from vero.optimization import (
    CandidateChange,
    CandidateEvaluationGateway,
    CandidateProposal,
    OptimizationContext,
)
from vero.runtime import ArtifactStore
from vero.utils.general import recursively_serialize
from vero.workspace import Workspace

logger = logging.getLogger(__name__)


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

    def _persist_artifacts(
        self,
        proposal: CandidateProposal,
        *,
        state: Any,
        trace: Any,
    ) -> None:
        if self.artifacts is None:
            return
        digest = hashlib.sha256(proposal.id.encode()).hexdigest()[:16]
        if state is not None:
            serialized_state = recursively_serialize(state)
            self.artifacts.write_json(f"agents/{digest}/state.json", serialized_state)
            self.artifacts.write_json(
                self._producer_state_path(proposal.producer_id),
                serialized_state,
            )
        if trace is not None:
            self.artifacts.write_json(
                f"agents/{digest}/trace.json",
                recursively_serialize(trace),
            )

    def _persist_failed_run(
        self,
        proposal: CandidateProposal,
        error: BaseException,
    ) -> None:
        if self.artifacts is None:
            return
        try:
            serialize_state = getattr(self.agent, "serialize_state", None)
            serialize_trace = getattr(self.agent, "serialize_trace", None)
            state = serialize_state() if callable(serialize_state) else None
            trace = serialize_trace() if callable(serialize_trace) else None
            self._persist_artifacts(proposal, state=state, trace=trace)
            digest = hashlib.sha256(proposal.id.encode()).hexdigest()[:16]
            self.artifacts.write_json(
                f"agents/{digest}/failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            )
        except Exception:
            logger.exception("Failed to persist coding-agent failure artifacts")

    async def produce(
        self,
        *,
        proposal: CandidateProposal,
        context: OptimizationContext,
        workspace: Workspace,
        evaluation: CandidateEvaluationGateway,
    ) -> CandidateChange | None:
        self.validate_workspace(workspace)
        parent = (
            context.candidates.get(proposal.parent_id)
            if proposal.parent_id is not None
            else None
        )
        if parent is None:
            parent = context.baseline.request.candidate
        try:
            result = await self.agent.run(
                context=AgentContext(
                    session_id=context.session_id,
                    workspace=workspace,
                    proposal=proposal,
                    parent=parent,
                    evaluation=evaluation,
                ),
                prompt=proposal.instruction or self.prompt,
                max_turns=self.max_turns,
                on_event=self.on_event,
            )
        except BaseException as error:
            self._persist_failed_run(proposal, error)
            raise
        if result is None:
            return None

        self._persist_artifacts(
            proposal,
            state=result.state,
            trace=result.trace,
        )
        return CandidateChange(
            description=result.description,
            metadata={"agent": type(self.agent).__name__, **result.metadata},
        )
