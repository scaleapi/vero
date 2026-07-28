from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from vero.agents.events import AgentEvent

if TYPE_CHECKING:
    from vero.session import BestVersion, Session


class BaseAgent(ABC):
    """Abstract base class for agent backends that execute optimization steps."""

    state: Any = field(default=None, repr=False)
    """Agent state for resumption."""

    trace: Any = field(default=None, repr=False)
    """A trace of the agent's execution."""

    @abstractmethod
    def init(self, session: Session) -> None:
        """Initialize the agent with a Session context."""
        ...

    @abstractmethod
    async def step(
        self,
        input: Any,
        max_turns: int = 200,
        on_event: Callable[[Any], Any] | None = None,
        **kwargs,
    ) -> Any:
        """Execute one or more tool-call turns.

        Args:
            input: Input to the agent.
            max_turns: Maximum number of turns.
            on_event: Optional callback fired with each raw SDK event during streaming.
        """
        ...

    def serialize_event(self, event: Any) -> AgentEvent | None:
        """Convert a raw SDK event to a normalized AgentEvent.

        Override in subclasses to handle SDK-specific event types.
        Returns None for events that should be ignored.
        """
        return None

    @abstractmethod
    def serialize_trace(self) -> Any:
        """Serialize the execution trace (event log) to a JSON-serializable format.

        The trace is a record of what happened — for observability and debugging.
        It is NOT necessarily the same as state (see serialize_state/deserialize_state).
        """
        ...

    @abstractmethod
    def serialize_state(self) -> Any:
        """Serialize the agent state needed to resume execution.

        Returns:
            JSON-serializable state, or None if no state to save.
        """
        ...

    @abstractmethod
    def deserialize_state(self, state: Any) -> None:
        """Restore agent state from a previously serialized state.

        After calling this, the agent should be able to continue
        from where it left off (via step()).

        Args:
            state: The state returned by a previous serialize_state() call.
        """
        ...

    def usage(self) -> dict:
        """Return usage statistics for the run."""
        return {}

    def summary(self) -> dict:
        """Return extra fields to include in the wandb finish summary."""
        return {}

    def dict(self) -> dict:
        """Return extra fields for as_dict() serialization."""
        return {}

    def artifacts(self) -> dict:
        """Get any agent-specific artifacts."""
        return {}

    def save_artifacts(self, session_dir: Path) -> None:
        """Save any agent-specific artifacts to the session directory."""
        artifacts_path = session_dir / "artifacts.json"
        with open(artifacts_path, "w") as f:
            json.dump(self.artifacts(), f, indent=2, default=str)

    def get_best_version(self) -> BestVersion:
        """Extract best commit from agent output (e.g. structured output). Returns None if not available."""
        from vero.session import BestVersion

        return BestVersion()
