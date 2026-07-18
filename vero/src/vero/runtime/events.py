"""Session-scoped runtime events and sinks."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

logger = logging.getLogger(__name__)


class RuntimeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    kind: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id", "session_id", "kind")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("event identity must not be empty")
        return value

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event timestamps must be timezone-aware")
        return value.astimezone(UTC)


class EventSink(Protocol):
    def __call__(self, event: RuntimeEvent) -> object: ...


class EventBus:
    """Publish runtime events without coupling execution to observability sinks."""

    def __init__(self, sinks: list[EventSink] | None = None):
        self.sinks: list[EventSink] = list(sinks or [])

    async def emit(
        self,
        *,
        session_id: str,
        kind: str,
        payload: dict[str, JsonValue] | None = None,
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            session_id=session_id,
            kind=kind,
            payload=payload or {},
        )
        for sink in self.sinks:
            try:
                result = sink(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Runtime event sink failed for %s", event.id)
        return event


def agent_event_emitter(
    bus: EventBus,
    session_id: str,
    agent: object,
    *,
    kind: str = "agent",
) -> Callable[[object], Awaitable[None]] | None:
    """Build an ``on_event`` callback that publishes agent activity to ``bus``.

    The coding agent streams SDK-specific stream events; the agent normalizes
    them via ``serialize_event`` into ``AgentEvent`` dicts (message / thinking /
    tool_call / tool_result). This adapts that normalized stream onto the runtime
    ``EventBus`` so tool calls, reasoning, and messages land in ``events.jsonl``
    and any registered sink (e.g. W&B) live — the native runner's introspection
    advantage over opaque environments. Returns ``None`` if the agent cannot
    normalize its events.
    """
    serialize = getattr(agent, "serialize_event", None)
    if not callable(serialize):
        return None

    async def _emit(raw_event: object) -> None:
        normalized = serialize(raw_event)
        if normalized:
            await bus.emit(
                session_id=session_id, kind=kind, payload=dict(normalized)
            )

    return _emit


class JsonlEventSink:
    """Append canonical runtime events to a session JSONL file."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()

    async def __call__(self, event: RuntimeEvent) -> None:
        line = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
