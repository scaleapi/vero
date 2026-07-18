from __future__ import annotations

import pytest

from vero.runtime.events import EventBus, RuntimeEvent, agent_event_emitter


class _FakeAgent:
    """Normalizes raw stream events the way VeroAgent.serialize_event does."""

    def serialize_event(self, raw):
        if raw == "noise":
            return None  # noise events are dropped, not published
        return {"kind": "tool_call", "name": "shell", "args": raw}


@pytest.mark.asyncio
async def test_agent_event_emitter_publishes_normalized_events_to_bus():
    captured: list[RuntimeEvent] = []
    bus = EventBus([captured.append])
    emit = agent_event_emitter(bus, "sess-1", _FakeAgent())
    assert emit is not None

    await emit("ls -la")
    await emit("noise")  # serialize_event -> None, so nothing is published

    assert len(captured) == 1
    event = captured[0]
    assert isinstance(event, RuntimeEvent)
    assert event.session_id == "sess-1"
    assert event.kind == "agent"
    assert event.payload == {"kind": "tool_call", "name": "shell", "args": "ls -la"}


@pytest.mark.asyncio
async def test_agent_event_emitter_supports_async_sinks():
    captured: list[RuntimeEvent] = []

    async def async_sink(event: RuntimeEvent) -> None:
        captured.append(event)

    bus = EventBus([async_sink])
    emit = agent_event_emitter(bus, "s", _FakeAgent())
    assert emit is not None
    await emit("pwd")

    assert [e.payload["args"] for e in captured] == ["pwd"]


def test_agent_event_emitter_is_none_without_serialize_event():
    bus = EventBus()
    assert agent_event_emitter(bus, "s", object()) is None
