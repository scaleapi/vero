"""Normalized agent event types.

These are the canonical event shapes that agent backends produce and
logging/callbacks consume. No SDK-specific types should leak beyond
the agent's serialize_event boundary.
"""

from __future__ import annotations

from typing import TypedDict


class MessageEvent(TypedDict):
    kind: str  # "message"
    text: str


class ThinkingEvent(TypedDict):
    kind: str  # "thinking"
    text: str


class ToolCallEvent(TypedDict):
    kind: str  # "tool_call"
    name: str
    args: str


class ToolResultEvent(TypedDict):
    kind: str  # "tool_result"
    name: str
    output: str
    is_error: bool


class SystemEvent(TypedDict):
    kind: str  # "system"
    text: str


class ResultEvent(TypedDict):
    kind: str  # "result"
    text: str


AgentEvent = MessageEvent | ThinkingEvent | ToolCallEvent | ToolResultEvent | SystemEvent | ResultEvent
