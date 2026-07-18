from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vero.agents.protocol import AgentContext


@runtime_checkable
class ToolSet(Protocol):
    """Protocol for tool sets that can be bound to an agent context.

    ToolSets group related tool methods (decorated with @is_tool).
    They are pre-created instances that self-wire to policy resources
    via bind().
    """

    exclude_tools: list[str]

    def bind(self, context: AgentContext) -> None: ...
