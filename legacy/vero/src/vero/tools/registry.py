"""Tool Registry for managing available tools."""

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

ToolT = TypeVar("ToolT")


@dataclass(frozen=True, slots=True)
class ToolSetInstance(Generic[ToolT]):
    """An instance of a tool set with its function tools."""

    instance: Callable | object
    tools: list[ToolT] = field(repr=False)


@dataclass
class ToolDefinition:
    """Definition of a tool in the registry."""

    tool_class: type | Callable
    description: str
    is_callable: bool = False

    @property
    def name(self) -> str:
        """The original class/function name."""
        return self.tool_class.__name__


class ToolRegistry:
    """Registry for tools available to agents.

    Uses the class/callable itself as the key.

    Example:
        >>> ToolRegistry.register(BashTool)
        >>> ToolRegistry.get(BashTool)
        ToolDefinition(tool_class=BashTool, ...)
    """

    _tools: dict[type | Callable, ToolDefinition] = {}

    @classmethod
    def register(cls, tool_class: type, description: str | None = None) -> type:
        """Register a tool class. Returns the class itself as the key."""
        desc = description or (tool_class.__doc__ or "").strip()
        cls._tools[tool_class] = ToolDefinition(
            tool_class=tool_class, description=desc, is_callable=False
        )
        return tool_class

    @classmethod
    def register_callable(cls, func: Callable, description: str | None = None) -> Callable:
        """Register a callable (function) as a tool. Returns the callable itself."""
        desc = description or (func.__doc__ or "").split("\n")[0].strip()
        cls._tools[func] = ToolDefinition(
            tool_class=func, description=desc, is_callable=True
        )
        return func

    @classmethod
    def get(cls, key: type | Callable) -> ToolDefinition:
        """Get a tool definition by class/callable."""
        if key not in cls._tools:
            raise KeyError(
                f"Tool '{key}' not found in registry. Available: {[t.__name__ for t in cls._tools.keys()]}"
            )
        return cls._tools[key]

    @classmethod
    def describe(cls, key: type | Callable) -> str:
        """Describe a tool by class/callable."""
        return cls.get(key).description

    @classmethod
    def exists(cls, key: type | Callable) -> bool:
        """Check if a tool exists in the registry."""
        return key in cls._tools

    @classmethod
    def all_keys(cls) -> set[type | Callable]:
        """Get all registered tool keys."""
        return set(cls._tools.keys())

    @classmethod
    def items(cls) -> list[tuple[type | Callable, ToolDefinition]]:
        """Get all registered tools as (key, definition) pairs."""
        return list(cls._tools.items())
