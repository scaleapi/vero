from .base import ToolSet
from .evaluation import EvaluationTools
from .planning import TodoList, think
from .registry import ToolDefinition, ToolRegistry, ToolSetInstance
from .sub_agent import SubAgentTool

# Register all tool classes
ToolRegistry.register(EvaluationTools)
ToolRegistry.register(SubAgentTool)
ToolRegistry.register(TodoList)
ToolRegistry.register_callable(think)


def __getattr__(name: str):
    if name in {"WebFetch", "WebSearch"}:
        from .web import WebFetch, WebSearch

        return {"WebFetch": WebFetch, "WebSearch": WebSearch}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Tool classes (these ARE the keys now)
    "EvaluationTools",
    "SubAgentTool",
    "TodoList",
    "think",
    "WebFetch",
    "WebSearch",
    # Registry
    "ToolRegistry",
    "ToolSetInstance",
    "ToolDefinition",
    "ToolSet",
]
