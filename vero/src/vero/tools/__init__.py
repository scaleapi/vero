from .base import ToolSet
from .bash import BashTool
from .evaluation import EvaluationTools
from .file_read import FileRead
from .file_write import FileWrite
from .git_control import GitControl
from .git_viewer import GitViewer
from .grep import Grep
from .history_viewer import HistoryViewer
from .planning import TodoList, think
from .registry import ToolDefinition, ToolRegistry, ToolSetInstance
from .version_control import VersionControl
from .sub_agent import SubAgentTool

# Register all tool classes
ToolRegistry.register(BashTool)
ToolRegistry.register(EvaluationTools)
ToolRegistry.register(FileRead)
ToolRegistry.register(FileWrite)
ToolRegistry.register(GitControl)
ToolRegistry.register(GitViewer)
ToolRegistry.register(Grep)
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
    "BashTool",
    "EvaluationTools",
    "FileWrite",
    "GitControl",
    "GitViewer",
    "HistoryViewer",
    "FileRead",
    "Grep",
    "SubAgentTool",
    "TodoList",
    "think",
    "WebFetch",
    "VersionControl",
    "WebSearch",
    # Registry
    "ToolRegistry",
    "ToolSetInstance",
    "ToolDefinition",
    "ToolSet",
]
