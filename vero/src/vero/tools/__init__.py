from .base import ToolSet
from .bash import BashTool
from .context_store import ContextStore, IndexedArtifact
from .dataset_viewer import DatasetViewer
from .experiment_runner import (
    ExperimentRunnerTool,
    SplitBudget,
)
from .experiment_viewer import ExperimentViewer
from .file_read import FileRead
from .file_write import FileWrite
from .git_control import GitControl
from .git_viewer import GitViewer
from .grep import Grep
from .history_viewer import HistoryViewer
from .planning import TodoList, think
from .registry import ToolDefinition, ToolRegistry, ToolSetInstance
from .resource_control import ResourceControl
from .version_control import VersionControl
from .sub_agent import SubAgentTool
from .web import WebFetch, WebSearch

# Register all tool classes
ToolRegistry.register(BashTool)
ToolRegistry.register(ContextStore)
ToolRegistry.register(DatasetViewer)
ToolRegistry.register(ExperimentRunnerTool)
ToolRegistry.register(ExperimentViewer)
ToolRegistry.register(FileRead)
ToolRegistry.register(FileWrite)
ToolRegistry.register(GitControl)
ToolRegistry.register(GitViewer)
ToolRegistry.register(Grep)
ToolRegistry.register(ResourceControl)
ToolRegistry.register(SubAgentTool)
ToolRegistry.register(TodoList)
ToolRegistry.register(WebFetch)
ToolRegistry.register(WebSearch)
ToolRegistry.register_callable(think)


__all__ = [
    # Tool classes (these ARE the keys now)
    "BashTool",
    "ContextStore",
    "DatasetViewer",
    "ExperimentRunnerTool",
    "ExperimentViewer",
    "FileWrite",
    "GitControl",
    "GitViewer",
    "HistoryViewer",
    "FileRead",
    "Grep",
    "IndexedArtifact",
    "ResourceControl",
    "SplitBudget",
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
