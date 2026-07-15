from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Coroutine, Generic, Protocol, TypeVar, runtime_checkable

from vero.exceptions import NoFilesChangedError
from vero.sandbox import Sandbox
from vero.workspace import Workspace
from vero.workspace.git import GitWorkspace

if TYPE_CHECKING:
    from vero.agents.protocol import AgentContext

logger = logging.getLogger(__name__)

T = TypeVar("T")


@runtime_checkable
class ToolSet(Protocol):
    """Protocol for tool sets that can be bound to an agent context.

    ToolSets group related tool methods (decorated with @is_tool).
    They are pre-created instances that self-wire to policy resources
    via bind().
    """

    exclude_tools: list[str]

    def bind(self, context: AgentContext) -> None: ...


@dataclass(frozen=True)
class FileSystemWriteResult(Generic[T]):
    commit: str
    result: T


@dataclass
class FileSystemWriteBase(ABC):
    """Base class for write tools.

    Config fields (set at construction):
        content_char_limit, etc. (defined by subclasses)

    Runtime fields (set during bind):
        sandbox: The sandbox for access control.
        workspace: The workspace for version control.
    """

    exclude_tools: list[str] = field(default_factory=list)

    # Runtime fields — set during bind()
    sandbox: Sandbox | None = None
    workspace: Workspace | None = None

    def bind(self, context: AgentContext) -> None:
        self.sandbox = context.workspace.sandbox
        self.workspace = context.workspace

    async def _is_file_tracked(self, path: str) -> bool:
        """Check if a file is tracked by the workspace. Git-specific for now."""
        if not isinstance(self.workspace, GitWorkspace):
            return True  # Non-git backends: assume all files are tracked

        resolved = self.workspace.resolve_path(path)
        try:
            # Compute path relative to workspace root via string prefix
            root = self.workspace.root.rstrip("/") + "/"
            rel_path = resolved[len(root):] if resolved.startswith(root) else resolved
            result = await self.workspace._git("ls-files", resolved)
            return rel_path in result.strip().splitlines()
        except RuntimeError:
            return False

    async def run_and_commit(
        self, coro: Coroutine[None, None, T], commit_message: str = "Commit changes"
    ) -> FileSystemWriteResult[T]:
        """Run a coroutine and save the resulting changes as a new version."""

        result = await coro
        if not await self.workspace.is_dirty():
            raise NoFilesChangedError("No files changed! Nothing to commit.")
        new_version = await self.workspace.save(commit_message)
        logger.info(f"{self.__class__.__name__} saved version {new_version}")
        return FileSystemWriteResult(commit=new_version, result=result)
