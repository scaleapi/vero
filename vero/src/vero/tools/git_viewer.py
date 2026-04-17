from __future__ import annotations

from dataclasses import dataclass

from vero.sandbox import Sandbox
from vero.tools.history_viewer import HistoryViewer
from vero.tools.utils import is_tool
from vero.workspace.git import GitWorkspace


@dataclass
class GitViewer(HistoryViewer):
    """Git-specific history viewer with additional constructors and git-only features."""

    @classmethod
    async def from_path_and_commit(
        cls, project_path: str, base_commit: str, sandbox: Sandbox | None = None
    ) -> GitViewer:
        """Create a GitViewer from a project path and base commit.

        Args:
            project_path: Path to the project (string).
            base_commit: The base commit hash.
            sandbox: Sandbox to use. If None, creates one via LocalSandbox.create().
        """
        if sandbox is None:
            workspace = await GitWorkspace.create(project_path)
        else:
            workspace = await GitWorkspace.from_path(sandbox, project_path)
        instance = cls()
        instance.workspace = workspace
        instance.base_version = base_commit
        return instance

    @is_tool
    async def git_log(
        self,
        max_count: int = 10,
        since_commit: str | None = None,
        oneline: bool = False,
    ) -> str:
        """
        View the git commit log.

        Args:
            max_count: Maximum number of commits to show (defaults to 10)
            since_commit: Show commits after this commit (defaults to None, showing from HEAD)
            oneline: If True, show condensed one-line format (defaults to False)

        Returns:
            The git log output as text
        """
        if not isinstance(self.workspace, GitWorkspace):
            return await self.workspace.log(max_count=max_count, since_version=since_commit)

        args = ["log", f"-n{max_count}"]

        if since_commit:
            args.append(f"{since_commit}..HEAD")

        if oneline:
            args.append("--oneline")
        else:
            args.append("--pretty=format:%h - %s (%cr) <%an>")

        return await self.workspace._git(*args)
