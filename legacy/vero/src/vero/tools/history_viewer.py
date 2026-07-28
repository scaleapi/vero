from __future__ import annotations

from dataclasses import dataclass, field

from vero.tools.utils import is_tool
from vero.workspace import Workspace


@dataclass
class HistoryViewer:
    """Inspect version history, including viewing the current version and analyzing diffs."""

    exclude_tools: list[str] = field(default_factory=list)

    # Runtime fields — set during bind()
    workspace: Workspace | None = None
    base_version: str | None = None

    def bind(self, session) -> None:
        self.workspace = session.workspace
        self.base_version = session.base_version

    @is_tool
    async def view_diff(
        self,
        from_version: str | None = None,
        to_version: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        """
        View a diff between two versions.

        Args:
            from_version: The version to compare from (defaults to the base version)
            to_version: The version to compare to (defaults to current version)
            offset: The number of characters to skip (defaults to 0)
            limit: The number of characters to return (defaults to None)

        Returns:
            The diff as text
        """
        from_version = from_version or self.base_version
        to_version = to_version or await self.workspace.current_version()
        diff = await self.workspace.diff(from_version, to_version)

        if offset > 0:
            diff = diff[offset:]
        if limit is not None:
            diff = diff[:limit]
        return diff

    @is_tool
    async def get_current_version(self) -> str:
        """Get the current version of the workspace."""
        return f"The current version is {await self.workspace.current_version()}"

    @is_tool
    async def get_base_version(self) -> str:
        """Get the base version of the workspace."""
        return f"The base version is {self.base_version}"

    @is_tool
    async def view_log(
        self,
        max_count: int = 10,
        since_version: str | None = None,
    ) -> str:
        """
        View the version history log.

        Args:
            max_count: Maximum number of versions to show (defaults to 10)
            since_version: Show versions after this one (defaults to None, showing from current)

        Returns:
            The version log as text
        """
        return await self.workspace.log(max_count=max_count, since_version=since_version)
