from __future__ import annotations

import logging
from dataclasses import dataclass, field

from vero.tools.utils import is_tool
from vero.workspace import Workspace

logger = logging.getLogger(__name__)


@dataclass
class VersionControl:
    """Version control operations — restore to previous versions with safety checks."""

    exclude_tools: list[str] = field(default_factory=list)

    # Runtime fields — set during bind()
    workspace: Workspace | None = None

    def bind(self, session) -> None:
        self.workspace = session.workspace

    async def _is_safe_to_restore(self, version_id: str) -> bool:
        """Check if restoring to this version is safe (it's an ancestor of the current version)."""
        current = await self.workspace.current_version()
        return version_id == current or await self.workspace.is_ancestor(version_id, current)

    @is_tool
    async def restore_to_version(
        self, version_id: str, message: str | None = None
    ) -> str:
        """
        Restore the workspace to match a previous version, preserving full history.

        Creates a new version with the file state from the target version.
        This is safe because it maintains the full history, making it easy
        to see what was tried and undo the restore if needed.

        Args:
            version_id: The version to restore the file state from
            message: Optional message for the new version (defaults to "Restore to <version_id>")

        Raises:
            ValueError: If the version is not in history or there are unsaved changes
        """
        if not await self._is_safe_to_restore(version_id):
            raise ValueError(
                f"Version {version_id} is not an ancestor of the current version"
            )

        if await self.workspace.is_dirty():
            await self.workspace.save("Auto-save before restore")

        new_version = await self.workspace.restore(version_id, message)
        logger.info(f"Restored to {version_id} with new version {new_version}")
        return f"Restored workspace to state at {version_id}. New version: {new_version}"
