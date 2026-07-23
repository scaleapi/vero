from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from vero.exceptions import CommitNotInBranchHistory
from vero.tools.utils import is_tool
from vero.tools.version_control import VersionControl
from vero.workspace.git import GitWorkspace

logger = logging.getLogger(__name__)


@dataclass
class GitControl(VersionControl):
    """Git-specific version control with branch safety checks."""

    # Git-specific runtime fields
    allowed_branch: str | None = None
    _lock: asyncio.Lock | None = None

    def bind(self, session) -> None:
        super().bind(session)
        if self._lock is None:
            self._lock = asyncio.Lock()
        # Branch detection deferred to first tool call since workspace methods are async

    async def _ensure_allowed_branch(self) -> None:
        """Lazily detect the allowed branch on first use."""
        if self.allowed_branch is not None:
            return
        if isinstance(self.workspace, GitWorkspace):
            self.allowed_branch = await self.workspace.current_branch()
            if self.allowed_branch is None:
                raise ValueError(
                    "GitControl requires an allowed_branch but the repository is in a detached HEAD state."
                )

    async def _is_commit_in_branch_history(self, commit_hash: str) -> bool:
        """Check if a commit is in the history of the allowed branch."""
        if not isinstance(self.workspace, GitWorkspace):
            return await self._is_safe_to_restore(commit_hash)

        await self._ensure_allowed_branch()
        try:
            branch_head = await self.workspace.get_head_commit(self.allowed_branch)
            return commit_hash == branch_head or await self.workspace.is_ancestor(commit_hash, branch_head)
        except Exception:
            return False

    @is_tool
    async def restore_to_commit(
        self, commit_hash: str, message: str | None = None
    ) -> str:
        """
        Restore the working tree to match a previous commit, preserving full history.

        Creates a new commit with the file state from the target commit.
        This is safer than a hard reset because it maintains the full commit history,
        making it easy to see what was tried and undo the restore if needed.

        Args:
            commit_hash: The commit hash to restore the file state from
            message: Optional commit message (defaults to "Restore to <commit_hash>")

        Raises:
            CommitNotInBranchHistory: If the commit is not in the allowed branch history
            ValueError: If there are uncommitted changes in the repository
        """
        await self._ensure_allowed_branch()

        async with self._lock:
            if not await self._is_commit_in_branch_history(commit_hash):
                raise CommitNotInBranchHistory(
                    f"Commit {commit_hash} is not in the history of branch '{self.allowed_branch}'"
                )

            if await self.workspace.is_dirty():
                await self.workspace.save("Auto-save before restore")

            # Git-specific: check we're on the allowed branch
            if isinstance(self.workspace, GitWorkspace):
                current_branch = await self.workspace.current_branch()
                if current_branch != self.allowed_branch:
                    raise ValueError(
                        f"Cannot restore: current branch {current_branch} is not the allowed branch {self.allowed_branch}"
                    )

            new_commit = await self.workspace.restore(commit_hash, message)
            logger.info(f"Restored to {commit_hash} with new commit {new_commit}")
            return f"Restored working tree to state at {commit_hash}. New commit: {new_commit}"
