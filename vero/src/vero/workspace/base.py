from __future__ import annotations

import posixpath
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, AsyncGenerator

if TYPE_CHECKING:
    from vero.sandbox import Sandbox


class Workspace(ABC):
    """Abstract version-control workspace.

    Combines an isolated working directory with snapshot/restore versioning.
    Workspace depends on a Sandbox for file and shell operations.  All
    version-control methods are async to support non-local sandbox backends.
    """

    # ── Properties ─────────────────────────────────────────────────

    @property
    @abstractmethod
    def sandbox(self) -> Sandbox:
        """The execution environment this workspace operates in."""
        ...

    @property
    @abstractmethod
    def root(self) -> str:
        """Workspace root directory (e.g. git repo root)."""
        ...

    @property
    @abstractmethod
    def project_path(self) -> str:
        """Project directory within this workspace."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this workspace."""
        ...

    # ── History ─────────────────────────────────────────────────────

    @abstractmethod
    async def current_version(self) -> str:
        """Return the current version ID (commit hash, change ID, etc.)."""
        ...

    @abstractmethod
    async def save(self, message: str = "Save") -> str:
        """Create a new version from current state. Returns the version ID."""
        ...

    @abstractmethod
    async def restore(self, version_id: str, message: str | None = None) -> str:
        """Revert to a previous version, preserving history. Returns new version ID."""
        ...

    # ── History inspection ──────────────────────────────────────────

    @abstractmethod
    async def diff(
        self, from_version: str | None = None, to_version: str | None = None
    ) -> str:
        """Diff between two versions."""
        ...

    @abstractmethod
    async def log(self, max_count: int = 10, since_version: str | None = None) -> str:
        """Version history."""
        ...

    @abstractmethod
    async def is_ancestor(self, version_a: str, version_b: str) -> bool:
        """Is version_a an ancestor of version_b?"""
        ...

    # ── Copies ──────────────────────────────────────────────────────

    @abstractmethod
    async def copy(
        self, name: str | None = None, from_version: str | None = None
    ) -> Workspace:
        """Create a persistent isolated copy of this workspace."""
        ...

    @asynccontextmanager
    async def temp_copy(
        self, from_version: str | None = None
    ) -> AsyncGenerator[Workspace, None]:
        """Temporary isolated copy, cleaned up on exit."""
        yield self  # pragma: no cover

    # ── Execution at a version ──────────────────────────────────────

    @asynccontextmanager
    async def at(self, version_id: str) -> AsyncGenerator[None, None]:
        """Temporarily switch to a version, restore on exit."""
        yield  # pragma: no cover

    # ── State ───────────────────────────────────────────────────────

    @abstractmethod
    async def is_dirty(self) -> bool:
        """Are there unsaved changes not yet captured in a version?"""
        ...

    async def destroy(self) -> None:
        """Clean up this workspace. Default: no-op."""
        pass

    # ── Path resolution ────────────────────────────────────────────

    def resolve_path(self, path: str) -> str:
        """Resolve a path relative to ``project_path``.

        Absolute paths pass through.  Relative paths (including ``"."``)
        are resolved against ``project_path``, not ``sandbox.root``.
        """
        value = PurePosixPath(str(path))
        if value.is_absolute():
            return posixpath.normpath(value.as_posix())
        root = posixpath.normpath(PurePosixPath(str(self.project_path)).as_posix())
        return posixpath.normpath(posixpath.join(root, value.as_posix()))

    def get_relative_path(self, path: str) -> str | None:
        """Get path relative to ``project_path``, or None if outside."""
        resolved = PurePosixPath(self.resolve_path(path))
        root = PurePosixPath(
            posixpath.normpath(PurePosixPath(str(self.project_path)).as_posix())
        )
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            return None
        return relative.as_posix()
