from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, AsyncGenerator

from vero.filesystem import AccessRule, AccessType, WorkspaceAccessPolicy

if TYPE_CHECKING:
    from vero.sandbox import Sandbox


class Workspace(ABC):
    """Abstract version-control workspace.

    Combines an isolated working directory with snapshot/restore versioning
    and access control.  Workspace depends on a Sandbox for file and shell
    operations.  All version-control methods are async to support non-local
    sandbox backends.
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

    # ── Access control ─────────────────────────────────────────────

    @property
    def accesses(self) -> list[AccessRule]:
        """Current access rules."""
        return self._fs.accesses

    @property
    def default_access(self) -> AccessType:
        """Default access when no rules match."""
        return self._fs.default_access

    def set_access(
        self, accesses: list[AccessRule], default_access: AccessType = AccessType.WRITE
    ) -> None:
        """Configure access rules for this workspace.

        Rules are glob patterns relative to ``project_path``, matching
        ``.veroaccess`` file format.  Called by ``Policy`` after workspace
        creation; non-Policy callers (evaluator, trace analysis) leave
        access fully open (the default set in the constructor).
        """
        self._fs = WorkspaceAccessPolicy(
            root=self.project_path,
            accesses=accesses,
            default_access=default_access,
        )

    def resolve_path(self, path: str) -> str:
        """Resolve a path relative to ``project_path``.

        Absolute paths pass through.  Relative paths (including ``"."``)
        are resolved against ``project_path``, not ``sandbox.root``.
        """
        return self._fs.resolve_path(path)

    def get_relative_path(self, path: str) -> str | None:
        """Get path relative to ``project_path``, or None if outside."""
        return self._fs.get_relative_path(path)

    def can_read(self, path: str) -> bool:
        """Check if path is readable under current access rules."""
        return self._fs.can_read(path)

    def can_write(self, path: str) -> bool:
        """Check if path is writable under current access rules."""
        return self._fs.can_write(path)

    def validate_read(self, path: str) -> str:
        """Resolve path and check read access. Raises AccessDeniedError."""
        return self._fs.validate_read(path)

    def validate_write(self, path: str) -> str:
        """Resolve path and check write access. Raises AccessDeniedError."""
        return self._fs.validate_write(path)

    async def _canonical_access_policy(self) -> WorkspaceAccessPolicy:
        """Return the current policy rooted at the sandbox-canonical project path.

        Sandbox paths can have equivalent spellings (for example, macOS maps
        ``/tmp`` to ``/private/tmp``).  Canonical paths must be checked against
        an equally canonical root or valid paths appear to leave the workspace.
        The original policy is still checked first by the callers below, so a
        symlink cannot be used to enter the workspace from an unauthorized path.
        """

        canonical_root = await self.sandbox.canonicalize(self._fs.root)
        return WorkspaceAccessPolicy(
            root=canonical_root,
            accesses=self._fs.accesses,
            default_access=self._fs.default_access,
        )

    async def validate_read_path(self, path: str) -> str:
        """Validate read access after resolving sandbox symlinks."""

        resolved = self._fs.validate_read(path)
        canonical = await self.sandbox.canonicalize(resolved)
        canonical_policy = await self._canonical_access_policy()
        return canonical_policy.validate_read(canonical)

    async def validate_write_path(self, path: str) -> str:
        """Validate write access after resolving existing sandbox ancestors."""

        resolved = self._fs.validate_write(path)
        canonical_policy = await self._canonical_access_policy()
        if await self.sandbox.exists(resolved):
            canonical = await self.sandbox.canonicalize(resolved)
            return canonical_policy.validate_write(canonical)

        current = PurePosixPath(resolved)
        missing: list[str] = []
        while not await self.sandbox.exists(current.as_posix()):
            if current.parent == current:
                raise FileNotFoundError(resolved)
            missing.append(current.name)
            current = current.parent
        canonical = PurePosixPath(await self.sandbox.canonicalize(current.as_posix()))
        for component in reversed(missing):
            canonical /= component
        return canonical_policy.validate_write(canonical.as_posix())

    def get_access(self, path: str) -> AccessType:
        """Get the access level for a path."""
        return self._fs.get_access(path)
