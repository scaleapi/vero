from dataclasses import dataclass, field
from enum import StrEnum
import posixpath
from pathlib import Path, PurePosixPath

from vero.exceptions import AccessDeniedError


class AccessType(StrEnum):
    """Types of filesystem access.

    Access levels are hierarchical:
    - EXCLUDE: No access at all
    - READ: Read-only access
    - WRITE: Read and write access (implies read)
    """

    EXCLUDE = "exclude"
    READ = "read"
    WRITE = "write"

    def can_read(self) -> bool:
        """Check if this access level allows reading."""
        return self in (AccessType.READ, AccessType.WRITE)

    def can_write(self) -> bool:
        """Check if this access level allows writing."""
        return self == AccessType.WRITE


@dataclass(frozen=True)
class AccessRule:
    """A single access rule combining a type with a glob pattern.

    Attributes:
        access_type: The type of access (exclude, read, or write).
        pattern: A glob pattern to match against file paths.
                 Patterns are matched against paths relative to the root directory.
                 Examples: "*.py", "src/**/*.py", "**/__pycache__/**"
    """

    access_type: AccessType
    pattern: str

    def __post_init__(self):
        object.__setattr__(self, "access_type", AccessType(self.access_type))

    def matches(self, relative_path: str) -> bool:
        """Check if this rule matches a given relative path.

        Args:
            relative_path: Path relative to the filesystem root.

        Returns:
            True if the pattern matches the path.
        """
        # we use wcmatch instead of pathlib because pathlib.Path.full_match() is only supported for Python 3.13+
        from wcmatch import pathlib as wcpathlib

        # Strip trailing slashes from pattern to match PurePath normalization
        pattern = self.pattern.rstrip("/")
        return wcpathlib.PurePath(relative_path).globmatch(
            pattern, flags=wcpathlib.GLOBSTAR | wcpathlib.BRACE | wcpathlib.DOTMATCH
        )


@dataclass
class WorkspaceAccessPolicy:
    """Sandbox-independent workspace path and access policy.

    Unlike :class:`Filesystem`, this class never touches the host filesystem.
    Its root and all resolved values are POSIX paths in the workspace's
    sandbox.  Filesystem operations and canonical symlink checks belong to the
    sandbox itself.
    """

    root: str
    accesses: list[AccessRule] = field(default_factory=list)
    default_access: AccessType = AccessType.EXCLUDE

    def __post_init__(self) -> None:
        root = PurePosixPath(self.root)
        if not root.is_absolute():
            raise ValueError("workspace access root must be an absolute sandbox path")
        self.root = posixpath.normpath(root.as_posix())

    def resolve_path(self, path: str | PurePosixPath) -> str:
        value = PurePosixPath(str(path))
        if value.is_absolute():
            return posixpath.normpath(value.as_posix())
        return posixpath.normpath(posixpath.join(self.root, value.as_posix()))

    def get_relative_path(self, path: str | PurePosixPath) -> str | None:
        resolved = PurePosixPath(self.resolve_path(path))
        try:
            relative = resolved.relative_to(PurePosixPath(self.root))
        except ValueError:
            return None
        return relative.as_posix()

    def get_access(self, path: str | PurePosixPath) -> AccessType:
        relative_path = self.get_relative_path(path)
        if relative_path is None:
            return AccessType.EXCLUDE
        result = self.default_access
        for rule in self.accesses:
            if rule.matches(relative_path):
                result = rule.access_type
        return result

    def can_read(self, path: str | PurePosixPath) -> bool:
        return self.get_access(path).can_read()

    def can_write(self, path: str | PurePosixPath) -> bool:
        return self.get_access(path).can_write()

    def validate_read(self, path: str | PurePosixPath) -> str:
        resolved = self.resolve_path(path)
        if not self.can_read(resolved):
            raise AccessDeniedError(f"Read access denied: {resolved}")
        return resolved

    def validate_write(self, path: str | PurePosixPath) -> str:
        resolved = self.resolve_path(path)
        if not self.can_write(resolved):
            raise AccessDeniedError(f"Write access denied: {resolved}")
        return resolved


@dataclass
class Filesystem:
    """Glob-based access control. Used internally by Workspace.

    Resolves paths on the local filesystem (using pathlib.Path.resolve()) and
    evaluates glob-pattern access rules. Workspace.set_access() creates a
    Filesystem rooted at ``project_path``; tools call workspace.validate_read()
    etc., which delegate here.

    Access Resolution Strategy:
        When a path matches multiple access rules, the LAST matching rule wins.
        This allows users to set broad rules first and then override with
        more specific rules later, similar to how .gitignore works.

        Example:
            accesses = [
                AccessRule(access_type=AccessType.READ, pattern="**"),           # Allow read to everything
                AccessRule(access_type=AccessType.EXCLUDE, pattern="**/.git/**"), # But exclude .git
                AccessRule(access_type=AccessType.WRITE, pattern="src/**"),       # Allow write to src/
            ]

    Attributes:
        root: The root directory. All paths are resolved relative to this.
        accesses: Access rules evaluated in order (last match wins).
        default_access: Access level when no rules match. Defaults to EXCLUDE.
    """

    root: Path
    accesses: list[AccessRule] = field(default_factory=list)
    default_access: AccessType = AccessType.EXCLUDE

    @staticmethod
    def _validate_dir(path: str | Path) -> Path:
        path = Path(path).resolve()
        if not path.exists():
            raise ValueError(f"Directory does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        return path

    def __post_init__(self):
        self.root = self._validate_dir(self.root)

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a path to an absolute path.

        Args:
            path: A path, either absolute or relative.

        Returns:
            The resolved absolute path. Relative paths are resolved against root.
        """
        path = Path(path)

        if path.is_absolute():
            return path.resolve()

        return (self.root / path).resolve()

    def get_relative_path(self, path: str | Path) -> str | None:
        """Get the path relative to root.

        Args:
            path: A path to convert.

        Returns:
            The path relative to root, or None if the path is outside root.
        """
        resolved = self.resolve_path(path)

        try:
            return str(resolved.relative_to(self.root))
        except ValueError:
            return None

    def get_access(self, path: str | Path) -> AccessType:
        """Determine the access level for a given path.

        Args:
            path: The path to check. Can be absolute or relative.

        Returns:
            The access level for the path. Returns EXCLUDE if the path
            is outside the root directory.

        Resolution:
            Rules are evaluated in order. The last matching rule determines
            the access level. If no rules match, returns default_access.
        """
        relative_path = self.get_relative_path(path)

        # Paths outside root are always excluded
        if relative_path is None:
            return AccessType.EXCLUDE

        # Find the last matching rule
        result = self.default_access

        for rule in self.accesses:
            if rule.matches(relative_path):
                result = rule.access_type

        return result

    def can_read(self, path: str | Path) -> bool:
        """Check if the path can be read.

        Args:
            path: The path to check.

        Returns:
            True if the path has read or write access.
        """
        return self.get_access(path).can_read()

    def can_write(self, path: str | Path) -> bool:
        """Check if the path can be written.

        Args:
            path: The path to check.

        Returns:
            True if the path has write access.
        """
        return self.get_access(path).can_write()

    def validate_read(self, path: str | Path) -> Path:
        """Validate that a path can be read and return the resolved path.

        Args:
            path: The path to validate.

        Returns:
            The resolved absolute path.

        Raises:
            AccessDeniedError: If the path cannot be read.
        """
        resolved = self.resolve_path(path)
        if not self.can_read(path):
            raise AccessDeniedError(f"Read access denied: {resolved}")
        return resolved

    def validate_write(self, path: str | Path) -> Path:
        """Validate that a path can be written and return the resolved path.

        Args:
            path: The path to validate.

        Returns:
            The resolved absolute path.

        Raises:
            AccessDeniedError: If the path cannot be written.
        """
        resolved = self.resolve_path(path)
        if not self.can_write(path):
            raise AccessDeniedError(f"Write access denied: {resolved}")
        return resolved
