from __future__ import annotations

import asyncio
import difflib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from vero.tools.utils import is_tool


@dataclass
class IndexedArtifact:
    key: str
    content: str
    namespace: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now())
    versions: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.versions.append(self.content)

    @classmethod
    def from_file(
        cls, path: Path | str, namespace: str | None = None
    ) -> "IndexedArtifact":
        """Create an IndexedArtifact from a file path.

        Args:
            path: Path to the file to read content from.
            namespace: Optional namespace for the artifact.

        Returns:
            An IndexedArtifact with the file's stem as key and contents as content.
        """
        path = Path(path)
        content = path.read_text()
        key = path.stem
        return cls(key=key, content=content, namespace=namespace)

    @property
    def version(self) -> int:
        return len(self.versions) - 1

    def view_content(self, offset: int = 0, limit: int = 10_000) -> str:
        return self.content[offset : offset + limit]

    @staticmethod
    def compute_diff(
        old: str, new: str, fromfile: str = "old_content", tofile: str = "new_content"
    ) -> str:
        return difflib.unified_diff(old, new, fromfile=fromfile, tofile=tofile)

    def update_content(
        self, old_string: str, new_string: str, replace_all: bool = False
    ) -> bool:
        if old_string not in self.content:
            raise ValueError("`old_string` not found in content.")

        if replace_all:
            self.content = self.content.replace(old_string, new_string)
        else:
            self.content = self.content.replace(old_string, new_string, 1)

        self.versions.append(self.content)
        return True

    def view_diff(self, from_version: int, to_version: int) -> str:
        return self.compute_diff(
            self.versions[from_version],
            self.versions[to_version],
            fromfile=f"version={from_version}",
            tofile=f"version={to_version}",
        )


@dataclass
class ContextStore:
    """Key-value store for text artifacts."""

    exclude_tools: list[str] = field(default_factory=list)
    artifacts: dict[str, IndexedArtifact] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def bind(self, session) -> None:
        if session.skills:
            for namespace, path in session.skills.items():
                store = ContextStore.from_paths(path, namespace=namespace)
                self.artifacts.update(store.artifacts)

    @classmethod
    def from_paths(
        cls,
        paths: Path | str | list[Path | str],
        namespace: str | None = None,
        glob: str = "*.md",
    ) -> "ContextStore":
        """Create a ContextStore pre-populated with artifacts from a directory or list of files.

        Args:
            paths: Either a directory path (will glob for files) or a list of file paths.
            namespace: Namespace to assign to all loaded artifacts.
            glob: Glob pattern for files when paths is a directory (default: "*.md").

        Returns:
            A ContextStore instance with artifacts loaded from the paths.
        """
        artifacts = {}

        if isinstance(paths, (str, Path)):
            path = Path(paths)
            if path.is_dir():
                file_paths = list(path.glob(glob))
            else:
                file_paths = [path]
        else:
            file_paths = [Path(p) for p in paths]

        for file_path in file_paths:
            artifact = IndexedArtifact.from_file(file_path, namespace=namespace)
            artifacts[artifact.key] = artifact

        return cls(artifacts=artifacts)

    def set_artifact(
        self, key: str, content: str, namespace: str | None = None
    ) -> "IndexedArtifact":
        """Set an artifact directly (not exposed as a tool). For programmatic use.

        Args:
            key: The key of the artifact.
            content: The content of the artifact.
            namespace: Optional namespace for the artifact.

        Returns:
            The created IndexedArtifact.
        """
        if key in self.artifacts:
            raise ValueError(f"`{key}` already exists.")
        artifact = IndexedArtifact(key=key, content=content, namespace=namespace)
        self.artifacts[key] = artifact
        return artifact

    def set_artifact_from_file(
        self, path: Path | str, namespace: str | None = None
    ) -> "IndexedArtifact":
        """Set an artifact from a file (not exposed as a tool). For programmatic use.

        Args:
            path: Path to the file.
            namespace: Optional namespace for the artifact.

        Returns:
            The created IndexedArtifact.
        """
        artifact = IndexedArtifact.from_file(path, namespace=namespace)
        if artifact.key in self.artifacts:
            raise ValueError(f"`{artifact.key}` already exists.")
        self.artifacts[artifact.key] = artifact
        return artifact

    @is_tool
    async def create_artifact(self, content: str, key: str) -> str:
        """Create an artifact in the context store.

        Args:
            content: The content of the artifact.
            key: The key of the artifact.

        Returns:
            A message indicating that the artifact was added to the context store.
        """
        async with self._lock:
            if key in self.artifacts:
                raise ValueError(f"`{key}` already exists in context store.")

            self.artifacts[key] = IndexedArtifact(key=key, content=content)
            return f"Artifact `{key}` added to context store."

    @is_tool
    async def list_artifacts(
        self, namespace: str | None = None
    ) -> list[dict[str, str | None]]:
        """List all artifacts in the context store.

        Args:
            namespace: Optional namespace to filter by. If None, returns all artifacts.

        Returns:
            A list of artifacts with their keys and namespaces.
        """
        async with self._lock:
            result = []
            for key, artifact in self.artifacts.items():
                if namespace is None or artifact.namespace == namespace:
                    result.append({"key": key, "namespace": artifact.namespace})
            return result

    @is_tool
    async def view_artifact(
        self, key: str, offset: int = 0, limit: int = 10_000
    ) -> str:
        """View an artifact from the context store.

        Args:
            key: The key of the artifact to view.
            offset: The starting character index to view.
            limit: The number of characters to view.

        Returns:
            The content of the artifact.
        """
        async with self._lock:
            return self.artifacts[key].view_content(offset=offset, limit=limit)

    @is_tool
    async def view_artifact_diff(
        self, key: str, from_version: int = -2, to_version: int = -1
    ) -> str:
        """View the diff between two versions of an artifact. Defaults to diff between the current and immediate previous version.

        Args:
            key: The key of the artifact to view the diff for.
            from_version: The version to view the diff from.
            to_version: The version to view the diff to.

        Returns:
            The diff between the two versions of the artifact.
        """
        async with self._lock:
            return self.artifacts[key].view_diff(
                from_version=from_version, to_version=to_version
            )

    @is_tool
    async def update_artifact(
        self, key: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Update an artifact in the context store.

        Args:
            key: The key of the artifact to update.
            old_string: The string to replace.
            new_string: The string to replace it with.
            replace_all: Whether to replace all occurrences of the old string.

        Returns:
            A message indicating that the artifact was updated in the context store.
        """
        async with self._lock:
            return self.artifacts[key].update_content(
                old_string=old_string, new_string=new_string, replace_all=replace_all
            )
