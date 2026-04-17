"""Abstract sandbox: filesystem + shell execution.

A Sandbox provides a unified interface for file I/O and shell commands.
Tools call sandbox methods instead of using pathlib/subprocess directly,
making it possible to swap in different backends (local, container, remote VM).

The default implementation (LocalSandbox) wraps pathlib.Path and
asyncio.create_subprocess_exec.  Access control is handled by Workspace,
not Sandbox — the sandbox is a dumb I/O layer.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple

# =============================================================================
# Data types
# =============================================================================


class FileStat(NamedTuple):
    """Minimal file stat info."""

    st_size: int


class CommandResult(NamedTuple):
    """Result of a shell command execution."""

    stdout: str
    stderr: str
    returncode: int


# =============================================================================
# Abstract base
# =============================================================================


class Sandbox(ABC):
    """Abstract sandbox providing filesystem access and shell execution.

    A Sandbox represents a "computer" — an execution environment where vero
    runs.  For a local sandbox the root is the user's home directory; for a
    Docker sandbox it would be ``/``.  Vero home (``~/.vero/``) and the git
    workspace both live *within* the sandbox.

    Access control is **not** a sandbox concern — it lives on Workspace.
    The sandbox is pure I/O: read, write, run, exist-checks.
    """

    # ── Construction ───────────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    async def create(cls, **kwargs) -> Sandbox:
        """Create a new, bare sandbox instance.

        Each subclass knows how to spin up its own environment.
        """
        ...

    # ── Properties ──────────────────────────────────────────────────────

    @property
    @abstractmethod
    def root(self) -> str:
        """Root directory of the sandbox."""
        ...

    # ── Path resolution ────────────────────────────────────────────────

    @abstractmethod
    def resolve_path(self, path: str) -> str:
        """Resolve a path relative to sandbox root. For sandbox-internal use."""
        ...

    # ── Filesystem (async) ──────────────────────────────────────────────

    @abstractmethod
    async def read_file(self, path: str, encoding: str = "utf-8") -> str: ...

    @abstractmethod
    async def read_file_bytes(self, path: str, limit: int | None = None) -> bytes: ...

    @abstractmethod
    async def write_file(
        self, path: str, content: str, encoding: str = "utf-8"
    ) -> None: ...

    @abstractmethod
    async def exists(self, path: str) -> bool: ...

    @abstractmethod
    async def is_file(self, path: str) -> bool: ...

    @abstractmethod
    async def is_dir(self, path: str) -> bool: ...

    @abstractmethod
    async def mkdir(self, path: str, parents: bool = True) -> None: ...

    @abstractmethod
    async def stat(self, path: str) -> FileStat: ...

    @abstractmethod
    async def list_dir(self, path: str) -> list[str]: ...

    # ── Shell (async) ───────────────────────────────────────────────────

    @abstractmethod
    async def run(
        self,
        command: str | list[str],
        cwd: str | None = None,
        timeout: int | None = 30,
        env: dict[str, str] | None = None,
    ) -> CommandResult: ...

    # ── Host ↔ Sandbox file transfer (async) ───────────────────────────

    @abstractmethod
    async def upload(self, local_path: str, remote_path: str) -> None:
        """Copy a file or directory from the host into the sandbox.

        For LocalSandbox this is a local copy (no-op if paths match).
        For remote sandboxes this would be docker cp, scp, etc.
        """
        ...

    @abstractmethod
    async def download(self, remote_path: str, local_path: str) -> None:
        """Copy a file or directory from the sandbox to the host.

        For LocalSandbox this is a local copy (no-op if paths match).
        For remote sandboxes this would be docker cp, scp, etc.
        """
        ...


# =============================================================================
# Local implementation
# =============================================================================


class LocalSandbox(Sandbox):
    """Sandbox backed by the local filesystem and subprocess execution.

    File I/O uses pathlib, shell execution uses asyncio.create_subprocess_exec.
    No access control — that is handled by Workspace.

    Uses pathlib.Path for path resolution and manipulation.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()

    @classmethod
    async def create(cls, root: Path | str | None = None, **kwargs) -> LocalSandbox:
        """Create a LocalSandbox rooted at the user's home directory."""
        if root is None:
            root = Path.home()
        root = Path(root).resolve()
        return cls(root=root)

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def root(self) -> str:
        return str(self._root)

    # ── Path resolution ────────────────────────────────────────────────

    def resolve_path(self, path: str) -> str:
        p = Path(path)
        if p.is_absolute():
            return str(p.resolve())
        return str((self._root / p).resolve())

    # ── Filesystem ──────────────────────────────────────────────────────

    async def read_file(self, path: str, encoding: str = "utf-8") -> str:
        resolved = Path(self.resolve_path(path))
        return resolved.read_text(encoding=encoding)

    async def read_file_bytes(self, path: str, limit: int | None = None) -> bytes:
        resolved = Path(self.resolve_path(path))
        with open(resolved, "rb") as f:
            return f.read(limit) if limit is not None else f.read()

    async def write_file(
        self, path: str, content: str, encoding: str = "utf-8"
    ) -> None:
        resolved = Path(self.resolve_path(path))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)

    async def exists(self, path: str) -> bool:
        return Path(self.resolve_path(path)).exists()

    async def is_file(self, path: str) -> bool:
        return Path(self.resolve_path(path)).is_file()

    async def is_dir(self, path: str) -> bool:
        return Path(self.resolve_path(path)).is_dir()

    async def mkdir(self, path: str, parents: bool = True) -> None:
        Path(self.resolve_path(path)).mkdir(parents=parents, exist_ok=True)

    async def stat(self, path: str) -> FileStat:
        st = Path(self.resolve_path(path)).stat()
        return FileStat(st_size=st.st_size)

    async def list_dir(self, path: str) -> list[str]:
        resolved = Path(self.resolve_path(path))
        return sorted(entry.name for entry in resolved.iterdir())

    # ── Shell ───────────────────────────────────────────────────────────

    async def run(
        self,
        command: str | list[str],
        cwd: str | None = None,
        timeout: int | None = 30,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Run a command via subprocess.

        Returns CommandResult with stdout, stderr, returncode.
        Does NOT raise on non-zero exit — caller decides how to handle.
        """
        if isinstance(command, str):
            command = ["bash", "-c", command]

        if cwd is None:
            cwd = str(self._root)

        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=env,
        )

        async def terminate():
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return CommandResult(
                stdout=stdout.decode().strip(),
                stderr=stderr.decode().strip(),
                returncode=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            await asyncio.shield(terminate())
            cmd_str = " ".join(command)
            return CommandResult(
                stdout="",
                stderr=f"Command '{cmd_str}' timed out after {timeout} seconds",
                returncode=-1,
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            await asyncio.shield(terminate())
            raise

    # ── Host ↔ Sandbox file transfer ───────────────────────────────────

    async def upload(self, local_path: str, remote_path: str) -> None:
        """Copy from host to sandbox. No-op if paths resolve to the same location."""
        import shutil

        src = Path(local_path).resolve()
        dst = Path(remote_path).resolve()
        if src == dst:
            return
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    async def download(self, remote_path: str, local_path: str) -> None:
        """Copy from sandbox to host. No-op if paths resolve to the same location."""
        import shutil

        src = Path(remote_path).resolve()
        dst = Path(local_path).resolve()
        if src == dst:
            return
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
