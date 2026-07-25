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
import os
import posixpath
import shutil
import signal
import tempfile
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, NamedTuple


async def _terminate_host_process_tree(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float = 5,
) -> None:
    """Terminate a host subprocess and every descendant in its process group."""

    if process.returncode is not None and os.name != "posix":
        return

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return

    if process.returncode is None:
        try:
            await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        except asyncio.TimeoutError:
            pass

    # The group leader may exit before descendants that ignore SIGTERM. Always
    # sweep the original process group with SIGKILL before returning.
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.returncode is None:
            process.kill()
    except ProcessLookupError:
        pass
    if process.returncode is None:
        await process.wait()


async def _cleanup_host_process(
    process: asyncio.subprocess.Process,
    before_terminate: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Run backend cleanup, then unconditionally reap the host process group."""

    try:
        if before_terminate is not None:
            await before_terminate()
    finally:
        await _terminate_host_process_tree(process)


class FileStat(NamedTuple):
    """Minimal file stat info."""

    st_size: int


class CommandResult(NamedTuple):
    """Result of a shell command execution."""

    stdout: str
    stderr: str
    returncode: int


@dataclass(frozen=True)
class SandboxCapabilities:
    """Execution features exposed by a sandbox implementation."""

    posix: bool = True
    host_paths: bool = False


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

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities()

    def host_path(self, path: str) -> Path | None:
        """Return a host-visible equivalent, or ``None`` for isolated paths."""

        return None

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
        run_as: str | None = None,
    ) -> CommandResult:
        """Run a command. ``run_as`` drops the process to that unprivileged
        user (and its same-named primary group), shedding supplementary
        groups; it requires the caller to be privileged and is how untrusted
        code is isolated from the trusted process that launches it."""
        ...

    async def canonicalize(self, path: str) -> str:
        """Resolve a sandbox path, including symlinks, inside the sandbox."""

        result = await self.run(["realpath", path])
        if result.returncode != 0:
            raise FileNotFoundError(result.stderr or path)
        return result.stdout.strip()

    async def remove(self, path: str, *, recursive: bool = False) -> None:
        """Remove a sandbox path."""

        command = ["rm"]
        if recursive:
            command.append("-rf")
        else:
            command.append("-f")
        command.extend(["--", path])
        result = await self.run(command)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or f"failed to remove {path}")

    @asynccontextmanager
    async def temporary_directory(self, prefix: str = "vero-") -> AsyncIterator[str]:
        """Create and clean up a temporary directory inside the sandbox."""

        safe_prefix = "".join(
            character
            for character in prefix
            if character.isalnum() or character in "-_"
        )
        template = f"/tmp/{safe_prefix or 'vero-'}XXXXXX"
        result = await self.run(["mktemp", "-d", template])
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr or "failed to create sandbox temporary directory"
            )
        path = await self.canonicalize(result.stdout.strip())
        try:
            yield path
        finally:
            await asyncio.shield(self.remove(path, recursive=True))

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

    async def close(self) -> None:
        """Release resources owned by this sandbox. Default: no-op."""

        return None


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

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(host_paths=True)

    def host_path(self, path: str) -> Path:
        return Path(self.resolve_path(path))

    # ── Path resolution ────────────────────────────────────────────────

    def resolve_path(self, path: str) -> str:
        p = Path(path)
        if p.is_absolute():
            return str(p.resolve())
        return str((self._root / p).resolve())

    async def canonicalize(self, path: str) -> str:
        resolved = Path(self.resolve_path(path))
        if not resolved.exists():
            raise FileNotFoundError(path)
        return str(resolved.resolve())

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
        run_as: str | None = None,
    ) -> CommandResult:
        """Run a command via subprocess.

        Returns CommandResult with stdout, stderr, returncode.
        Does NOT raise on non-zero exit — caller decides how to handle.
        """
        if isinstance(command, str):
            command = ["bash", "-c", command]

        if cwd is None:
            cwd = str(self._root)

        privilege_kwargs: dict[str, object] = {}
        if run_as is not None:
            # Drop to the unprivileged user and its same-named primary group,
            # shedding supplementary groups. Requires the launching process to
            # be privileged (root); otherwise create_subprocess_exec raises
            # PermissionError — the intended fail-closed behavior for isolation.
            privilege_kwargs = {
                "user": run_as,
                "group": run_as,
                "extra_groups": [],
            }

        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=env,
            start_new_session=os.name == "posix",
            **privilege_kwargs,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return CommandResult(
                stdout=stdout.decode().strip(),
                stderr=stderr.decode().strip(),
                returncode=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            await asyncio.shield(_cleanup_host_process(proc))
            cmd_str = " ".join(command)
            return CommandResult(
                stdout="",
                stderr=f"Command '{cmd_str}' timed out after {timeout} seconds",
                returncode=-1,
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            await asyncio.shield(_cleanup_host_process(proc))
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


class DockerSandbox(Sandbox):
    """POSIX sandbox backed by a Docker container with no shared filesystem."""

    def __init__(
        self,
        container_id: str,
        *,
        root: str = "/workspace",
        docker_executable: str = "docker",
        owns_container: bool = False,
    ) -> None:
        self.container_id = container_id
        self._root = posixpath.normpath(root)
        self.docker_executable = docker_executable
        self.owns_container = owns_container
        self._closed = False

    @classmethod
    async def create(
        cls,
        *,
        image: str,
        root: str = "/workspace",
        name: str | None = None,
        docker_executable: str | None = None,
        **kwargs,
    ) -> DockerSandbox:
        executable = docker_executable or shutil.which("docker")
        if executable is None:
            raise ValueError("docker is required to create a DockerSandbox")
        command = [
            executable,
            "run",
            "--detach",
            "--rm",
            "--workdir",
            root,
        ]
        if name is not None:
            command.extend(["--name", name])
        command.extend([image, "sh", "-c", "while :; do sleep 3600; done"])
        result = await cls._host_command(command, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "failed to create Docker sandbox")
        sandbox = cls(
            result.stdout.strip(),
            root=root,
            docker_executable=executable,
            owns_container=True,
        )
        mkdir_result = await sandbox.run(["mkdir", "-p", root])
        if mkdir_result.returncode != 0:
            await sandbox.close()
            raise RuntimeError(mkdir_result.stderr or f"failed to create {root}")
        return sandbox

    @classmethod
    async def from_container(
        cls,
        container_id: str,
        *,
        root: str = "/workspace",
        docker_executable: str | None = None,
    ) -> DockerSandbox:
        executable = docker_executable or shutil.which("docker")
        if executable is None:
            raise ValueError("docker is required to attach a DockerSandbox")
        sandbox = cls(
            container_id,
            root=root,
            docker_executable=executable,
            owns_container=False,
        )
        result = await sandbox.run(["mkdir", "-p", root])
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr or f"cannot access Docker container {container_id}"
            )
        return sandbox

    @staticmethod
    async def _host_command(
        command: list[str],
        *,
        timeout: int | float | None = 30,
        before_terminate: Callable[[], Awaitable[None]] | None = None,
    ) -> CommandResult:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            await asyncio.shield(_cleanup_host_process(process, before_terminate))
            return CommandResult("", f"Command timed out after {timeout} seconds", -1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            await asyncio.shield(_cleanup_host_process(process, before_terminate))
            raise
        return CommandResult(
            stdout.decode(errors="replace").strip(),
            stderr.decode(errors="replace").strip(),
            process.returncode or 0,
        )

    async def _docker(
        self, *arguments: str, timeout: int | float | None = 30
    ) -> CommandResult:
        return await self._host_command(
            [self.docker_executable, *arguments],
            timeout=timeout,
        )

    @property
    def root(self) -> str:
        return self._root

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(posix=True, host_paths=False)

    def resolve_path(self, path: str) -> str:
        if posixpath.isabs(path):
            return posixpath.normpath(path)
        return posixpath.normpath(posixpath.join(self._root, path))

    async def read_file(self, path: str, encoding: str = "utf-8") -> str:
        return (await self.read_file_bytes(path)).decode(encoding)

    async def read_file_bytes(self, path: str, limit: int | None = None) -> bytes:
        command = [self.docker_executable, "exec", self.container_id]
        if limit is None:
            command.extend(["cat", self.resolve_path(path)])
        else:
            command.extend(["head", "-c", str(limit), self.resolve_path(path)])
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise FileNotFoundError(stderr.decode(errors="replace") or path)
        return stdout

    async def write_file(
        self, path: str, content: str, encoding: str = "utf-8"
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="vero-docker-") as directory:
            local_path = Path(directory) / "content"
            local_path.write_text(content, encoding=encoding)
            await self.upload(str(local_path), self.resolve_path(path))

    async def exists(self, path: str) -> bool:
        return (await self.run(["test", "-e", self.resolve_path(path)])).returncode == 0

    async def is_file(self, path: str) -> bool:
        return (await self.run(["test", "-f", self.resolve_path(path)])).returncode == 0

    async def is_dir(self, path: str) -> bool:
        return (await self.run(["test", "-d", self.resolve_path(path)])).returncode == 0

    async def mkdir(self, path: str, parents: bool = True) -> None:
        command = ["mkdir"]
        if parents:
            command.append("-p")
        command.append(self.resolve_path(path))
        result = await self.run(command)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or f"failed to create {path}")

    async def stat(self, path: str) -> FileStat:
        result = await self.run(["stat", "-c", "%s", self.resolve_path(path)])
        if result.returncode != 0:
            raise FileNotFoundError(result.stderr or path)
        return FileStat(st_size=int(result.stdout))

    async def list_dir(self, path: str) -> list[str]:
        result = await self.run(["ls", "-1A", self.resolve_path(path)])
        if result.returncode != 0:
            raise FileNotFoundError(result.stderr or path)
        return sorted(line for line in result.stdout.splitlines() if line)

    async def run(
        self,
        command: str | list[str],
        cwd: str | None = None,
        timeout: int | None = 30,
        env: dict[str, str] | None = None,
        run_as: str | None = None,
    ) -> CommandResult:
        pid_file = f"/tmp/vero-exec-{uuid.uuid4().hex}.pid"
        arguments = ["exec"]
        if run_as is not None:
            arguments.extend(["-u", run_as])
        if cwd is not None:
            arguments.extend(["--workdir", self.resolve_path(cwd)])
        for name, value in (env or {}).items():
            arguments.extend(["--env", f"{name}={value}"])
        arguments.append(self.container_id)
        payload = ["sh", "-c", command] if isinstance(command, str) else command
        wrapper = (
            'pid_file="$1"; shift; '
            'printf \'%s\\n\' "$$" > "$pid_file"; '
            "trap 'rm -f \"$pid_file\"' EXIT; "
            '"$@"'
        )
        arguments.extend(
            ["setsid", "--wait", "sh", "-c", wrapper, "sh", pid_file, *payload]
        )

        async def terminate_container_group() -> None:
            script = """
pid_file=$1
attempt=0
while [ ! -s "$pid_file" ] && [ "$attempt" -lt 20 ]; do
    sleep 0.05
    attempt=$((attempt + 1))
done
if [ -s "$pid_file" ]; then
    pgid=$(cat "$pid_file")
    kill -TERM -- "-$pgid" 2>/dev/null || true
    attempt=0
    while kill -0 -- "-$pgid" 2>/dev/null && [ "$attempt" -lt 10 ]; do
        sleep 0.5
        attempt=$((attempt + 1))
    done
    kill -KILL -- "-$pgid" 2>/dev/null || true
fi
rm -f "$pid_file"
"""
            await self._docker(
                "exec",
                self.container_id,
                "sh",
                "-c",
                script,
                "sh",
                pid_file,
                timeout=10,
            )

        return await self._host_command(
            [self.docker_executable, *arguments],
            timeout=timeout,
            before_terminate=terminate_container_group,
        )

    async def canonicalize(self, path: str) -> str:
        result = await self.run(["readlink", "-f", self.resolve_path(path)])
        if result.returncode != 0:
            raise FileNotFoundError(result.stderr or path)
        return result.stdout.strip()

    async def upload(self, local_path: str, remote_path: str) -> None:
        source = Path(local_path).resolve()
        if not source.exists():
            raise FileNotFoundError(local_path)
        destination = self.resolve_path(remote_path)
        if source.is_dir():
            await self.mkdir(destination)
            docker_source = f"{source}/."
        else:
            await self.mkdir(posixpath.dirname(destination))
            docker_source = str(source)
        result = await self._docker(
            "cp", docker_source, f"{self.container_id}:{destination}", timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or f"failed to upload {local_path}")

    async def download(self, remote_path: str, local_path: str) -> None:
        source = self.resolve_path(remote_path)
        destination = Path(local_path).resolve()
        if not await self.exists(source):
            raise FileNotFoundError(remote_path)
        if await self.is_dir(source):
            destination.mkdir(parents=True, exist_ok=True)
            docker_source = f"{self.container_id}:{source}/."
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            docker_source = f"{self.container_id}:{source}"
        result = await self._docker("cp", docker_source, str(destination), timeout=120)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or f"failed to download {remote_path}")

    async def close(self) -> None:
        if self._closed or not self.owns_container:
            return
        self._closed = True
        result = await self._docker("rm", "--force", self.container_id, timeout=30)
        if result.returncode != 0 and "No such container" not in result.stderr:
            raise RuntimeError(result.stderr or "failed to remove Docker sandbox")
