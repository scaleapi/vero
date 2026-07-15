"""Tests verifying sandbox boundary correctness with a simulated remote sandbox.

MockRemoteSandbox uses two separate directories — "host" and "remote" — to
simulate a non-local sandbox where host and sandbox don't share a filesystem.
upload() copies host→remote, download() copies remote→host. All file ops
happen in the remote dir. This catches any code that assumes shared paths.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from vero.sandbox import CommandResult, FileStat, Sandbox
from vero.staging import SandboxStagingArea


class MockRemoteSandbox(Sandbox):
    """Sandbox that simulates a remote environment.

    File ops happen in `remote_root`. The host writes to `host_root`.
    upload/download actually copy between them. This catches any code
    that assumes host paths are accessible inside the sandbox.
    """

    def __init__(self, host_root: Path, remote_root: Path):
        self._host_root = host_root
        self._remote_root = remote_root

    @classmethod
    async def create(cls, **kwargs) -> MockRemoteSandbox:
        raise NotImplementedError("MockRemoteSandbox requires explicit host/remote roots")

    @property
    def root(self) -> str:
        return str(self._remote_root)

    def resolve_path(self, path: str) -> str:
        p = Path(path)
        if p.is_absolute():
            return str(p.resolve())
        return str((self._remote_root / p).resolve())

    async def read_file(self, path: str, encoding: str = "utf-8") -> str:
        resolved = Path(self.resolve_path(path))
        return resolved.read_text(encoding=encoding)

    async def read_file_bytes(self, path: str, limit: int | None = None) -> bytes:
        resolved = Path(self.resolve_path(path))
        with open(resolved, "rb") as f:
            return f.read(limit) if limit is not None else f.read()

    async def write_file(self, path: str, content: str, encoding: str = "utf-8") -> None:
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
        return sorted(e.name for e in Path(self.resolve_path(path)).iterdir())

    async def run(self, command, cwd=None, timeout=30, env=None) -> CommandResult:
        import asyncio

        if isinstance(command, str):
            command = ["bash", "-c", command]
        if cwd is None:
            cwd = str(self._remote_root)
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=env,
        )
        stdout, stderr = await proc.communicate()
        return CommandResult(
            stdout=stdout.decode().strip(),
            stderr=stderr.decode().strip(),
            returncode=proc.returncode or 0,
        )

    async def upload(self, local_path: str, remote_path: str) -> None:
        """Copy from host to remote. This MUST actually copy (not no-op)."""
        src = Path(local_path)
        dst = Path(remote_path)
        if not src.exists():
            return
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    async def download(self, remote_path: str, local_path: str) -> None:
        """Copy from remote to host. This MUST actually copy (not no-op)."""
        src = Path(remote_path)
        dst = Path(local_path)
        if not src.exists():
            return
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


class TestMockRemoteSandboxBasics:
    """Verify the mock itself works correctly."""

    @pytest.fixture
    def sandbox(self, tmp_path):
        host = tmp_path / "host"
        remote = tmp_path / "remote"
        host.mkdir()
        remote.mkdir()
        return MockRemoteSandbox(
            host_root=host,
            remote_root=remote,
        )

    @pytest.mark.asyncio
    async def test_host_and_remote_are_separate(self, sandbox, tmp_path):
        """Writing to sandbox doesn't create files on the host."""
        await sandbox.write_file("test.txt", "hello")

        # File exists in remote
        assert (tmp_path / "remote" / "test.txt").exists()
        # File does NOT exist in host
        assert not (tmp_path / "host" / "test.txt").exists()

    @pytest.mark.asyncio
    async def test_upload_copies_to_remote(self, sandbox, tmp_path):
        host_file = tmp_path / "host" / "data.txt"
        host_file.write_text("from host")

        remote_path = str(tmp_path / "remote" / "data.txt")
        await sandbox.upload(str(host_file), remote_path)

        assert (tmp_path / "remote" / "data.txt").read_text() == "from host"

    @pytest.mark.asyncio
    async def test_download_copies_from_remote(self, sandbox, tmp_path):
        await sandbox.write_file("result.txt", "from remote")

        local_path = str(tmp_path / "host" / "result.txt")
        await sandbox.download(str(tmp_path / "remote" / "result.txt"), local_path)

        assert (tmp_path / "host" / "result.txt").read_text() == "from remote"

    @pytest.mark.asyncio
    async def test_staging_area_exchanges_data_and_cleans_up(self, sandbox, tmp_path):
        source = tmp_path / "host" / "input.txt"
        source.write_text("input", encoding="utf-8")
        destination = tmp_path / "host" / "output.txt"

        async with SandboxStagingArea(sandbox) as staging:
            staging_root = Path(staging.root)
            await staging.upload(source, "inputs/input.txt")
            assert await staging.read_text("inputs/input.txt") == "input"
            await staging.write_text("outputs/output.txt", "output")
            await staging.download("outputs/output.txt", destination)

        assert destination.read_text(encoding="utf-8") == "output"
        assert not staging_root.exists()
