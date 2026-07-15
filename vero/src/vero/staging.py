"""Managed exchange directory between the host and a sandbox."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Self

from vero.sandbox import Sandbox


class SandboxStagingArea:
    """Temporary sandbox directory with explicit host transfer operations."""

    def __init__(self, sandbox: Sandbox, *, prefix: str = "vero-") -> None:
        self.sandbox = sandbox
        self.prefix = prefix
        self.root: str | None = None
        self._temporary_directory = None

    async def __aenter__(self) -> Self:
        self._temporary_directory = self.sandbox.temporary_directory(self.prefix)
        self.root = await self._temporary_directory.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        assert self._temporary_directory is not None
        await self._temporary_directory.__aexit__(exc_type, exc, traceback)
        self.root = None

    @staticmethod
    def _relative_path(relative_path: str) -> PurePosixPath:
        value = PurePosixPath(relative_path)
        if (
            not relative_path
            or "\\" in relative_path
            or value.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        ):
            raise ValueError("staging path must be a safe relative POSIX path")
        return value

    def path(self, relative_path: str) -> str:
        if self.root is None:
            raise RuntimeError("sandbox staging area is not active")
        value = self._relative_path(relative_path)
        return (PurePosixPath(self.root) / value).as_posix()

    async def mkdir(self, relative_path: str) -> str:
        path = self.path(relative_path)
        await self.sandbox.mkdir(path)
        return path

    async def write_text(self, relative_path: str, value: str) -> str:
        path = self.path(relative_path)
        await self.sandbox.write_file(path, value)
        return path

    async def read_text(self, relative_path: str) -> str:
        return await self.sandbox.read_file(self.path(relative_path))

    async def exists(self, relative_path: str) -> bool:
        return await self.sandbox.exists(self.path(relative_path))

    async def upload(self, local_path: Path | str, relative_path: str) -> str:
        remote_path = self.path(relative_path)
        await self.sandbox.upload(str(local_path), remote_path)
        return remote_path

    async def download(self, relative_path: str, local_path: Path | str) -> Path:
        destination = Path(local_path)
        await self.sandbox.download(self.path(relative_path), str(destination))
        return destination
