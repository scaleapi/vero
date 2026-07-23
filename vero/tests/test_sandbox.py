"""Tests for the Sandbox abstraction (LocalSandbox)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from vero.sandbox import CommandResult, FileStat, LocalSandbox, Sandbox
import vero.sandbox as sandbox_module


@pytest.fixture
def sandbox(tmp_path):
    """Create a LocalSandbox for I/O testing."""
    (tmp_path / "hello.txt").write_text("hello world\n")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.py").write_text("x = 1\n")
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "secret.txt").write_text("top secret\n")

    return LocalSandbox(root=tmp_path)


@pytest.fixture
def fast_process_group_termination(monkeypatch):
    terminate = sandbox_module._terminate_host_process_tree

    async def terminate_quickly(process):
        await terminate(process, grace_seconds=0.1)

    monkeypatch.setattr(
        sandbox_module,
        "_terminate_host_process_tree",
        terminate_quickly,
    )


class TestSandboxABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Sandbox()


class TestProperties:
    def test_root(self, sandbox, tmp_path):
        assert sandbox.root == str(tmp_path)

    def test_resolve_path(self, sandbox, tmp_path):
        result = sandbox.resolve_path("hello.txt")
        assert result == str(tmp_path / "hello.txt")

    def test_local_paths_are_host_visible(self, sandbox, tmp_path):
        assert sandbox.capabilities.host_paths
        assert sandbox.host_path("hello.txt") == tmp_path / "hello.txt"

    @pytest.mark.asyncio
    async def test_temporary_directory_is_cleaned_up(self, sandbox):
        async with sandbox.temporary_directory("vero-test-") as directory:
            assert await sandbox.is_dir(directory)
            assert Path(directory).name.startswith("vero-test-")
            assert directory == str(Path(directory).resolve())
        assert not await sandbox.exists(directory)


class TestFileOperations:
    @pytest.mark.asyncio
    async def test_read_file(self, sandbox):
        content = await sandbox.read_file("hello.txt")
        assert content == "hello world\n"

    @pytest.mark.asyncio
    async def test_read_file_bytes(self, sandbox):
        data = await sandbox.read_file_bytes("hello.txt")
        assert data == b"hello world\n"

    @pytest.mark.asyncio
    async def test_read_file_bytes_with_limit(self, sandbox):
        data = await sandbox.read_file_bytes("hello.txt", limit=5)
        assert data == b"hello"

    @pytest.mark.asyncio
    async def test_write_file(self, sandbox, tmp_path):
        await sandbox.write_file("new.txt", "new content\n")
        assert (tmp_path / "new.txt").read_text() == "new content\n"

    @pytest.mark.asyncio
    async def test_write_file_creates_parents(self, sandbox, tmp_path):
        await sandbox.write_file("a/b/c.txt", "deep\n")
        assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep\n"

    @pytest.mark.asyncio
    async def test_exists(self, sandbox):
        assert await sandbox.exists("hello.txt") is True
        assert await sandbox.exists("nonexistent.txt") is False

    @pytest.mark.asyncio
    async def test_is_file(self, sandbox):
        assert await sandbox.is_file("hello.txt") is True
        assert await sandbox.is_file("subdir") is False

    @pytest.mark.asyncio
    async def test_is_dir(self, sandbox):
        assert await sandbox.is_dir("subdir") is True
        assert await sandbox.is_dir("hello.txt") is False

    @pytest.mark.asyncio
    async def test_mkdir(self, sandbox, tmp_path):
        await sandbox.mkdir("new_dir/nested")
        assert (tmp_path / "new_dir" / "nested").is_dir()

    @pytest.mark.asyncio
    async def test_stat(self, sandbox):
        result = await sandbox.stat("hello.txt")
        assert isinstance(result, FileStat)
        assert result.st_size == len("hello world\n")

    @pytest.mark.asyncio
    async def test_list_dir(self, sandbox):
        entries = await sandbox.list_dir(sandbox.root)
        assert "hello.txt" in entries
        assert "subdir" in entries
        assert "private" in entries
        assert entries == sorted(entries)

    @pytest.mark.asyncio
    async def test_list_dir_subdir(self, sandbox):
        entries = await sandbox.list_dir("subdir")
        assert entries == ["nested.py"]


class TestShellExecution:
    @pytest.mark.asyncio
    async def test_run_simple_command(self, sandbox):
        result = await sandbox.run(["echo", "hello"])
        assert isinstance(result, CommandResult)
        assert result.stdout == "hello"
        assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_run_returns_stderr(self, sandbox):
        result = await sandbox.run(["ls", "/nonexistent_path_xyz"])
        assert result.returncode != 0
        assert result.stderr != ""

    @pytest.mark.asyncio
    async def test_run_does_not_raise_on_nonzero(self, sandbox):
        result = await sandbox.run(["false"])
        assert result.returncode != 0

    @pytest.mark.asyncio
    async def test_run_default_cwd_is_root(self, sandbox):
        result = await sandbox.run(["pwd"])
        assert result.stdout == sandbox.root

    @pytest.mark.asyncio
    async def test_run_custom_cwd(self, sandbox):
        result = await sandbox.run(["pwd"], cwd=sandbox.resolve_path("subdir"))
        assert result.stdout.endswith("subdir")

    @pytest.mark.asyncio
    async def test_run_string_command(self, sandbox):
        result = await sandbox.run("echo hello && echo world")
        assert "hello" in result.stdout
        assert "world" in result.stdout

    @pytest.mark.asyncio
    async def test_run_timeout(self, sandbox):
        result = await sandbox.run(["sleep", "10"], timeout=1)
        assert result.returncode == -1
        assert "timed out" in result.stderr

    @pytest.mark.asyncio
    async def test_timeout_terminates_descendant_processes(
        self,
        sandbox,
        tmp_path,
        fast_process_group_termination,
    ):
        marker = tmp_path / "timeout-descendant-survived"
        descendant = (
            "import signal, time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"time.sleep(0.8); Path({str(marker)!r}).write_text('leaked')"
        )
        parent = (
            "import subprocess, sys, time; "
            f"subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
            "time.sleep(60)"
        )

        result = await sandbox.run([sys.executable, "-c", parent], timeout=0.1)
        assert result.returncode == -1
        await asyncio.sleep(1)
        assert not marker.exists()

    @pytest.mark.asyncio
    async def test_cancellation_terminates_descendant_processes(
        self,
        sandbox,
        tmp_path,
        fast_process_group_termination,
    ):
        marker = tmp_path / "cancelled-descendant-survived"
        descendant = (
            "import signal, time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"time.sleep(0.8); Path({str(marker)!r}).write_text('leaked')"
        )
        parent = (
            "import subprocess, sys, time; "
            f"subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
            "time.sleep(60)"
        )
        task = asyncio.create_task(
            sandbox.run([sys.executable, "-c", parent], timeout=None)
        )
        await asyncio.sleep(0.1)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(1)
        assert not marker.exists()

    @pytest.mark.asyncio
    async def test_timeout_cleans_descendants_after_group_leader_exits(
        self,
        sandbox,
        tmp_path,
        fast_process_group_termination,
    ):
        marker = tmp_path / "detached-descendant-survived"
        descendant = (
            "import signal, time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"time.sleep(0.8); Path({str(marker)!r}).write_text('leaked')"
        )
        parent = (
            "import subprocess, sys; "
            f"subprocess.Popen([sys.executable, '-c', {descendant!r}])"
        )

        result = await sandbox.run([sys.executable, "-c", parent], timeout=0.1)
        assert result.returncode == -1
        await asyncio.sleep(1)
        assert not marker.exists()

    @pytest.mark.asyncio
    async def test_run_env(self, sandbox):
        import os

        env = {**os.environ, "MY_TEST_VAR": "test_value_123"}
        result = await sandbox.run(["printenv", "MY_TEST_VAR"], env=env)
        assert result.stdout == "test_value_123"


class TestFileTransfer:
    @pytest.mark.asyncio
    async def test_upload_file(self, sandbox, tmp_path):
        # Create a file outside the sandbox
        external = tmp_path / "external"
        external.mkdir()
        (external / "data.txt").write_text("uploaded\n")

        dest = sandbox.resolve_path("uploaded.txt")
        await sandbox.upload(str(external / "data.txt"), dest)

        content = await sandbox.read_file("uploaded.txt")
        assert content == "uploaded\n"

    @pytest.mark.asyncio
    async def test_upload_directory(self, sandbox, tmp_path):
        external = tmp_path / "external_dir"
        external.mkdir()
        (external / "a.txt").write_text("aaa\n")
        (external / "b.txt").write_text("bbb\n")

        dest = sandbox.resolve_path("imported")
        await sandbox.upload(str(external), dest)

        assert await sandbox.is_dir("imported")
        assert await sandbox.read_file(str(Path(dest) / "a.txt")) == "aaa\n"
        assert await sandbox.read_file(str(Path(dest) / "b.txt")) == "bbb\n"

    @pytest.mark.asyncio
    async def test_download_file(self, sandbox, tmp_path):
        await sandbox.write_file("to_download.txt", "download me\n")

        local_dest = tmp_path / "downloaded.txt"
        await sandbox.download(sandbox.resolve_path("to_download.txt"), str(local_dest))

        assert local_dest.read_text() == "download me\n"

    @pytest.mark.asyncio
    async def test_download_directory(self, sandbox, tmp_path):
        await sandbox.mkdir("export_dir")
        await sandbox.write_file(
            str(Path(sandbox.resolve_path("export_dir")) / "x.txt"), "xxx\n"
        )

        local_dest = tmp_path / "exported"
        await sandbox.download(sandbox.resolve_path("export_dir"), str(local_dest))

        assert local_dest.is_dir()
        assert (local_dest / "x.txt").read_text() == "xxx\n"

    @pytest.mark.asyncio
    async def test_upload_noop_same_path(self, sandbox):
        """upload is a no-op when source and dest resolve to the same path."""
        path = sandbox.resolve_path("hello.txt")
        await sandbox.upload(path, path)  # should not crash or duplicate
        content = await sandbox.read_file("hello.txt")
        assert content == "hello world\n"

    @pytest.mark.asyncio
    async def test_download_noop_same_path(self, sandbox):
        """download is a no-op when source and dest resolve to the same path."""
        path = sandbox.resolve_path("hello.txt")
        await sandbox.download(path, path)
        content = await sandbox.read_file("hello.txt")
        assert content == "hello world\n"


class _ImmediateProcess:
    returncode = 0

    async def communicate(self):
        return (b"", b"")


@pytest.mark.asyncio
async def test_run_as_drops_to_unprivileged_user(sandbox, monkeypatch):
    """run_as forwards the user/group and sheds supplementary groups."""
    captured: dict = {}

    async def fake_exec(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _ImmediateProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await sandbox.run(["true"], run_as="harness")

    assert captured["kwargs"]["user"] == "harness"
    assert captured["kwargs"]["group"] == "harness"
    assert captured["kwargs"]["extra_groups"] == []


@pytest.mark.asyncio
async def test_run_without_run_as_does_not_drop_privileges(sandbox, monkeypatch):
    captured: dict = {}

    async def fake_exec(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _ImmediateProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await sandbox.run(["true"])

    assert "user" not in captured["kwargs"]
    assert "group" not in captured["kwargs"]
