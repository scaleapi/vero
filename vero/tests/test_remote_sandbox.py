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


class TestArtifactsWithRemoteSandbox:
    """Verify artifacts write into the remote sandbox, not the host."""

    @pytest.mark.asyncio
    async def test_traces_artifact_writes_to_sandbox(self, tmp_path):
        from vero.artifacts import TracesArtifact
        from vero.core.db.candidate import Candidate
        from vero.core.db.dataset import DatasetSample, DatasetSubset
        from vero.core.db.result import ExperimentResult, SampleResult
        from vero.core.db.run import ExperimentRun

        host = tmp_path / "host"
        remote = tmp_path / "remote"
        host.mkdir()
        remote.mkdir()
        vero_dir = str(remote / "_vero")

        sandbox = MockRemoteSandbox(
            host_root=host,
            remote_root=remote,
        )

        # Create a minimal experiment
        candidate = Candidate(commit="abc12345" * 5, repo_name="test")
        dataset_subset = DatasetSubset(split="train", dataset_id="ds")
        run = ExperimentRun(candidate=candidate, dataset_subset=dataset_subset)
        sample = SampleResult(
            output="hello",
            score=1.0,
            dataset_sample=DatasetSample(sample_id=0, split="train", dataset_id="ds"),
        )
        result = ExperimentResult.create_with_status(
            run_id=run.id, sample_results={0: sample}, error_rate=0.1
        )

        from vero.core.db.database import Experiment

        experiment = Experiment(run=run, result=result)

        # Create a minimal policy-like object for split_accesses
        class FakePolicy:
            split_accesses = []

        artifact = TracesArtifact()
        await artifact.on_experiment(FakePolicy(), experiment, vero_dir, sandbox)

        # Traces should be in remote, not host
        trace_dir = remote / "_vero" / "traces" / "train__abc12345"
        assert trace_dir.exists()
        assert (trace_dir / "summary.json").exists()
        assert (trace_dir / "0.json").exists()

        # Nothing in host
        assert not (host / "_vero").exists()

    @pytest.mark.asyncio
    async def test_dataset_artifact_writes_to_sandbox(self, tmp_path, monkeypatch):
        import json

        from vero.artifacts import DatasetArtifact

        host = tmp_path / "host"
        remote = tmp_path / "remote"
        host.mkdir()
        remote.mkdir()
        vero_dir = str(remote / "_vero")

        sandbox = MockRemoteSandbox(
            host_root=host,
            remote_root=remote,
        )

        # Set up a dataset in the host's cache
        from vero.core.dataset.store import save_dataset

        vero_home = tmp_path / "vero_home"
        sessions_dir = vero_home / "sessions"
        dataset_cache = vero_home / "datasets"
        sessions_dir.mkdir(parents=True)
        dataset_cache.mkdir(parents=True)

        from datasets import Dataset, DatasetDict

        ds = DatasetDict({"train": Dataset.from_dict({"q": ["hi"], "a": ["bye"]})})
        session_id = "test-session"
        save_dataset(sessions_dir, dataset_cache, session_id, "myds", ds)

        class FakePolicy:
            pass

        FakePolicy.session_id = "test-session"
        FakePolicy.sessions_dir = sessions_dir
        FakePolicy.dataset_cache = dataset_cache
        FakePolicy.split_accesses = []

        artifact = DatasetArtifact()
        await artifact.on_init(FakePolicy(), vero_dir, sandbox)

        # Dataset should be in remote
        sample_path = remote / "_vero" / "datasets" / "myds" / "train" / "0.json"
        assert sample_path.exists()
        sample = json.loads(sample_path.read_text())
        assert sample["q"] == "hi"

        # Nothing in host
        assert not (host / "_vero").exists()
