"""Regression tests for the git plumbing a resumed run depends on.

The motivating incident: a run died late, and the resume could not recognise
work its own previous attempt had already committed, because every save
produced a fresh sha from the wall clock. The other two cases here are the
crash residue that stopped the retry from getting that far at all: a git call
killed by the sandbox's 30 second default leaving an index lock, and a ref lock
left behind by a writer that was killed mid-transaction.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from vero.candidate import Candidate
from vero.candidate_repository import GitCandidateRepository
from vero.sandbox import CommandResult, LocalSandbox
from vero.workspace import GitWorkspace


def _git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _initialize(path: Path) -> str:
    _git(path, "init", "-b", "main")
    (path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _git(path, "add", "--all")
    _git(
        path,
        "-c",
        "user.name=vero",
        "-c",
        "user.email=vero@localhost",
        "commit",
        "-m",
        "baseline",
    )
    return _git(path, "rev-parse", "HEAD")


class _TimeoutRecordingSandbox(LocalSandbox):
    """Local sandbox that remembers the timeout each command was given."""

    def __init__(self, root: Path) -> None:
        super().__init__(root=root)
        self.timeouts: list[int | None] = []

    async def run(
        self,
        command: str | list[str],
        cwd: str | None = None,
        timeout: int | None = 30,
        env: dict[str, str] | None = None,
        run_as: str | None = None,
    ) -> CommandResult:
        self.timeouts.append(timeout)
        return await super().run(
            command, cwd=cwd, timeout=timeout, env=env, run_as=run_as
        )


@pytest.mark.asyncio
async def test_workspace_git_calls_get_two_minutes(tmp_path: Path):
    """A git call must not inherit the sandbox's 30 second default."""

    _initialize(tmp_path)
    sandbox = _TimeoutRecordingSandbox(tmp_path)
    workspace = GitWorkspace(sandbox=sandbox, root=str(tmp_path))

    await workspace.current_version()

    assert sandbox.timeouts == [120]


@pytest.mark.asyncio
async def test_save_pins_both_commit_dates(tmp_path: Path):
    """Author and committer dates are pinned, so the sha is a function of content."""

    _initialize(tmp_path)
    sandbox = LocalSandbox(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(tmp_path))

    (tmp_path / "main.py").write_text("x = 2\n", encoding="utf-8")
    await workspace.save("candidate")

    assert _git(tmp_path, "log", "-1", "--format=%at %ct") == "0 0"


@pytest.mark.asyncio
async def test_save_of_identical_content_reproduces_the_same_sha(tmp_path: Path):
    """The sha a resumed run recomputes has to match the one it already stored."""

    baseline = _initialize(tmp_path)
    sandbox = LocalSandbox(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(tmp_path))

    (tmp_path / "main.py").write_text("x = 2\n", encoding="utf-8")
    first = await workspace.save("candidate")

    # Rewind to the same parent and redo the identical save, the way a resumed
    # run replays a round it had already completed. The sleep pushes the wall
    # clock past a whole second, which is the granularity a commit date is
    # stored at: without it the two commits could share a timestamp and the
    # test would pass even with the dates unpinned.
    _git(tmp_path, "reset", "--hard", baseline)
    await asyncio.sleep(1.1)
    (tmp_path / "main.py").write_text("x = 2\n", encoding="utf-8")
    second = await workspace.save("candidate")

    assert second == first


@pytest.mark.asyncio
async def test_create_sweeps_stale_vero_ref_locks_only(tmp_path: Path):
    """Opening the repository clears vero's own crash residue and nothing else."""

    source = tmp_path / "source"
    source.mkdir()
    baseline_version = _initialize(source)
    sandbox = await LocalSandbox.create(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(source))
    repository = await GitCandidateRepository.create(
        tmp_path / "session" / "candidates",
        workspace=workspace,
    )

    candidate = Candidate.from_version(baseline_version, candidate_id="candidate")
    stale_lock = repository.repository_path / (
        repository._candidate_ref(candidate.id) + ".lock"
    )
    stale_lock.parent.mkdir(parents=True, exist_ok=True)
    stale_lock.write_text("", encoding="utf-8")

    # Locks git owns are not ours to remove: a sweep wide enough to take these
    # could stomp on a live git process, which is worse than the bug.
    foreign_locks = [
        repository.repository_path / "packed-refs.lock",
        repository.repository_path / "refs" / "heads" / "main.lock",
    ]
    for lock in foreign_locks:
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text("", encoding="utf-8")

    await GitCandidateRepository.create(repository.root, workspace=workspace)

    assert not stale_lock.exists()
    assert all(lock.exists() for lock in foreign_locks)


@pytest.mark.asyncio
async def test_capture_survives_a_ref_lock_left_by_a_dead_writer(tmp_path: Path):
    """The next run captures normally instead of dying on a lock nobody holds."""

    source = tmp_path / "source"
    source.mkdir()
    baseline_version = _initialize(source)
    sandbox = await LocalSandbox.create(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(source))
    session_root = tmp_path / "session" / "candidates"
    repository = await GitCandidateRepository.create(session_root, workspace=workspace)

    candidate = Candidate.from_version(baseline_version, candidate_id="candidate")
    stale_lock = repository.repository_path / (
        repository._candidate_ref(candidate.id) + ".lock"
    )
    stale_lock.parent.mkdir(parents=True, exist_ok=True)
    stale_lock.write_text("", encoding="utf-8")

    resumed = await GitCandidateRepository.create(session_root, workspace=workspace)
    await resumed.capture(candidate, workspace)

    assert resumed.get(candidate.id) == candidate
